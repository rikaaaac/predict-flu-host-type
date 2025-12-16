import torch
import esm
import pandas as pd
import numpy as np
from Bio import SeqIO
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap

# load ESM-2 model
def load_esm2_model(model_name="esm2_t33_650M_UR50D"):
    print(f"loading {model_name}...")
    model, alphabet = esm.pretrained.__dict__[model_name]()
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"model loaded on {device}")

    return model, alphabet, device

def parse_gisaid_fasta(fasta_path):
    """
    parse the concatenated fasta file with headers
    the header should look like this:
    EPI4586304|PB2|A/dairy_cow/USA/019914-004/2025|EPI_ISL_20094668|A_/_H5N1|HOST_0

    host encoding:
    - HOST_0 = human
    - HOST_1 = avian
    - HOST_2 = mammal
    """
    # host mapping
    HOST_MAPPING = {
        'HOST_0': 'human',
        'HOST_1': 'avian',
        'HOST_2': 'mammal'
    }

    sequences = []
    metadata = []

    for record in SeqIO.parse(fasta_path, "fasta"):
        header_parts = record.description.split('|')
        seq_str = str(record.seq)

        # extract host type from last field
        if len(header_parts) >= 6:
            host_code = header_parts[5].strip()
            host_type = HOST_MAPPING.get(host_code, None)

            if host_type and len(seq_str) > 0:
                sequences.append((record.id, seq_str))
                metadata.append({
                    'seq_id': record.id,
                    'epi_id': header_parts[0],
                    'gene': header_parts[1],
                    'strain': header_parts[2],
                    'epi_isl': header_parts[3],
                    'subtype': header_parts[4],
                    'host_code': host_code,
                    'host_type': host_type,
                    'sequence_length': len(seq_str)
                })

    metadata_df = pd.DataFrame(metadata)
    print(f"\nLoaded {len(sequences)} sequences")
    print(f"\nHost distribution:")
    print(metadata_df['host_type'].value_counts())
    print(f"\nSequence length statistics:")
    print(metadata_df.groupby('host_type')['sequence_length'].describe())

    return sequences, metadata_df

sequences, metadata_df = parse_gisaid_fasta('/insomnia001/depts/edu/BIOLBC3141_Fall2025/rc3517/nlp/data/all_pb2_sequences_labeled_balanced.fasta')

def extract_esm2_embeddings(sequences, model, alphabet, device, batch_size=8):
    """
    extract only sequence embeddings
    """
    # ESM-2 batch_converter to convert tokenization to the model input format
    batch_converter = alphabet.get_batch_converter()
    sequence_embeddings = {}

    print(f"\nextracting embeddings for {len(sequences)} sequences...")

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        # convert batch to model format: labels, strings (sequence), tokens
        batch_labels, batch_strs, batch_tokens = batch_converter(batch)
        # calculate actual length of each sequence by counting non-padding tokens
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens.to(device)

        # run the model to get representation from layer 33 (final layer)
        with torch.no_grad():
            results = model(
                batch_tokens,
                repr_layers=[33],
                return_contacts=False  # False to save memory during embedding extraction
            )

        # extract the token-level representations from layer 33
        # shape: batch_size × sequence_length × embedding_dim
        token_representations = results["representations"][33]

        # loop through each sequence in the current batch to get index, ID, sequence string
        for j, (seq_id, seq) in enumerate(batch):
            tokens_len = batch_lens[j].item()
            # extract embeddings for this sequence excluding special tokens
            residue_emb = token_representations[j, 1:tokens_len-1].cpu().numpy()
            # create sequence-level embedding by averaging all amino acids embeddings
            seq_emb = residue_emb.mean(0)
            # store the mean-pooled sequence embedding in the dict with the sequence ID as key
            sequence_embeddings[seq_id] = seq_emb

        # clear GPU memory
        del batch_tokens, results, token_representations
        torch.cuda.empty_cache()

        # print progress every 40 sequences
        if (i + batch_size) % 40 == 0 or (i + batch_size) >= len(sequences):
            print(f"Processed {min(i+batch_size, len(sequences))}/{len(sequences)} sequences")

    # return the dict containing sequence embeddings for all processed sequences
    return sequence_embeddings

def train_baseline_classifier(embeddings, metadata_df, val_size=0.2, test_size=0.2, random_state=42):
    """
    train baseline classifiers on ESM-2 embeddings
    """
    # prepare data
    # extracts all sequence IDs from the embeddings dictionary (obtained from the above function)
    seq_ids = list(embeddings.keys())
    # creates a matrix X by converting embeddings to a numpy array, where each row is one sequence's embedding vector
    X = np.array([embeddings[seq_id] for seq_id in seq_ids])

    # get labels in same order
    # creates a mapping dictionary from sequence ID to host type label using the metadata (obtained from parse_gisaid_fasta function)
    label_dict = dict(zip(metadata_df['seq_id'], metadata_df['host_type']))
    # creates the label array y in the same order as the matrix X
    y = np.array([label_dict[seq_id] for seq_id in seq_ids])

    # encode labels
    # convert the host labels to numerical values
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"\nLabel encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # first split: train+val vs test
    X_temp, X_test, y_temp, y_test, ids_temp, ids_test = train_test_split(
        X, y_encoded, seq_ids, test_size=test_size, random_state=random_state, stratify=y_encoded)

    # second split: train vs val
    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X_temp, y_temp, ids_temp, test_size=val_size/(1-test_size), random_state=random_state, stratify=y_temp)

    print(f"\nTraining set: {len(X_train)}, Val set: {len(X_val)}, Test set: {len(X_test)}")
    print(f"Training distribution: {np.bincount(y_train)}")
    print(f"Validation distribution: {np.bincount(y_val)}")
    print(f"Test distribution: {np.bincount(y_test)}")

    # train multiple baseline classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced')
    }

    results = {}

    for name, clf in classifiers.items():
        print(f"\n{'='*60}")
        print(f"Training {name}...")
        clf.fit(X_train, y_train)

        # predictions
        y_pred_train = clf.predict(X_train)
        y_pred_val = clf.predict(X_val)
        y_pred = clf.predict(X_test)

        # metrics
        train_acc = accuracy_score(y_train, y_pred_train)
        val_acc = accuracy_score(y_val, y_pred_val)
        test_acc = accuracy_score(y_test, y_pred)

        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Validation accuracy: {val_acc:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")
        print(f"\nClassification report (Test Set):")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        # confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
          'model': clf,
          'train_accuracy': train_acc,
          'val_accuracy': val_acc,
          'test_accuracy': test_acc,
          'y_train': y_train,
          'y_val': y_val,
          'y_test': y_test,
          'y_pred_train': y_pred_train,
          'y_pred_val': y_pred_val,
          'y_pred': y_pred,
          'confusion_matrix': cm,
          'test_ids': ids_test,
          'train_ids': ids_train,
          'val_ids': ids_val
      }

    return results, le, (X_train, X_val, X_test, y_train, y_val, y_test, ids_train, ids_val, ids_test)

def extract_contact_map_baseline(
    sequence: str,
    seq_id: str,
    model,
    alphabet,
    device
):
    """
    extract contact map from ESM-2 model for a single sequence

    args:
        sequence: protein sequence string
        seq_id: sequence identifier
        model: ESM-2 model
        alphabet: ESM-2 alphabet
        device: device to run on

    returns:
        contact_map: [seq_len, seq_len] contact probability matrix
    """
    batch_converter = alphabet.get_batch_converter()
    data = [(seq_id, sequence)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    with torch.no_grad():
        results = model(
            tokens,
            repr_layers=[33],
            return_contacts=True
        )

    contacts = results["contacts"]
    contact_map = contacts[0].cpu().numpy()

    # trim to actual sequence length (remove special tokens)
    seq_len = len(sequence)
    contact_map = contact_map[1:seq_len+1, 1:seq_len+1]

    return contact_map

def compute_group_contact_difference_baseline(
    model,
    alphabet,
    device,
    group1_seqs: list,  # [(id, seq), ...]
    group2_seqs: list,
    metadata_df
):
    """
    Compare average contact maps between two groups of sequences.

    Args:
        model: ESM-2 model
        alphabet: ESM-2 alphabet
        device: device to run on
        group1_seqs: List of (id, sequence) tuples for group 1
        group2_seqs: List of (id, sequence) tuples for group 2
        metadata_df: metadata dataframe

    Returns:
        Dict with mean contact maps, difference, and statistics
    """
    maps1 = []
    maps2 = []

    max_len = 0

    print(f"extracting contact maps for group 1 ({len(group1_seqs)} sequences)...")
    for i, (seq_id, seq) in enumerate(group1_seqs):
        cmap = extract_contact_map_baseline(seq, seq_id, model, alphabet, device)
        maps1.append(cmap)
        max_len = max(max_len, cmap.shape[0])
        if (i + 1) % 10 == 0:
            print(f"  processed {i+1}/{len(group1_seqs)}")

    print(f"extracting contact maps for group 2 ({len(group2_seqs)} sequences)...")
    for i, (seq_id, seq) in enumerate(group2_seqs):
        cmap = extract_contact_map_baseline(seq, seq_id, model, alphabet, device)
        maps2.append(cmap)
        max_len = max(max_len, cmap.shape[0])
        if (i + 1) % 10 == 0:
            print(f"  processed {i+1}/{len(group2_seqs)}")

    # pad all maps to same size
    def pad_map(m, target_size):
        if m.shape[0] < target_size:
            pad = target_size - m.shape[0]
            return np.pad(m, ((0, pad), (0, pad)), mode='constant')
        return m

    maps1 = np.stack([pad_map(m, max_len) for m in maps1])
    maps2 = np.stack([pad_map(m, max_len) for m in maps2])

    # compute statistics
    mean1 = np.mean(maps1, axis=0)
    mean2 = np.mean(maps2, axis=0)
    std1 = np.std(maps1, axis=0)
    std2 = np.std(maps2, axis=0)

    difference = mean2 - mean1

    # compute effect size (Cohen's d style)
    pooled_std = np.sqrt((std1**2 + std2**2) / 2 + 1e-8)
    effect_size = difference / pooled_std

    # per-residue summary
    residue_effect = np.mean(np.abs(effect_size), axis=1)
    top_residues = np.argsort(residue_effect)[::-1]

    return {
        'mean_contact_map_group1': mean1,
        'mean_contact_map_group2': mean2,
        'difference': difference,
        'effect_size': effect_size,
        'residue_effect': residue_effect,
        'top_changed_residues': top_residues,
        'n_group1': len(group1_seqs),
        'n_group2': len(group2_seqs)
    }

def plot_group_contact_difference(
    group_result,
    group1_name: str = "Group 1",
    group2_name: str = "Group 2",
    effect_threshold: float = 0.5,
    save_path = None
):
    """
    Visualize group-level contact map comparison with effect sizes.

    Args:
        group_result: Output from compute_group_contact_difference_baseline()
        group1_name: Label for first group
        group2_name: Label for second group
        effect_threshold: Effect size threshold for significance
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(18, 10))

    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.8], hspace=0.3, wspace=0.3)

    mean1 = group_result['mean_contact_map_group1']
    mean2 = group_result['mean_contact_map_group2']
    effect_size = group_result['effect_size']
    residue_effect = group_result['residue_effect']

    seq_len = mean1.shape[0]

    # top row: contact maps and difference
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(mean1, cmap='Blues', vmin=0, vmax=1)
    ax1.set_title(f"{group1_name}\nMean Contact Map (n={group_result['n_group1']})", fontsize=11)
    ax1.set_xlabel("Residue")
    ax1.set_ylabel("Residue")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(mean2, cmap='Blues', vmin=0, vmax=1)
    ax2.set_title(f"{group2_name}\nMean Contact Map (n={group_result['n_group2']})", fontsize=11)
    ax2.set_xlabel("Residue")
    ax2.set_ylabel("Residue")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(gs[0, 2])
    max_effect = max(abs(effect_size.min()), abs(effect_size.max()))
    im3 = ax3.imshow(effect_size, cmap='RdBu_r', vmin=-max_effect, vmax=max_effect)
    ax3.set_title(f"Effect Size (Cohen's d)\n({group2_name} − {group1_name})", fontsize=11)
    ax3.set_xlabel("Residue")
    ax3.set_ylabel("Residue")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # bottom row: per-residue analysis
    ax4 = fig.add_subplot(gs[1, :2])
    x = np.arange(seq_len)
    colors = ['#d62728' if v > effect_threshold else '#1f77b4' for v in residue_effect]
    ax4.bar(x, residue_effect, color=colors, width=1.0, edgecolor='none')
    ax4.set_xlabel("Residue Position", fontsize=11)
    ax4.set_ylabel("Mean |Effect Size|", fontsize=11)
    ax4.set_title("Per-Residue Contact Environment Change", fontsize=12)
    ax4.axhline(y=effect_threshold, color='red', linestyle='--', alpha=0.7,
                label=f'Threshold (|d|={effect_threshold})')
    ax4.set_xlim(0, seq_len)
    ax4.legend(loc='upper right')

    # bottom right: top changed residues table
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')

    top_n = 15
    top_residues = group_result['top_changed_residues'][:top_n]

    table_data = []
    for rank, res in enumerate(top_residues, 1):
        effect = residue_effect[res]
        table_data.append([rank, res + 1, f"{effect:.3f}"])

    table = ax5.table(
        cellText=table_data,
        colLabels=['Rank', 'Position', '|Effect|'],
        cellLoc='center',
        loc='center',
        colWidths=[0.2, 0.3, 0.3]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax5.set_title("Top Changed Residues", fontsize=12, pad=20)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"saved contact difference plot to {save_path}")

    return fig

def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    """plot confusion matrix"""
    plt.figure(figsize=(8, 6))

    # normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})

    # add normalized values as text
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j+0.5, i+0.7, f'({cm_norm[i,j]:.2f})',
                    ha='center', va='center', fontsize=9, color='gray')

    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return plt.gcf()

def extract_attention(sequences, metadata_df, model, alphabet, device,
                                 n_per_host=100, batch_size=1):
    """
    extract attention weights
    (this is only for visualization, not for training)
    """
    attention_weights = {}

    # sample sequences from each host type
    sampled_seqs = []
    for host_type in ['human', 'avian', 'mammal']:
        host_seqs = metadata_df[metadata_df['host_type'] == host_type]['seq_id'].values
        n_sample = min(n_per_host, len(host_seqs))
        sampled_ids = np.random.choice(host_seqs, n_sample, replace=False)

        for seq_id in sampled_ids:
            seq = next(seq for sid, seq in sequences if sid == seq_id)
            sampled_seqs.append((seq_id, seq))

    print(f"\nExtracting attention for {len(sampled_seqs)} sequences (subset)...")

    batch_converter = alphabet.get_batch_converter()

    # process one at a time
    for i, (seq_id, seq) in enumerate(sampled_seqs):
        batch = [(seq_id, seq)]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(
                batch_tokens,
                repr_layers=[33],
                return_contacts=True
            )

        tokens_len = batch_lens[0].item()
        attn = results["contacts"][0, 1:tokens_len-1, 1:tokens_len-1].cpu().numpy()
        attention_weights[seq_id] = attn

        # memory cleanup
        del batch_tokens, results
        torch.cuda.empty_cache()

        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(sampled_seqs)} sequences")

    return attention_weights, sampled_seqs

def plot_representative_contact_maps(attention_weights, sampled_seqs, seq_to_host,
                                   output_dir, n_examples=3):
    """plot individual contact maps for representative sequences from each host type"""

    # group sequences by host type
    host_seqs = {'human': [], 'avian': [], 'mammal': []}
    for seq_id, seq in sampled_seqs:
        host_type = seq_to_host[seq_id]
        host_seqs[host_type].append((seq_id, seq))

    fig, axes = plt.subplots(3, n_examples, figsize=(4*n_examples, 12))
    if n_examples == 1:
        axes = axes.reshape(-1, 1)

    for row, (host_type, seqs) in enumerate(host_seqs.items()):
        # sample random examples
        sample_seqs = np.random.choice(len(seqs), min(n_examples, len(seqs)), replace=False)

        for col, idx in enumerate(sample_seqs):
            seq_id, seq = seqs[idx]
            attention = attention_weights[seq_id]

            im = axes[row, col].imshow(attention, cmap='viridis', aspect='equal')
            axes[row, col].set_title(f'{host_type.title()}\n{seq_id[:15]}...', fontsize=10)
            axes[row, col].set_xlabel('Position')
            axes[row, col].set_ylabel('Position')

            # add colorbar
            plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)

    # hide unused subplots
    for row in range(3):
        for col in range(n_examples):
            if col >= len(host_seqs[list(host_seqs.keys())[row]]):
                axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'representative_contact_maps.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_average_contact_maps(attention_weights, sampled_seqs, seq_to_host, output_dir):
    """plot average contact maps for each host type"""

    # group and average by host type
    host_attention = {'human': [], 'avian': [], 'mammal': []}

    for seq_id, seq in sampled_seqs:
        host_type = seq_to_host[seq_id]
        attention = attention_weights[seq_id]
        host_attention[host_type].append(attention)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, (host_type, attentions) in enumerate(host_attention.items()):
        if len(attentions) > 0:
            # find common size (minimum dimensions)
            min_size = min(att.shape[0] for att in attentions)

            # crop all attention maps to same size and average
            cropped_attentions = [att[:min_size, :min_size] for att in attentions]
            avg_attention = np.mean(cropped_attentions, axis=0)

            im = axes[i].imshow(avg_attention, cmap='viridis', aspect='equal')
            axes[i].set_title(f'{host_type.title()} (n={len(attentions)})')
            axes[i].set_xlabel('Position')
            axes[i].set_ylabel('Position')

            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        else:
            axes[i].axis('off')
            axes[i].set_title(f'{host_type.title()} (no data)')

    plt.tight_layout()
    plt.savefig(output_dir / 'average_contact_maps_by_host.png', dpi=300, bbox_inches='tight')
    plt.close()

def main(fasta_path, output_dir='baseline_results', do_extract_attention_visuals=True, n_attention_samples=100):
    """
    args:
        do_extract_attention_visuals: if True, extract attention for subset
        n_attention_samples: # of sequences per host to analyze attention
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("="*80)
    print("ESM-2 BASELINE HOST PREDICTION PIPELINE")
    print("="*80)

    # 1. load model
    model, alphabet, device = load_esm2_model()

    # 2. load sequences
    print("\n" + "="*80)
    print("LOADING SEQUENCES")
    print("="*80)
    sequences, metadata_df = parse_gisaid_fasta(fasta_path)
    metadata_df.to_csv(output_dir / 'metadata.csv', index=False)

    # 3. extract embeddings
    print("\n" + "="*80)
    print("EXTRACTING ESM-2 EMBEDDINGS")
    print("="*80)
    sequence_embeddings = extract_esm2_embeddings(
        sequences, model, alphabet, device, batch_size=8
    )

    print("\nSaving embeddings...")
    np.savez_compressed(output_dir / 'esm2_sequence_embeddings.npz',
                       **{k: v for k, v in sequence_embeddings.items()})

    # 4. train classifiers
    print("\n" + "="*80)
    print("TRAINING BASELINE CLASSIFIERS")
    print("="*80)

    results, label_encoder, split_data = train_baseline_classifier(
        sequence_embeddings, metadata_df
    )

    for name, result in results.items():
        fig = plot_confusion_matrix(
            result['confusion_matrix'],
            label_encoder.classes_,
            title=f'{name} - Test Set'
        )
        fig.savefig(output_dir / f'confusion_matrix_{name.replace(" ", "_").lower()}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

    best_model_name = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
    print(f"\nBest model: {best_model_name} (Acc: {results[best_model_name]['test_accuracy']:.4f})")

    # 5. extract contact maps for group comparisons
    print("\n" + "="*80)
    print("EXTRACTING CONTACT MAPS FOR GROUP COMPARISONS")
    print("="*80)

    # get sequences by host type
    human_seqs = [(seq_id, seq) for (seq_id, seq) in sequences
                  if metadata_df[metadata_df['seq_id'] == seq_id]['host_type'].values[0] == 'human']

    avian_seqs = [(seq_id, seq) for (seq_id, seq) in sequences
                  if metadata_df[metadata_df['seq_id'] == seq_id]['host_type'].values[0] == 'avian']

    mammal_seqs = [(seq_id, seq) for (seq_id, seq) in sequences
                   if metadata_df[metadata_df['seq_id'] == seq_id]['host_type'].values[0] == 'mammal']

    # compute group contact differences
    print("\nComputing human vs avian contact maps...")
    human_avian = compute_group_contact_difference_baseline(
        model, alphabet, device,
        human_seqs, avian_seqs, metadata_df
    )

    print("\nComputing human vs mammal contact maps...")
    human_mammal = compute_group_contact_difference_baseline(
        model, alphabet, device,
        human_seqs, mammal_seqs, metadata_df
    )

    print("\nComputing avian vs mammal contact maps...")
    avian_mammal = compute_group_contact_difference_baseline(
        model, alphabet, device,
        avian_seqs, mammal_seqs, metadata_df
    )

    # create contact map visualizations directory
    contact_map_dir = output_dir / 'contact_maps'
    contact_map_dir.mkdir(exist_ok=True)

    # plot group contact differences
    print("\nGenerating contact map visualizations...")
    plot_group_contact_difference(
        human_avian,
        group1_name="Human",
        group2_name="Avian",
        save_path=contact_map_dir / 'human_avian_group_contact_diff.png'
    )

    plot_group_contact_difference(
        human_mammal,
        group1_name="Human",
        group2_name="Other Mammals",
        save_path=contact_map_dir / 'human_mammal_group_contact_diff.png'
    )

    plot_group_contact_difference(
        avian_mammal,
        group1_name="Avian",
        group2_name="Other Mammals",
        save_path=contact_map_dir / 'avian_mammal_group_contact_diff.png'
    )

    # save contact map results
    print("\nSaving contact map results...")
    np.savez_compressed(
        contact_map_dir / 'contact_map_results.npz',
        human_avian_mean1=human_avian['mean_contact_map_group1'],
        human_avian_mean2=human_avian['mean_contact_map_group2'],
        human_avian_diff=human_avian['difference'],
        human_mammal_mean1=human_mammal['mean_contact_map_group1'],
        human_mammal_mean2=human_mammal['mean_contact_map_group2'],
        human_mammal_diff=human_mammal['difference'],
        avian_mammal_mean1=avian_mammal['mean_contact_map_group1'],
        avian_mammal_mean2=avian_mammal['mean_contact_map_group2'],
        avian_mammal_diff=avian_mammal['difference']
    )

    # 6. extract attention for subset (original code)
    if do_extract_attention_visuals:
        print("\n" + "="*80)
        print(f"EXTRACTING ATTENTION FOR SUBSET ({n_attention_samples} per host)")
        print("="*80)

        attention_weights, sampled_seqs = extract_attention(
            sequences, metadata_df, model, alphabet, device,
            n_per_host=n_attention_samples, batch_size=1
        )

        # save attention weights
        np.savez_compressed(output_dir / 'attention_subset.npz',
                           **{k: v for k, v in attention_weights.items()})

        # 6. plot contact maps
        print("\n" + "="*80)
        print("GENERATING CONTACT MAP VISUALIZATIONS")
        print("="*80)

        contact_map_dir = output_dir / 'contact_maps'
        contact_map_dir.mkdir(exist_ok=True)

        # create host type mapping for sampled sequences
        seq_to_host = dict(zip(metadata_df['seq_id'], metadata_df['host_type']))

        # plot representative contact maps for each host type
        plot_representative_contact_maps(
            attention_weights, sampled_seqs, seq_to_host,
            contact_map_dir, n_examples=3
        )

        # plot average contact maps by host type
        plot_average_contact_maps(
            attention_weights, sampled_seqs, seq_to_host,
            contact_map_dir
        )

    # 7. save summary
    summary = {
        'n_sequences': len(sequences),
        'n_human': len(metadata_df[metadata_df['host_type'] == 'human']),
        'n_avian': len(metadata_df[metadata_df['host_type'] == 'avian']),
        'n_mammal': len(metadata_df[metadata_df['host_type'] == 'mammal']),
        'best_model': best_model_name,
        'best_accuracy': results[best_model_name]['test_accuracy'],
        'lr_accuracy': results['Logistic Regression']['test_accuracy'],
        'rf_accuracy': results['Random Forest']['test_accuracy'],
        'contact_maps_extracted': True,
        'human_avian_top_residue': int(human_avian['top_changed_residues'][0]),
        'human_mammal_top_residue': int(human_mammal['top_changed_residues'][0]),
        'avian_mammal_top_residue': int(avian_mammal['top_changed_residues'][0])
    }

    with open(output_dir / 'summary.txt', 'w') as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE")
    print(f"Results saved to: {output_dir}")
    print(f"Contact map visualizations saved to: {contact_map_dir}")
    print(f"{'='*80}")

    return results, sequence_embeddings

if __name__ == "__main__":
    fasta_path = '/insomnia001/depts/edu/BIOLBC3141_Fall2025/rc3517/nlp/data/all_pb2_sequences_labeled_balanced.fasta'
    output_dir = '/insomnia001/depts/edu/BIOLBC3141_Fall2025/rc3517/nlp/baseline_results'

    results, embeddings = main(
        fasta_path,
        output_dir=output_dir,
        do_extract_attention_visuals=True,  # can set to False to skip attention entirely
        n_attention_samples=100   # sample for attention analysis
    )