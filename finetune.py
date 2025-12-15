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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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

def load_embeddings(filepath):
    """
    Load sequence embeddings from a .npz file
    """
    data = np.load(filepath, allow_pickle=True)
    if 'embeddings' in data.files:
        return data['embeddings'].item()
    return {key: data[key] for key in data.files}

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

# fine tune
class ESM2Dataset(Dataset):
    """Dataset for fine-tuning ESM-2"""
    def __init__(self, sequences, labels, batch_converter):
        self.sequences = sequences
        self.labels = labels
        self.batch_converter = batch_converter
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_id, seq = self.sequences[idx]
        label = self.labels[idx]
        return seq_id, seq, label

def collate_fn(batch, batch_converter):
    """Custom collate function for ESM-2 data"""
    seq_ids, seqs, class_labels = zip(*batch)
    # Convert sequences to tokens - batch_converter returns (labels, strs, tokens)
    # where 'labels' are just the seq_ids we passed in
    _, _, tokens = batch_converter(list(zip(seq_ids, seqs)))
    # Use our actual classification labels, not the seq_ids from batch_converter
    return tokens, torch.tensor(class_labels, dtype=torch.long)

class ESM2FineTuned(nn.Module):
    def __init__(self, num_classes, model_name="esm2_t33_650M_UR50D",
                 freeze_layers=0, dropout=0.3):
        super().__init__()
        self.esm, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.batch_converter = self.alphabet.get_batch_converter()
        
        # Freeze early layers, fine-tune later ones
        for name, param in self.esm.named_parameters():
            layer_num = self._get_layer_num(name)
            if layer_num is not None and layer_num < freeze_layers:
                param.requires_grad = False
        
        embed_dim = self.esm.embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def _get_layer_num(self, name):
        if "layers." in name:
            try:
                return int(name.split("layers.")[1].split(".")[0])
            except (IndexError, ValueError):
                return None
        return None
    
    def forward(self, tokens):
        # Don't use torch.no_grad() here since we want gradients for training
        results = self.esm(tokens, repr_layers=[33], return_contacts=False)
        
        # Mean pooling over sequence length (excluding special tokens)
        embeddings = results["representations"][33]
        # Get actual sequence lengths
        seq_lens = (tokens != self.alphabet.padding_idx).sum(1)
        
        pooled_embeddings = []
        for i, seq_len in enumerate(seq_lens):
            # Average over actual sequence (excluding special tokens)
            if seq_len > 2:  # Make sure we have actual sequence tokens
                seq_emb = embeddings[i, 1:seq_len-1].mean(0)
            else:
                # Fallback for very short sequences
                seq_emb = embeddings[i].mean(0)
            pooled_embeddings.append(seq_emb)
        
        pooled_embeddings = torch.stack(pooled_embeddings)
        return self.classifier(pooled_embeddings)

def train_esm2_finetuned(sequences, metadata_df,
                        val_size=0.2, test_size=0.2, random_state=42,
                        model_name="esm2_t33_650M_UR50D", freeze_layers=0,
                        lr=1e-4, batch_size=4, num_epochs=10, device=None):
    """
    Fine-tune ESM-2 for host classification
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare data
    seq_ids = [seq_id for seq_id, _ in sequences]
    
    # Get labels in same order
    label_dict = dict(zip(metadata_df['seq_id'], metadata_df['host_type']))
    y = np.array([label_dict[seq_id] for seq_id in seq_ids])
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"\nLabel encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Split data
    seq_temp, seq_test, y_temp, y_test = train_test_split(
        sequences, y_encoded, test_size=test_size, random_state=random_state, 
        stratify=y_encoded)
    
    seq_train, seq_val, y_train, y_val = train_test_split(
        seq_temp, y_temp, test_size=val_size/(1-test_size), 
        random_state=random_state, stratify=y_temp)
    
    print(f"\nTraining set: {len(seq_train)}, Val set: {len(seq_val)}, Test set: {len(seq_test)}")
    
    # Create model
    num_classes = len(le.classes_)
    model = ESM2FineTuned(num_classes, model_name, freeze_layers).to(device)
    
    # Create datasets and dataloaders
    train_dataset = ESM2Dataset(seq_train, y_train, model.batch_converter)
    val_dataset = ESM2Dataset(seq_val, y_val, model.batch_converter)
    test_dataset = ESM2Dataset(seq_test, y_test, model.batch_converter)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=lambda x: collate_fn(x, model.batch_converter))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          collate_fn=lambda x: collate_fn(x, model.batch_converter))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                           collate_fn=lambda x: collate_fn(x, model.batch_converter))
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    best_val_acc = 0.0
    best_state_dict = None
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_tokens, batch_labels in train_loader:
            batch_tokens = batch_tokens.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_tokens)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted = outputs.argmax(1)
            correct += (predicted == batch_labels).sum().item()
            total += batch_labels.size(0)
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        
        # Validation
        model.eval()
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for batch_tokens, batch_labels in val_loader:
                batch_tokens = batch_tokens.to(device)
                batch_labels = batch_labels.to(device)
                outputs = model(batch_tokens)
                predicted = outputs.argmax(1)
                correct_val += (predicted == batch_labels).sum().item()
                total_val += batch_labels.size(0)
        
        val_acc = correct_val / total_val
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = model.state_dict().copy()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, "
              f"Val Acc: {val_acc:.4f}")
    
    # Load best model and evaluate
    model.load_state_dict(best_state_dict)
    
    def evaluate_loader(model, loader):
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_tokens, batch_labels in loader:
                batch_tokens = batch_tokens.to(device)
                batch_labels = batch_labels.to(device)
                outputs = model(batch_tokens)
                predicted = outputs.argmax(1).cpu().numpy()
                all_preds.append(predicted)
                all_labels.append(batch_labels.cpu().numpy())
        return np.concatenate(all_preds), np.concatenate(all_labels)
    
    y_pred_train, y_train_actual = evaluate_loader(model, train_loader)
    y_pred_val, y_val_actual = evaluate_loader(model, val_loader)
    y_pred_test, y_test_actual = evaluate_loader(model, test_loader)
    
    train_acc = accuracy_score(y_train_actual, y_pred_train)
    val_acc = accuracy_score(y_val_actual, y_pred_val)
    test_acc = accuracy_score(y_test_actual, y_pred_test)
    
    print(f"\nFinal Training accuracy: {train_acc:.4f}")
    print(f"Final Validation accuracy: {val_acc:.4f}")
    print(f"Final Test accuracy: {test_acc:.4f}")
    
    cm = confusion_matrix(y_test_actual, y_pred_test)
    
    results = {
        "ESM-2 Fine-tuned": {
            "model": model,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "test_accuracy": test_acc,
            "y_train": y_train_actual,
            "y_val": y_val_actual,
            "y_test": y_test_actual,
            "y_pred_train": y_pred_train,
            "y_pred_val": y_pred_val,
            "y_pred": y_pred_test,
            "confusion_matrix": cm,
        }
    }
    
    return results, le

def main(fasta_path, output_dir='finetune_results', do_extract_attention_visuals=False, n_attention_samples=100):

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("="*80)
    print("ESM-2 FINE-TUNING HOST PREDICTION PIPELINE")
    print("="*80)

    # 1. Load sequences
    print("\n" + "="*80)
    print("LOADING SEQUENCES")
    print("="*80)
    sequences, metadata_df = parse_gisaid_fasta(fasta_path)
    metadata_df.to_csv(output_dir / 'metadata.csv', index=False)

    # 2. Train fine-tuned model
    print("\n" + "="*80)
    print("FINE-TUNING ESM-2")
    print("="*80)

    results, label_encoder = train_esm2_finetuned(sequences, metadata_df)

    # 3. Save results
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

    return results, sequences

if __name__ == "__main__":
    fasta_path = '/insomnia001/depts/edu/BIOLBC3141_Fall2025/rc3517/nlp/data/all_pb2_sequences_labeled_balanced.fasta'
    
    results, sequences = main(
        fasta_path,
        output_dir='/insomnia001/depts/edu/BIOLBC3141_Fall2025/rc3517/nlp/finetune_results',
        do_extract_attention_visuals=False,
        n_attention_samples=100
    )