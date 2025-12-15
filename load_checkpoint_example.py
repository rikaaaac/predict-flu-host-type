"""
Example: Loading a saved ESM-2 checkpoint to explore contact maps

This script demonstrates how to load a previously trained model checkpoint
and use it to extract contact maps from new sequences without retraining.
"""

import torch
from esm2_finetuning import (
    load_model_checkpoint,
    extract_contact_map,
    extract_contact_maps_batch,
    visualize_contact_map,
    compute_group_contact_difference,
    plot_group_contact_difference,
    parse_gisaid_fasta
)


def example_load_and_predict():
    """Load checkpoint and make predictions on new sequences"""

    # path to your saved checkpoint
    checkpoint_path = '/insomnia001/depts/edu/BIOLBC3141_Fall2025/rc3517/nlp/finetune_results/esm2_finetuned_checkpoint.pt'

    print("="*80)
    print("LOADING SAVED MODEL CHECKPOINT")
    print("="*80)

    # load the checkpoint
    model, label_encoder, info = load_model_checkpoint(
        checkpoint_path=checkpoint_path,
        device='cuda'  # or 'cpu' if no GPU
    )

    # show saved metrics
    print("\nModel Performance (from saved checkpoint):")
    metrics = info['metrics']
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  Precision:    {metrics['precision_macro']:.4f}")
    print(f"  Recall:       {metrics['recall_macro']:.4f}")
    print(f"  F1-Score:     {metrics['f1_macro']:.4f}")

    # show class names
    print(f"\nClasses: {label_encoder.classes_.tolist()}")

    return model, label_encoder, info


def example_extract_single_contact_map(model):
    """Extract and visualize contact map for a single sequence"""

    print("\n" + "="*80)
    print("EXTRACTING CONTACT MAP FOR SINGLE SEQUENCE")
    print("="*80)

    # example sequence
    sequence = "MERIKELRDLMSQSRTREILTKTTVDHMAIIKKYTSGRQEKNPALRMKWMMAMKYPITADKRIMEMIPERNEQGQTLWSKMNDAGSDRVMVSPLAVTWWNRNGPTTSTVHYPKVYKTYFEKVERLKHGTFGPVHFRNQVKIRR"
    seq_id = "example_PB2_sequence"

    print(f"Sequence ID: {seq_id}")
    print(f"Length: {len(sequence)}")

    # extract contact map
    contact_map, _ = extract_contact_map(
        model=model,
        sequence=sequence,
        seq_id=seq_id,
        device='cuda'
    )

    print(f"\nContact map shape: {contact_map.shape}")
    print(f"Contact probability range: [{contact_map.min():.3f}, {contact_map.max():.3f}]")

    # visualize
    fig = visualize_contact_map(
        contact_map=contact_map,
        seq_id=seq_id,
        save_path='loaded_model_contact_map.png'
    )

    print("Contact map saved to: loaded_model_contact_map.png")

    return contact_map


def example_batch_contact_maps_from_fasta(model):
    """Load sequences from FASTA and extract contact maps in batch"""

    print("\n" + "="*80)
    print("EXTRACTING CONTACT MAPS FROM FASTA FILE")
    print("="*80)

    # load sequences
    fasta_path = "/Users/rikac/Documents/NLP/project/all_pb2_sequences_labeled_balanced.fasta"
    sequences, metadata_df = parse_gisaid_fasta(fasta_path)

    # get 5 human sequences
    human_seqs = [(seq_id, seq) for (seq_id, seq) in sequences
                  if metadata_df[metadata_df['seq_id'] == seq_id]['host_type'].values[0] == 'human'][:5]

    print(f"\nExtracting contact maps for {len(human_seqs)} human sequences...")

    # extract contact maps
    contact_maps = extract_contact_maps_batch(
        model=model,
        sequences=human_seqs,
        device='cuda',
        max_sequences=5
    )

    print(f"Extracted {len(contact_maps)} contact maps")

    # visualize first one
    if len(contact_maps) > 0:
        first_id = list(contact_maps.keys())[0]
        fig = visualize_contact_map(
            contact_maps[first_id],
            seq_id=first_id,
            save_path='batch_contact_map_sample.png'
        )
        print(f"Sample contact map saved to: batch_contact_map_sample.png")

    return contact_maps


def example_compare_host_groups(model):
    """Compare contact patterns between host groups using loaded model"""

    print("\n" + "="*80)
    print("COMPARING CONTACT PATTERNS BETWEEN HOST GROUPS")
    print("="*80)

    # load sequences
    fasta_path = "/Users/rikac/Documents/NLP/project/all_pb2_sequences_labeled_balanced.fasta"
    sequences, metadata_df = parse_gisaid_fasta(fasta_path)

    # get sequences by host type (taking subset for speed)
    human_seqs = [(seq_id, seq) for (seq_id, seq) in sequences
                  if metadata_df[metadata_df['seq_id'] == seq_id]['host_type'].values[0] == 'human'][:10]

    avian_seqs = [(seq_id, seq) for (seq_id, seq) in sequences
                  if metadata_df[metadata_df['seq_id'] == seq_id]['host_type'].values[0] == 'avian'][:10]

    print(f"\nComparing {len(human_seqs)} human vs {len(avian_seqs)} avian sequences...")

    # compute group difference
    result = compute_group_contact_difference(
        model=model,
        group1_seqs=human_seqs,
        group2_seqs=avian_seqs,
        device='cuda'
    )

    # visualize
    fig = plot_group_contact_difference(
        group_result=result,
        group1_name="Human isolates",
        group2_name="Avian isolates",
        effect_threshold=0.5,
        save_path='loaded_model_group_comparison.png'
    )

    print("Group comparison plot saved to: loaded_model_group_comparison.png")

    # print top changed residues
    print("\nTop 10 residues with largest contact changes:")
    top_residues = result['top_changed_residues'][:10]
    residue_effect = result['residue_effect']

    for rank, res_idx in enumerate(top_residues, 1):
        print(f"  {rank}. Residue {res_idx+1}: effect size = {residue_effect[res_idx]:.3f}")

    return result


def example_classify_new_sequence(model, label_encoder):
    """Use loaded model to classify a new sequence"""

    print("\n" + "="*80)
    print("CLASSIFYING NEW SEQUENCE")
    print("="*80)

    # example sequence
    sequence = "MERIKELRDLMSQSRTREILTKTTVDHMAIIKKYTSGRQEKNPALRMKWMMAMKYPITADKRIMEMIPERNEQGQTLWSKMNDAGSDRVMVSPLAVTWWNRNGPTTSTVHYPKVYKTYFEKVERLKHGTFGPVHFRNQVKIRR"
    seq_id = "new_sequence"

    print(f"Sequence: {seq_id}")
    print(f"Length: {len(sequence)}")

    # prepare input
    batch_converter = model.batch_converter
    data = [(seq_id, sequence)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(next(model.parameters()).device)

    # predict
    model.eval()
    with torch.no_grad():
        logits = model(tokens)
        probs = torch.softmax(logits, dim=1)
        pred_class = logits.argmax(1).item()

    # decode prediction
    pred_label = label_encoder.classes_[pred_class]

    print(f"\nPrediction: {pred_label}")
    print(f"Class probabilities:")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"  {class_name}: {probs[0, i].item():.4f}")

    return pred_label, probs


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    print("ESM-2 Model Checkpoint Loading Example")
    print("This demonstrates how to load and use a saved checkpoint\n")

    # Example 1: Load the checkpoint
    model, label_encoder, info = example_load_and_predict()

    # Example 2: Extract single contact map
    contact_map = example_extract_single_contact_map(model)

    # Example 3: Classify a new sequence
    pred, probs = example_classify_new_sequence(model, label_encoder)

    # Example 4: Batch extract from FASTA (optional - commented out for speed)
    # contact_maps = example_batch_contact_maps_from_fasta(model)

    # Example 5: Compare host groups (optional - commented out for speed)
    # result = example_compare_host_groups(model)

    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETE!")
    print("="*80)
    print("\nYou can now:")
    print("  1. Load your saved checkpoint anytime")
    print("  2. Extract contact maps from new sequences")
    print("  3. Classify new sequences")
    print("  4. Compare structural patterns between groups")
    print("\nNo retraining needed!")
