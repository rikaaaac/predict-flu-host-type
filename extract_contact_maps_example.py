"""
Example script to extract contact maps from ESM-2

Contact maps predict which amino acid residues are likely to be in
physical contact in the 3D protein structure.
"""

import torch
import esm
from esm2_finetuning import (
    extract_contact_map,
    extract_contact_maps_batch,
    visualize_contact_map,
    compare_contact_maps,
    parse_gisaid_fasta
)

# ============================================================================
# Example 1: Extract contact map for a single sequence
# ============================================================================

def example_single_sequence():
    """extract contact map for a single protein sequence"""

    print("="*80)
    print("EXAMPLE 1: Single Sequence Contact Map")
    print("="*80)

    # load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    # add batch_converter as attribute (needed for extract_contact_map)
    model.batch_converter = batch_converter
    model.esm = model  # reference to self
    model.repr_layer = 33  # last layer for this model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # example sequence (PB2 fragment)
    sequence = "MERIKELRDLMSQSRTREILTKTTVDHMAIIKKYTSGRQEKNP"
    seq_id = "example_pb2"

    print(f"\nsequence: {sequence}")
    print(f"length: {len(sequence)}")

    # extract contact map
    print("\nextracting contact map...")
    contact_map, _ = extract_contact_map(model, sequence, seq_id, device)

    print(f"contact map shape: {contact_map.shape}")
    print(f"contact probability range: [{contact_map.min():.3f}, {contact_map.max():.3f}]")

    # visualize
    print("\nvisualizing contact map...")
    fig = visualize_contact_map(
        contact_map,
        seq_id=seq_id,
        save_path='contact_map_single.png'
    )

    # also show with threshold
    fig2 = visualize_contact_map(
        contact_map,
        seq_id=seq_id,
        threshold=0.5,  # show only high-confidence contacts
        save_path='contact_map_single_threshold.png'
    )

    print("done!")


# ============================================================================
# Example 2: Extract contact maps from FASTA file
# ============================================================================

def example_from_fasta():
    """extract contact maps from sequences in FASTA file"""

    print("\n" + "="*80)
    print("EXAMPLE 2: Contact Maps from FASTA File")
    print("="*80)

    # load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    # add necessary attributes
    model.batch_converter = batch_converter
    model.esm = model
    model.repr_layer = 33

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # load sequences from FASTA
    fasta_path = "/Users/rikac/Documents/NLP/project/all_pb2_sequences_labeled_balanced.fasta"
    print(f"\nloading sequences from {fasta_path}...")
    sequences, metadata_df = parse_gisaid_fasta(fasta_path)

    # extract contact maps for first 5 sequences
    print(f"\nextracting contact maps for first 5 sequences...")
    contact_maps = extract_contact_maps_batch(
        model,
        sequences[:5],
        device=device,
        max_sequences=5
    )

    print(f"extracted {len(contact_maps)} contact maps")

    # visualize first 3
    seq_ids = list(contact_maps.keys())[:3]
    fig = compare_contact_maps(
        contact_maps,
        seq_ids,
        figsize=(18, 6),
        save_path='contact_maps_comparison.png'
    )

    print("done!")


# ============================================================================
# Example 3: Compare contacts between different host types
# ============================================================================

def example_compare_hosts():
    """compare contact patterns between human, avian, and mammal sequences"""

    print("\n" + "="*80)
    print("EXAMPLE 3: Compare Contact Maps by Host Type")
    print("="*80)

    # load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    model.batch_converter = batch_converter
    model.esm = model
    model.repr_layer = 33

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # load sequences
    fasta_path = "/Users/rikac/Documents/NLP/project/all_pb2_sequences_labeled_balanced.fasta"
    sequences, metadata_df = parse_gisaid_fasta(fasta_path)

    # get one sequence from each host type
    contact_maps_by_host = {}

    for host_type in ['human', 'avian', 'mammal']:
        # find first sequence of this host type
        host_seqs = metadata_df[metadata_df['host_type'] == host_type]
        if len(host_seqs) > 0:
            seq_id = host_seqs.iloc[0]['seq_id']
            # find the sequence
            seq = next(s for sid, s in sequences if sid == seq_id)

            print(f"\nextracting contact map for {host_type} sequence: {seq_id}")
            contact_map, _ = extract_contact_map(model, seq, seq_id, device)
            contact_maps_by_host[f"{host_type}_{seq_id}"] = contact_map

    # compare side by side
    if len(contact_maps_by_host) > 0:
        fig = compare_contact_maps(
            contact_maps_by_host,
            list(contact_maps_by_host.keys()),
            figsize=(20, 6),
            save_path='contact_maps_by_host.png'
        )
        print("\nsaved comparison figure!")

    print("done!")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # run examples
    print("ESM-2 Contact Map Extraction Examples\n")

    # choose which example to run:

    # example 1: single sequence
    example_single_sequence()

    # example 2: from FASTA file
    # example_from_fasta()

    # example 3: compare by host type
    # example_compare_hosts()

    # show all plots
    plt.show()
