import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, List, Optional, Sequence, Union
from dataclasses import dataclass
import esm
import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random


# ============================================================================
# loss reduction method
# ============================================================================

@dataclass
class LossDict:
    """container for loss values"""
    avg: torch.Tensor


class ClassificationLossReduction:
    """
    - categorical cross-entropy for loss
    - computes cross-entropy loss for multi-class classification tasks and provides
    reduction across micro-batches for distributed training scenarios.
    """

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        forward_out: torch.Tensor
    ) -> Tuple[torch.Tensor, LossDict]:
        """
        compute cross-entropy loss for classification.

        args:
            batch: Dictionary containing 'labels' with shape [batch_size] or [batch_size, 1]
            forward_out: Model logits with shape [batch_size, num_classes]

        returns:
            Tuple of (loss_tensor, loss_dict) where loss_dict contains the averaged loss
        """
        targets = batch["labels"]  # [batch_size] or [batch_size, 1]

        # ensure targets are 1D with class indices (long type)
        if targets.dim() == 2:
            targets = targets.squeeze(-1)  # [batch_size]
        targets = targets.long()

        logits = forward_out  # [batch_size, num_classes]
        loss = F.cross_entropy(logits, targets)

        return loss, LossDict(avg=loss)

    def reduce(self, losses_reduced_per_micro_batch: Sequence[LossDict]) -> torch.Tensor:
        """
        reduce losses across micro-batches.

        args:
            losses_reduced_per_micro_batch: Sequence of LossDict from each micro-batch

        returns:
            Mean loss across all micro-batches
        """
        losses = torch.stack([loss.avg for loss in losses_reduced_per_micro_batch])
        return losses.mean()


# ============================================================================
# fine-tune model head
# ============================================================================

class ClassificationHead(nn.Module):
    """
    - downstream task head for protein sequence classification.
    - takes pooled protein embeddings and maps them to class logits through
    a multi-layer perceptron with dropout regularization.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.2
    ):
        """
        - initialize classification head.
        - args:
            input_dim: Dimension of input embeddings (ESM-2 embed_dim)
            num_classes: Number of output classes
            hidden_dim: Dimension of hidden layer
            dropout: Dropout probability for regularization
        """
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes)
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        forward pass through classification head.

        args:
            embeddings: Pooled sequence embeddings [batch_size, input_dim]

        returns:
            Logits for each class [batch_size, num_classes]
        """
        return self.classifier(embeddings)


# ============================================================================
# fine-tune model
# ============================================================================

class ESM2FineTuned(nn.Module):
    """
    - complete fine-tuned model combining ESM-2 encoder with classification head
    - this model loads a pre-trained ESM-2 encoder, optionally freezes early layers,
    and adds a classification head for supervised fine-tuning.
    """

    def __init__(
        self,
        num_classes: int,
        model_name: str = "esm2_t33_650M_UR50D",
        freeze_layers: int = 0,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        pooling_strategy: str = 'mean'
    ):
        """
        - initialize ESM-2 fine-tuned model
        - args:
            num_classes: Number of output classes
            model_name: Name of pre-trained ESM-2 model to load
            freeze_layers: Number of initial transformer layers to freeze (0 = fine-tune all)
            hidden_dim: Hidden dimension for classification head
            dropout: Dropout probability
            pooling_strategy: Strategy for pooling sequence embeddings ('mean', 'cls', 'max')
        """
        super().__init__()

        # load pre-trained ESM-2 model
        self.esm, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.pooling_strategy = pooling_strategy

        # determine which representation layer to use (last layer)
        self.repr_layer = self._get_num_layers()

        # freeze early layers if specified
        self._freeze_layers(freeze_layers)

        # initialize classification head
        embed_dim = self.esm.embed_dim
        self.head = ClassificationHead(
            input_dim=embed_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

    def _get_num_layers(self) -> int:
        """get the number of transformer layers in the ESM-2 model."""
        return len(self.esm.layers)

    def _freeze_layers(self, freeze_layers: int):
        """
        - freeze parameters in early transformer layers.
        - args:
            freeze_layers: Number of layers to freeze from the beginning
        """
        if freeze_layers > 0:
            for name, param in self.esm.named_parameters():
                layer_num = self._get_layer_num(name)
                if layer_num is not None and layer_num < freeze_layers:
                    param.requires_grad = False

    def _get_layer_num(self, name: str) -> Optional[int]:
        """extract layer number from parameter name."""
        if "layers." in name:
            try:
                return int(name.split("layers.")[1].split(".")[0])
            except (IndexError, ValueError):
                return None
        return None

    def _pool_embeddings(
        self,
        embeddings: torch.Tensor,
        tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        - pool sequence embeddings according to specified strategy.
        - args:
            embeddings: Token embeddings [batch_size, seq_len, embed_dim]
            tokens: Token IDs [batch_size, seq_len]
        - returns:
            Pooled embeddings [batch_size, embed_dim]
        """
        if self.pooling_strategy == "cls":
            # use CLS token (first token)
            return embeddings[:, 0, :]

        elif self.pooling_strategy == "mean":
            # mean pooling over sequence (excluding special tokens)
            seq_lens = (tokens != self.alphabet.padding_idx).sum(1)
            pooled = []

            for i, seq_len in enumerate(seq_lens):
                if seq_len > 2:  # exclude <cls> and <eos>
                    seq_emb = embeddings[i, 1:seq_len-1].mean(0)
                else:
                    # fallback for very short sequences
                    seq_emb = embeddings[i].mean(0)
                pooled.append(seq_emb)

            return torch.stack(pooled)

        elif self.pooling_strategy == "max":
            # max pooling over sequence (excluding special tokens)
            seq_lens = (tokens != self.alphabet.padding_idx).sum(1)
            pooled = []

            for i, seq_len in enumerate(seq_lens):
                if seq_len > 2:
                    seq_emb = embeddings[i, 1:seq_len-1].max(0)[0]
                else:
                    seq_emb = embeddings[i].max(0)[0]
                pooled.append(seq_emb)

            return torch.stack(pooled)

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        - forward pass through ESM-2 encoder and classification head
        - args:
            tokens: Tokenized sequences [batch_size, seq_len]
        - returns:
            Class logits [batch_size, num_classes]
        """
        # get ESM-2 representations
        results = self.esm(
            tokens,
            repr_layers=[self.repr_layer],
            return_contacts=False
        )

        # extract embeddings from specified layer
        embeddings = results["representations"][self.repr_layer]

        # pool embeddings
        pooled_embeddings = self._pool_embeddings(embeddings, tokens)

        # pass through classification head
        logits = self.head(pooled_embeddings)

        return logits


# ============================================================================
# fine-tuning configs
# ============================================================================

@dataclass # creates boilerplate
class FineTuningConfig:
    """
    Configs for ESM-2 fine-tuning.
    Contains all hyperparameters and settings for model architecture,
    training procedure, and data handling.
    """
    # model architecture
    model_name: str = "esm2_t33_650M_UR50D"
    num_classes: int = 3
    freeze_layers: int = 0
    hidden_dim: int = 256
    dropout: float = 0.2
    pooling_strategy: str = "mean"
    
    # training hyperparameters
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    batch_size: int = 4
    num_epochs: int = 20
    patience: int = 5
    
    # data splitting
    val_size: float = 0.2
    test_size: float = 0.2
    random_state: int = 42
    
    # device
    device: Optional[str] = None
    
    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __getitem__(self, key: str):
        """allow dict-style access like config['learning_rate']."""
        return getattr(self, key)

# ============================================================================
# dataset classes
# ============================================================================

class ESM2Dataset(Dataset):
    """
    - data for ESM-2 fine-tuning and classification
    """

    def __init__(
        self,
        sequences: List[Tuple[str, str]],
        labels: np.ndarray,
        batch_converter: Optional[object] = None
    ):
        """
        - initialize dataset
        - args:
            sequences: List of (sequence_id, sequence) tuples
            labels: Array of integer class labels
            batch_converter: ESM-2 batch converter (optional, for reference)
        """
        self.sequences = sequences
        self.labels = labels
        self.batch_converter = batch_converter

    def __len__(self) -> int:
        """return dataset size."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[str, str, int]:
        """
        get a single sample
        args:
            idx: Sample index
        returns:
            Tuple of (sequence_id, sequence, label)
        """
        seq_id, seq = self.sequences[idx]
        label = self.labels[idx]
        return seq_id, seq, label


def collate_fn(batch: List[Tuple], batch_converter) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function for ESM-2 data.

    Converts sequences to tokens using ESM-2's batch converter and
    prepares labels as tensors.

    Args:
        batch: List of (seq_id, sequence, label) tuples
        batch_converter: ESM-2 batch converter

    Returns:
        Tuple of (tokens, labels) where:
            - tokens: [batch_size, max_seq_len]
            - labels: [batch_size]
    """
    seq_ids, seqs, class_labels = zip(*batch)

    # Convert sequences to tokens
    # batch_converter returns (labels, strs, tokens) where 'labels' are seq_ids
    _, _, tokens = batch_converter(list(zip(seq_ids, seqs)))

    # Convert classification labels to tensor
    labels = torch.tensor(class_labels, dtype=torch.long)

    return tokens, labels


# ============================================================================
# data loading utilities
# ============================================================================

def parse_gisaid_fasta(fasta_path: str) -> Tuple[List[Tuple[str, str]], pd.DataFrame]:
    """
    - parse GISAID FASTA file with labeled host information
    - header format: EPI4586304|PB2|A/dairy_cow/USA/019914-004/2025|EPI_ISL_20094668|A_/_H5N1|HOST_0

    host encoding:
        - HOST_0 = human
        - HOST_1 = avian
        - HOST_2 = mammal

    args:
        fasta_path: Path to FASTA file

    returns:
        Tuple of (sequences, metadata_df) where:
            - sequences: List of (seq_id, sequence) tuples
            - metadata_df: DataFrame with sequence metadata and labels
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

    print(f"\nloaded {len(sequences)} sequences")
    print(f"\nhost distribution:")
    print(metadata_df['host_type'].value_counts())
    print(f"\nsequence length statistics:")
    print(metadata_df.groupby('host_type')['sequence_length'].describe())

    return sequences, metadata_df


def prepare_data_splits(
    sequences: List[Tuple[str, str]],
    metadata_df: pd.DataFrame,
    label_column: str = 'host_type',
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[
    List[Tuple[str, str]], np.ndarray,
    List[Tuple[str, str]], np.ndarray,
    List[Tuple[str, str]], np.ndarray,
    LabelEncoder
]:
    """
    - prepare train/val/test splits from sequences and metadata

    args:
        sequences: List of (seq_id, sequence) tuples
        metadata_df: DataFrame with metadata and labels
        label_column: Column name containing class labels
        val_size: Proportion of data for validation
        test_size: Proportion of data for test
        random_state: Random seed for reproducibility

    returns:
        Tuple of (train_seqs, train_labels, val_seqs, val_labels,
                  test_seqs, test_labels, label_encoder)
    """
    # get labels in same order as sequences
    seq_ids = [seq_id for seq_id, _ in sequences]
    label_dict = dict(zip(metadata_df['seq_id'], metadata_df[label_column]))
    y = np.array([label_dict[seq_id] for seq_id in seq_ids])

    # encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"\nlabel encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # split data: first split off test set
    seq_temp, seq_test, y_temp, y_test = train_test_split(
        sequences, y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded
    )

    # split remaining into train and validation
    seq_train, seq_val, y_train, y_val = train_test_split(
        seq_temp, y_temp,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
        stratify=y_temp
    )

    print(f"\ntrain set: {len(seq_train)}, val set: {len(seq_val)}, test set: {len(seq_test)}")

    return seq_train, y_train, seq_val, y_val, seq_test, y_test, le


# ============================================================================
# Training Utilities
# ============================================================================

class Trainer:
    """
    trainer class for ESM-2 fine-tuning.

    Handles the training loop, validation, and model checkpointing.
    """

    def __init__(
        self,
        model: ESM2FineTuned,
        config: FineTuningConfig,
        loss_fn: Optional[ClassificationLossReduction] = None
    ):
        """
        Initialize trainer.

        Args:
            model: ESM-2 fine-tuned model
            config: Training configuration
            loss_fn: Loss reduction method (optional)
        """
        self.model = model.to(config.device)
        self.config = config
        self.loss_fn = loss_fn if loss_fn is not None else ClassificationLossReduction()
        self.device = config.device

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Track best model
        self.best_val_acc = 0.0
        self.best_state_dict = None

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_tokens, batch_labels in train_loader:
            batch_tokens = batch_tokens.to(self.device)
            batch_labels = batch_labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(batch_tokens)

            # Compute loss
            batch_dict = {"labels": batch_labels}
            loss, _ = self.loss_fn.forward(batch_dict, logits)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics
            running_loss += loss.item()
            predicted = logits.argmax(1)
            correct += (predicted == batch_labels).sum().item()
            total += batch_labels.size(0)

        avg_loss = running_loss / len(train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self, eval_loader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate model on validation/test set.

        Args:
            eval_loader: Evaluation data loader

        Returns:
            Tuple of (accuracy, predictions, true_labels)
        """
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for batch_tokens, batch_labels in eval_loader:
            batch_tokens = batch_tokens.to(self.device)
            batch_labels = batch_labels.to(self.device)

            # Forward pass
            logits = self.model(batch_tokens)
            predicted = logits.argmax(1)

            # Track metrics
            correct += (predicted == batch_labels).sum().item()
            total += batch_labels.size(0)

            all_preds.append(predicted.cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())

        accuracy = correct / total
        predictions = np.concatenate(all_preds)
        true_labels = np.concatenate(all_labels)

        return accuracy, predictions, true_labels

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            verbose: Whether to print progress

        Returns:
            Dictionary with training history
        """
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_acc": []
        }

        for epoch in range(self.config.num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            val_acc, _, _ = self.evaluate(val_loader)

            # Track history
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_state_dict = self.model.state_dict().copy()

            # Print progress
            if verbose:
                print(f"Epoch [{epoch+1}/{self.config.num_epochs}], "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Train Acc: {train_acc:.4f}, "
                      f"Val Acc: {val_acc:.4f}")

        return history

    def load_best_model(self):
        """load the best model from training."""
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        return self.model


# ============================================================================
# evaluation metrics
# ============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    - compute comprehensive classification metrics

    args:
        y_true: true labels
        y_pred: predicted labels
        class_names: optional class names for display
        verbose: whether to print results

    returns:
        dictionary with accuracy, precision, recall, f1 (macro and per-class)
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )

    # weighted averages
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
    }

    # per-class metrics
    if class_names is None:
        class_names = [f"class_{i}" for i in range(len(precision))]

    for i, class_name in enumerate(class_names):
        metrics[f'precision_{class_name}'] = precision[i]
        metrics[f'recall_{class_name}'] = recall[i]
        metrics[f'f1_{class_name}'] = f1[i]
        metrics[f'support_{class_name}'] = support[i]

    if verbose:
        print("\n" + "="*60)
        print("CLASSIFICATION METRICS")
        print("="*60)
        print(f"Accuracy:           {accuracy:.4f}")
        print(f"\nMacro Averages:")
        print(f"  Precision:        {precision_macro:.4f}")
        print(f"  Recall:           {recall_macro:.4f}")
        print(f"  F1-Score:         {f1_macro:.4f}")
        print(f"\nWeighted Averages:")
        print(f"  Precision:        {precision_weighted:.4f}")
        print(f"  Recall:           {recall_weighted:.4f}")
        print(f"  F1-Score:         {f1_weighted:.4f}")

        print(f"\nPer-Class Metrics:")
        print("-"*60)
        print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-"*60)
        for i, class_name in enumerate(class_names):
            print(f"{class_name:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")
        print("="*60)

        # confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        print()

        # classification report
        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    return metrics


# ============================================================================
# model checkpoint saving/loading
# ============================================================================

def save_model_checkpoint(
    model: ESM2FineTuned,
    config: FineTuningConfig,
    label_encoder: LabelEncoder,
    metrics: Dict[str, float],
    save_path: str,
    additional_info: Optional[Dict] = None
):
    """
    - save complete model checkpoint with all necessary components for later loading

    args:
        model: trained ESM-2 model
        config: training configuration
        label_encoder: fitted label encoder
        metrics: evaluation metrics dictionary
        save_path: path to save checkpoint
        additional_info: optional additional information to save
    """
    checkpoint = {
        # model
        'model_state_dict': model.state_dict(),
        'model_name': config.model_name,

        # architecture config
        'num_classes': config.num_classes,
        'freeze_layers': config.freeze_layers,
        'hidden_dim': config.hidden_dim,
        'dropout': config.dropout,
        'pooling_strategy': config.pooling_strategy,

        # label encoding
        'label_encoder_classes': label_encoder.classes_.tolist(),

        # metrics
        'metrics': metrics,

        # full config
        'config': {
            'learning_rate': config.learning_rate,
            'weight_decay': config.weight_decay,
            'batch_size': config.batch_size,
            'num_epochs': config.num_epochs,
            'val_size': config.val_size,
            'test_size': config.test_size,
            'random_state': config.random_state,
        }
    }

    if additional_info:
        checkpoint['additional_info'] = additional_info

    torch.save(checkpoint, save_path)
    print(f"\nmodel checkpoint saved to: {save_path}")
    print(f"checkpoint includes: model weights, architecture config, label encoder, and metrics")


def load_model_checkpoint(
    checkpoint_path: str,
    device: Optional[str] = None
) -> Tuple[ESM2FineTuned, LabelEncoder, Dict]:
    """
    - load complete model checkpoint for inference or continued training

    args:
        checkpoint_path: path to saved checkpoint
        device: device to load model on (defaults to cuda if available)

    returns:
        tuple of (model, label_encoder, checkpoint_info)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # recreate model
    model = ESM2FineTuned(
        num_classes=checkpoint['num_classes'],
        model_name=checkpoint['model_name'],
        freeze_layers=checkpoint['freeze_layers'],
        hidden_dim=checkpoint['hidden_dim'],
        dropout=checkpoint['dropout'],
        pooling_strategy=checkpoint['pooling_strategy']
    )

    # load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # recreate label encoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(checkpoint['label_encoder_classes'])

    # extract info
    info = {
        'metrics': checkpoint['metrics'],
        'config': checkpoint['config'],
        'model_name': checkpoint['model_name'],
    }

    if 'additional_info' in checkpoint:
        info['additional_info'] = checkpoint['additional_info']

    print(f"\nmodel checkpoint loaded from: {checkpoint_path}")
    print(f"model: {checkpoint['model_name']}")
    print(f"classes: {checkpoint['label_encoder_classes']}")
    print(f"test accuracy: {checkpoint['metrics'].get('accuracy', 'N/A')}")

    return model, label_encoder, info


# ============================================================================
# dataloaders
# ============================================================================

def create_data_loaders(
    train_sequences: List[Tuple[str, str]],
    train_labels: np.ndarray,
    val_sequences: List[Tuple[str, str]],
    val_labels: np.ndarray,
    test_sequences: List[Tuple[str, str]],
    test_labels: np.ndarray,
    batch_converter,
    batch_size: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.

    Args:
        train_sequences: Training sequences
        train_labels: Training labels
        val_sequences: Validation sequences
        val_labels: Validation labels
        test_sequences: Test sequences
        test_labels: Test labels
        batch_converter: ESM-2 batch converter
        batch_size: Batch size

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = ESM2Dataset(train_sequences, train_labels, batch_converter)
    val_dataset = ESM2Dataset(val_sequences, val_labels, batch_converter)
    test_dataset = ESM2Dataset(test_sequences, test_labels, batch_converter)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, batch_converter)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, batch_converter)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, batch_converter)
    )

    return train_loader, val_loader, test_loader


def create_model_and_trainer(
    config: FineTuningConfig
) -> Tuple[ESM2FineTuned, Trainer]:
    """
    - create model and trainer from configuration
    - args:
        config: Fine-tuning configuration
    - returns:
        Tuple of (model, trainer)
    """
    model = ESM2FineTuned(
        num_classes=config.num_classes,
        model_name=config.model_name,
        freeze_layers=config.freeze_layers,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        pooling_strategy=config.pooling_strategy
    )

    trainer = Trainer(model, config)

    return model, trainer


# ============================================================================
# contact map extraction
# ============================================================================

def extract_contact_map(
    model: ESM2FineTuned,
    sequence: str,
    seq_id: str = "sequence",
    device: Optional[str] = None
) -> Tuple[np.ndarray, torch.Tensor]:
    """
    - extract contact map from ESM-2 model for a single sequence

    args:
        model: ESM-2 fine-tuned model
        sequence: protein sequence string
        seq_id: sequence identifier
        device: device to run on (defaults to model's device)

    returns:
        Tuple of (contact_map_numpy, attention_contacts) where:
            - contact_map_numpy: [seq_len, seq_len] contact probability matrix
            - attention_contacts: Raw attention-based contact predictions
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # convert sequence to tokens
    batch_converter = model.batch_converter
    data = [(seq_id, sequence)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    # get predictions with contact map
    with torch.no_grad():
        results = model.esm(
            tokens,
            repr_layers=[model.repr_layer],
            return_contacts=True  
        )

    # extract contact predictions
    # contacts shape: [batch, seq_len, seq_len]
    contacts = results["contacts"]

    # remove batch dimension and convert to numpy
    contact_map = contacts[0].cpu().numpy()

    # the contact map includes special tokens, so trim them
    # tokens are: <cls> seq <eos> <pad>...
    seq_len = len(sequence)
    contact_map = contact_map[:seq_len, :seq_len]

    return contact_map, contacts


def extract_contact_maps_batch(
    model: ESM2FineTuned,
    sequences: List[Tuple[str, str]],
    device: Optional[str] = None,
    max_sequences: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    - extract contact maps for multiple sequences

    args:
        model: ESM-2 model
        sequences: List of (seq_id, sequence) tuples
        device: Device to run on
        max_sequences: Maximum number of sequences to process (None = all)

    returns:
        Dictionary mapping seq_id to contact map array
    """
    if device is None:
        device = next(model.parameters()).device

    contact_maps = {}

    sequences_to_process = sequences[:max_sequences] if max_sequences else sequences

    for seq_id, seq in sequences_to_process:
        contact_map, _ = extract_contact_map(model, seq, seq_id, device)
        contact_maps[seq_id] = contact_map

    return contact_maps
    
def compute_contact_map_difference(
    model: ESM2FineTuned,
    seq1: str,
    seq2: str,
    seq_id1: str = "sequence1",
    seq_id2: str = "sequence2",
    device: Optional[str] = None,
    normalize: bool = True
) -> Dict[str, np.ndarray]:
    """
    compute difference between contact maps of two sequences.
    
    Args:
        model: ESM-2 fine-tuned model
        seq1: First protein sequence (e.g., avian)
        seq2: Second protein sequence (e.g., human-adapted)
        seq_id1: Identifier for first sequence
        seq_id2: Identifier for second sequence
        device: Device to run on
        normalize: Whether to normalize by sequence length
        
    Returns:
        Dict containing:
            - 'contact_map1': Contact map for seq1
            - 'contact_map2': Contact map for seq2
            - 'difference': seq2 - seq1 (positive = gained contacts, negative = lost)
            - 'abs_difference': Absolute difference
            - 'residue_contact_change': Per-residue sum of contact changes
            - 'top_changed_residues': Indices sorted by magnitude of contact change
    """
    # extract contact maps
    contact_map1, _ = extract_contact_map(model, seq1, seq_id1, device)
    contact_map2, _ = extract_contact_map(model, seq2, seq_id2, device)
    
    # handle length differences - align or pad to same size
    len1, len2 = contact_map1.shape[0], contact_map2.shape[0]
    
    if len1 != len2:
        # pad shorter to match longer
        max_len = max(len1, len2)
        if len1 < max_len:
            pad_size = max_len - len1
            contact_map1 = np.pad(contact_map1, ((0, pad_size), (0, pad_size)), mode='constant')
        if len2 < max_len:
            pad_size = max_len - len2
            contact_map2 = np.pad(contact_map2, ((0, pad_size), (0, pad_size)), mode='constant')
    
    # compute differences
    difference = contact_map2 - contact_map1  # positive = gained contact in seq2
    abs_difference = np.abs(difference)
    
    # per-residue contact change (sum of row/column changes)
    residue_contact_change = np.sum(abs_difference, axis=1)
    
    if normalize:
        residue_contact_change = residue_contact_change / abs_difference.shape[0]
    
    # rank residues by contact change magnitude
    top_changed_residues = np.argsort(residue_contact_change)[::-1]
    
    return {
        'contact_map1': contact_map1,
        'contact_map2': contact_map2,
        'difference': difference,
        'abs_difference': abs_difference,
        'residue_contact_change': residue_contact_change,
        'top_changed_residues': top_changed_residues,
        'seq_lengths': (len1, len2)
    }

def compute_group_contact_difference(
    model: ESM2FineTuned,
    group1_seqs: List[Tuple[str, str]],  # [(id, seq), ...]
    group2_seqs: List[Tuple[str, str]],
    device: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Compare average contact maps between two groups of sequences.
    
    E.g., compare avian isolates vs human isolates to find
    systematic contact pattern differences associated with host adaptation.
    
    Args:
        model: ESM-2 model
        group1_seqs: List of (id, sequence) tuples for group 1
        group2_seqs: List of (id, sequence) tuples for group 2
        device: Device to run on
        
    Returns:
        Dict with mean contact maps, difference, and statistics
    """
    # extract all contact maps
    maps1 = []
    maps2 = []
    
    max_len = 0
    
    # first pass: get all maps and find max length
    for seq_id, seq in group1_seqs:
        cmap, _ = extract_contact_map(model, seq, seq_id, device)
        maps1.append(cmap)
        max_len = max(max_len, cmap.shape[0])
        
    for seq_id, seq in group2_seqs:
        cmap, _ = extract_contact_map(model, seq, seq_id, device)
        maps2.append(cmap)
        max_len = max(max_len, cmap.shape[0])
    
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


def visualize_contact_map(
    contact_map: np.ndarray,
    seq_id: str = "sequence",
    figsize: Tuple[int, int] = (10, 10),
    threshold: Optional[float] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    - visualize a contact map as a heatmap

    args:
        contact_map: [seq_len, seq_len] contact probability matrix
        seq_id: Sequence identifier for title
        figsize: Figure size
        threshold: If provided, show binary contacts above this threshold
        save_path: Optional path to save figure

    returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if threshold is not None:
        # binarize contact map
        plot_data = (contact_map >= threshold).astype(float)
        cmap = 'binary'
        cbar_label = f'Contact (threshold={threshold})'
    else:
        plot_data = contact_map
        cmap = 'Blues'
        cbar_label = 'Contact Probability'

    # plot heatmap
    im = ax.imshow(plot_data, cmap=cmap, origin='lower', aspect='auto')

    # add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, rotation=270, labelpad=20)

    # labels
    ax.set_xlabel('Residue Position')
    ax.set_ylabel('Residue Position')
    ax.set_title(f'Contact Map - {seq_id}')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"saved contact map to {save_path}")

    return fig


def compare_contact_maps(
    contact_maps: Dict[str, np.ndarray],
    seq_ids: List[str],
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    - compare multiple contact maps side by side

    args:
        contact_maps: Dictionary of seq_id -> contact map
        seq_ids: List of seq_ids to compare
        figsize: Figure size
        save_path: Optional path to save figure

    returns:
        Matplotlib figure
    """
    n_maps = len(seq_ids)
    fig, axes = plt.subplots(1, n_maps, figsize=figsize)

    if n_maps == 1:
        axes = [axes]

    for ax, seq_id in zip(axes, seq_ids):
        contact_map = contact_maps[seq_id]
        im = ax.imshow(contact_map, cmap='Blues', origin='lower', aspect='auto')
        ax.set_title(seq_id)
        ax.set_xlabel('Residue Position')
        if ax == axes[0]:
            ax.set_ylabel('Residue Position')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"saved comparison to {save_path}")

    return fig

def plot_group_contact_difference(
    group_result: Dict[str, np.ndarray],
    group1_name: str = "Group 1",
    group2_name: str = "Group 2",
    highlight_residues: Optional[List[int]] = None,
    highlight_labels: Optional[List[str]] = None,
    effect_threshold: float = 0.5,
    figsize: Tuple[int, int] = (18, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize group-level contact map comparison with effect sizes.
    
    Args:
        group_result: Output from compute_group_contact_difference()
        group1_name: Label for first group (e.g., "Avian isolates")
        group2_name: Label for second group (e.g., "Human isolates")
        highlight_residues: Key residue positions to mark
        highlight_labels: Labels for highlighted residues
        effect_threshold: Effect size threshold for significance
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=figsize)
    
    # create grid layout
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.8], hspace=0.3, wspace=0.3)
    
    mean1 = group_result['mean_contact_map_group1']
    mean2 = group_result['mean_contact_map_group2']
    difference = group_result['difference']
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
    
    # add highlights
    if highlight_residues:
        for res in highlight_residues:
            if res < seq_len:
                ax3.axhline(y=res, color='green', alpha=0.3, linewidth=1)
                ax3.axvline(x=res, color='green', alpha=0.3, linewidth=1)
    
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
    
    if highlight_residues:
        for i, res in enumerate(highlight_residues):
            if res < seq_len:
                label = highlight_labels[i] if highlight_labels and i < len(highlight_labels) else str(res)
                ax4.axvline(x=res, color='green', alpha=0.6, linewidth=1.5)
                y_pos = min(residue_effect[res] + 0.1, max(residue_effect) * 0.95)
                ax4.annotate(label, xy=(res, y_pos), fontsize=9, ha='center', 
                           color='darkgreen', fontweight='bold')
    
    # bottom right: top changed residues table
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    top_n = 15
    top_residues = group_result['top_changed_residues'][:top_n]
    
    table_data = []
    for rank, res in enumerate(top_residues, 1):
        effect = residue_effect[res]
        table_data.append([rank, res + 1, f"{effect:.3f}"])  # +1 for 1-indexed display
    
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
    
    return fig

def plot_contact_network_change(
    diff_result: Dict[str, np.ndarray],
    threshold: float = 0.2,
    top_n_edges: int = 50,
    min_sequence_distance: int = 6,  # ignore nearby contacts
    highlight_residues: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    if 'effect_size' in diff_result:
        diff_matrix = diff_result['effect_size']
    else:
        diff_matrix = diff_result['difference']
    
    seq_len = diff_matrix.shape[0]
    
    gained_contacts = []
    lost_contacts = []
    
    for i in range(seq_len):
        for j in range(i + min_sequence_distance, seq_len):  # skip nearby residues
            val = diff_matrix[i, j]
            if abs(val) > threshold:
                if val > 0:
                    gained_contacts.append((i, j, val))
                else:
                    lost_contacts.append((i, j, abs(val)))
    
    gained_contacts.sort(key=lambda x: x[2], reverse=True)
    lost_contacts.sort(key=lambda x: x[2], reverse=True)
    
    gained_contacts = gained_contacts[:top_n_edges]
    lost_contacts = lost_contacts[:top_n_edges]
    
    # plot gained contacts
    ax = axes[0]
    for i, j, val in gained_contacts:
        alpha = min(val / (2 * threshold), 1.0)
        ax.scatter([i], [j], c='red', alpha=alpha, s=val * 50)
        ax.scatter([j], [i], c='red', alpha=alpha, s=val * 50)  # symmetric
    
    ax.plot([0, seq_len], [0, seq_len], 'k--', alpha=0.2)  # diagonal reference
    ax.set_xlim(0, seq_len)
    ax.set_ylim(0, seq_len)
    ax.set_xlabel("Residue Position")
    ax.set_ylabel("Residue Position")
    ax.set_title(f"Gained Contacts\n(n={len(gained_contacts)} pairs)")
    ax.set_aspect('equal')
    
    # plot lost contacts
    ax = axes[1]
    for i, j, val in lost_contacts:
        alpha = min(val / (2 * threshold), 1.0)
        ax.scatter([i], [j], c='blue', alpha=alpha, s=val * 50)
        ax.scatter([j], [i], c='blue', alpha=alpha, s=val * 50)
    
    ax.plot([0, seq_len], [0, seq_len], 'k--', alpha=0.2)
    ax.set_xlim(0, seq_len)
    ax.set_ylim(0, seq_len)
    ax.set_xlabel("Residue Position")
    ax.set_ylabel("Residue Position")
    ax.set_title(f"Lost Contacts\n(n={len(lost_contacts)} pairs)")
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

# ============================================================================
# usage - complete pipeline example
# ============================================================================

if __name__ == "__main__":
    fasta_path = '/insomnia001/depts/edu/BIOLBC3141_Fall2025/rc3517/nlp/data/all_pb2_sequences_labeled_balanced.fasta'
    output_dir = '/insomnia001/depts/edu/BIOLBC3141_Fall2025/rc3517/nlp/finetune_results'

    # configure fine-tuning parameters
    config = FineTuningConfig(
        num_classes=3,  
        model_name="esm2_t33_650M_UR50D",
        freeze_layers=0,  
        hidden_dim=256,
        dropout=0.2,
        batch_size=16,
        num_epochs=20,
        patience=5,
        learning_rate=1e-3,
        val_size=0.2,
        test_size=0.2,
        random_state=42
    )

    print("="*80)
    print("ESM-2 FINE-TUNING PIPELINE")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  Freeze layers: {config.freeze_layers}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Device: {config.device}")

    # STEP 1: load data from FASTA file
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA FROM FASTA")
    print("="*80)
    sequences, metadata_df = parse_gisaid_fasta(fasta_path)

    # STEP 2: prepare train/val/test splits
    print("\n" + "="*80)
    print("STEP 2: PREPARING DATA SPLITS")
    print("="*80)
    train_seqs, train_labels, val_seqs, val_labels, test_seqs, test_labels, label_encoder = prepare_data_splits(
        sequences, metadata_df,
        label_column='host_type',
        val_size=config.val_size,
        test_size=config.test_size,
        random_state=config.random_state
    )

    # STEP 3: create model and trainer
    print("\n" + "="*80)
    print("STEP 3: CREATING MODEL")
    print("="*80)
    model, trainer = create_model_and_trainer(config)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # STEP 4: create data loaders
    print("\n" + "="*80)
    print("STEP 4: CREATING DATA LOADERS")
    print("="*80)
    train_loader, val_loader, test_loader = create_data_loaders(
        train_seqs, train_labels,
        val_seqs, val_labels,
        test_seqs, test_labels,
        model.batch_converter,
        batch_size=config.batch_size
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # STEP 5: train the model
    print("\n" + "="*80)
    print("STEP 5: TRAINING MODEL")
    print("="*80)
    history = trainer.train(train_loader, val_loader, verbose=True)

    # STEP 6: evaluate on all sets with comprehensive metrics
    print("\n" + "="*80)
    print("STEP 6: FINAL EVALUATION")
    print("="*80)
    best_model = trainer.load_best_model()

    # evaluate on all three sets
    train_acc, train_preds, train_labels_true = trainer.evaluate(train_loader)
    val_acc, val_preds, val_labels_true = trainer.evaluate(val_loader)
    test_acc, test_preds, test_labels_true = trainer.evaluate(test_loader)

    print(f"\nAccuracy Summary:")
    print(f"  Train: {train_acc:.4f}")
    print(f"  Val:   {val_acc:.4f}")
    print(f"  Test:  {test_acc:.4f}")

    # compute comprehensive metrics for test set
    print("\n" + "="*80)
    print("TEST SET METRICS")
    print("="*80)
    test_metrics = compute_metrics(
        test_labels_true,
        test_preds,
        class_names=label_encoder.classes_.tolist(),
        verbose=True
    )

    # also compute metrics for train and val (without verbose output)
    train_metrics = compute_metrics(
        train_labels_true,
        train_preds,
        class_names=label_encoder.classes_.tolist(),
        verbose=False
    )

    val_metrics = compute_metrics(
        val_labels_true,
        val_preds,
        class_names=label_encoder.classes_.tolist(),
        verbose=False
    )

    # save complete model checkpoint
    checkpoint_path = f'{output_dir}/esm2_finetuned_checkpoint.pt'
    save_model_checkpoint(
        model=best_model,
        config=config,
        label_encoder=label_encoder,
        metrics=test_metrics,
        save_path=checkpoint_path,
        additional_info={
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'training_history': history
        }
    )

    # STEP 7: extract contact maps for sample sequences
    print("\n" + "="*80)
    print("STEP 7: EXTRACTING CONTACT MAPS")
    print("="*80)

    # extract contact maps for 3 random sequences from each class
    sample_sequences = random.sample(sequences, 3)
    print(f"extracting contact maps for {len(sample_sequences)} sample sequences...")

    contact_maps = extract_contact_maps_batch(
        best_model,
        sample_sequences,
        device=config.device,
        max_sequences=3
    )

    # visualize first contact map
    if len(contact_maps) > 0:
        first_seq_id = list(contact_maps.keys())[0]
        fig = visualize_contact_map(
            contact_maps[first_seq_id],
            seq_id=first_seq_id,
            save_path=f'{output_dir}/contact_map_sample.png'
        )

    # compute contact map difference (batch)
    # get human sequences from sequences
    human_seqs = [(seq_id, seq) for (seq_id, seq) in sequences
                if metadata_df[metadata_df['seq_id'] == seq_id]['host_type'].values[0] == 'human']

    # get human sequences from sequences
    avian_seqs = [(seq_id, seq) for (seq_id, seq) in sequences
                    if metadata_df[metadata_df['seq_id'] == seq_id]['host_type'].values[0] == 'avian']

    # get human sequences from sequences
    mammal_seqs = [(seq_id, seq) for (seq_id, seq) in sequences
                    if metadata_df[metadata_df['seq_id'] == seq_id]['host_type'].values[0] == 'mammal']

    # human vs avian
    human_avian = compute_group_contact_difference(
        best_model,
        human_seqs,
        avian_seqs,
        device = config.device        
    )

    # human vs mammals
    human_mammals = compute_group_contact_difference(
        best_model,
        human_seqs,
        mammal_seqs,
        device = config.device        
    )

    # avian vs mammals
    avian_mammals = compute_group_contact_difference(
        best_model,
        avian_seqs,
        mammal_seqs,
        device = config.device        
    )

    # visualize contact maps difference
    fig1 = plot_group_contact_difference(
        human_avian,
        group1_name="Human",
        group2_name="Avian",
        save_path=f'{output_dir}/human_avian_group_contact_diff.png'
    )

    fig2 = plot_group_contact_difference(
        human_mammals,
        group1_name="Human",
        group2_name="Other Mammals",
        save_path=f'{output_dir}/human_mammals_group_contact_diff.png'
    )

    fig3 = plot_group_contact_difference(
        avian_mammals,
        group1_name="Avian",
        group2_name="Other Mammals",
        save_path=f'{output_dir}/avian_mammals_group_contact_diff.png'
    )

    # network plots
    fig4 = plot_contact_network_change(
        diff_result=human_avian,
        threshold = 0.2,
        top_n_edges = 50,
        save_path = f'{output_dir}/human_avian_network.png'
    )

    fig5 = plot_contact_network_change(
        diff_result=human_mammals,
        threshold = 0.2,
        top_n_edges = 50,
        save_path = f'{output_dir}/human_mammals_network.png'
    )

    fig6 = plot_contact_network_change(
        diff_result=avian_mammals,
        threshold = 0.2,
        top_n_edges = 50,
        save_path = f'{output_dir}/avian_mammals_network.png'
    )

    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nSaved files:")
    print(f"  - Model checkpoint: {checkpoint_path}")
    print(f"  - Contact maps and analysis plots in: {output_dir}")
    print(f"\nTo load the model later for exploring contact maps:")
    print(f"  from esm2_finetuning import load_model_checkpoint, extract_contact_map")
    print(f"  model, label_encoder, info = load_model_checkpoint('{checkpoint_path}')")
    print(f"  contact_map, _ = extract_contact_map(model, your_sequence, 'seq_id')")
    print("="*80)
