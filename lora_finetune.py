# lora_finetune.py

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import datetime
from my_model import ViolinFingeringTransformer
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader
from my_model import *

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


class PrefixViolinFingeringDataset(Dataset):
    """Dataset for violin fingering data with prefix tokens"""
    
    def __init__(self, segments, lens, prefix_labels):
        self.segments = segments
        self.lens = lens
        self.prefix_labels = prefix_labels
        
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        segment = self.segments[idx]
        length = self.lens[idx]
        prefix_label = self.prefix_labels[idx]
        
        # Extract features
        pitch = torch.LongTensor(segment['pitch'])
        start = torch.FloatTensor(segment['start'])
        duration = torch.FloatTensor(segment['duration'])
        beat_type = torch.LongTensor(segment['beat_type'])
        
        # Extract targets - convert to joint label
        string = segment['string']
        finger = segment['finger']
        
        # Joint label: string*5 + finger
        joint_label = string * 5 + finger
        joint_label = torch.LongTensor(joint_label)
        
        return {
            'pitch': pitch,
            'start': start,
            'duration': duration,
            'beat_type': beat_type,
            'joint_label': joint_label,
            'string': torch.LongTensor(string),
            'finger': torch.LongTensor(finger),
            'length': length,
            'prefix_label': prefix_label  # Integer index for prefix token
        }


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer for fine-tuning.
    Adds trainable low-rank matrices A and B to frozen weight W.
    Forward: h = Wx + (BAx) * scaling
    """
    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x):
        # x shape: [..., in_features]
        # Compute low-rank update: x @ A @ B
        lora_out = x @ self.lora_A  # [..., rank]
        lora_out = self.dropout(lora_out)
        lora_out = lora_out @ self.lora_B  # [..., out_features]
        return lora_out * self.scaling


class LoRALinear(nn.Module):
    """
    Wrapper around nn.Linear that adds LoRA adaptation.
    """
    def __init__(self, linear_layer, rank=8, alpha=16, dropout=0.1):
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(
            linear_layer.in_features,
            linear_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        
        # Freeze the original linear layer
        for param in self.linear.parameters():
            param.requires_grad = False
    
    @property
    def weight(self):
        """Expose weight attribute for compatibility with modules that access it directly"""
        return self.linear.weight
    
    @property
    def bias(self):
        """Expose bias attribute for compatibility with modules that access it directly"""
        return self.linear.bias
    
    @property
    def in_features(self):
        """Expose in_features for compatibility"""
        return self.linear.in_features
    
    @property
    def out_features(self):
        """Expose out_features for compatibility"""
        return self.linear.out_features
    
    def forward(self, x):
        return self.linear(x) + self.lora(x)


class PrefixViolinFingeringTransformer(nn.Module):
    """
    Wrapper around ViolinFingeringTransformer that adds prefix token support.
    The prefix token is prepended to the sequence and encodes user conditions.
    """
    def __init__(self, base_model, n_prefix_tokens=9, d_model=None):
        super().__init__()
        self.base_model = base_model
        self.n_prefix_tokens = n_prefix_tokens
        
        # Use d_model from base model config
        if d_model is None:
            d_model = base_model.config['d_model']
        
        # Learnable prefix embeddings for each condition combination
        self.prefix_embeddings = nn.Parameter(torch.randn(n_prefix_tokens, d_model) * 0.01)
        
        # Expose base model attributes
        self.no_fingering_token = base_model.no_fingering_token
        self.config = base_model.config
    
    def forward(self, pitch, start, duration, beat_type, prev_fingerings, prefix_labels, padding_mask=None):
        """
        Args:
            pitch: [batch, seq_len]
            start: [batch, seq_len]
            duration: [batch, seq_len]
            beat_type: [batch, seq_len]
            prev_fingerings: [batch, seq_len]
            prefix_labels: [batch] - integer indices for prefix tokens
            padding_mask: [batch, seq_len]
        """
        batch_size, seq_len = pitch.shape
        device = pitch.device
        
        # Get embeddings from base model (but don't run transformer yet)
        pitch_adjusted = (pitch - self.base_model.lowest_pitch).clamp(
            0, self.base_model.config['n_pitch_classes'] - 1
        )
        
        pitch_emb = self.base_model.pitch_embedding(pitch_adjusted)
        beat_emb = self.base_model.beat_embedding(beat_type)
        fingering_emb = self.base_model.fingering_embedding(prev_fingerings)
        start_emb = self.base_model.start_projection(start.unsqueeze(-1))
        duration_emb = self.base_model.duration_projection(duration.unsqueeze(-1))
        
        # Concatenate and project
        x = torch.cat([pitch_emb, start_emb, duration_emb, beat_emb, fingering_emb], dim=-1)
        x = self.base_model.input_projection(x)
        x = self.base_model.input_norm(x)
        
        # Prepend prefix tokens
        prefix_tokens = self.prefix_embeddings[prefix_labels]  # [batch, d_model]
        prefix_tokens = prefix_tokens.unsqueeze(1)  # [batch, 1, d_model]
        x = torch.cat([prefix_tokens, x], dim=1)  # [batch, seq_len+1, d_model]
        
        # Add positional encoding
        x = self.base_model.pos_encoder(x)
        
        # Update padding mask to account for prefix token
        if padding_mask is not None:
            # Prefix token is never padding (False)
            prefix_mask = torch.zeros(batch_size, 1, device=device, dtype=torch.bool)
            padding_mask = torch.cat([prefix_mask, padding_mask], dim=1)
        
        # Transformer encoding
        transformer_output = self.base_model.transformer_encoder(
            x,
            src_key_padding_mask=padding_mask
        )
        
        # Extract middle position (accounting for prefix token shift)
        mid_idx = seq_len // 2
        mid_position_output = transformer_output[:, mid_idx + 1, :]  # +1 for prefix token
        
        # Predict joint string+finger
        joint_logits = self.base_model.joint_head(mid_position_output)
        
        # Apply fingering mask
        mid_pitch = pitch[:, mid_idx]
        joint_mask = self.base_model.create_joint_mask_single(mid_pitch)
        joint_logits = joint_logits.masked_fill(~joint_mask, -1000)
        
        return joint_logits


def add_lora_to_model(model, rank=8, alpha=16, dropout=0.1):
    """
    Add LoRA adapters to specified modules in the model.
    
    Args:
        model: The model to add LoRA to
        rank: Rank of LoRA matrices
        alpha: LoRA scaling parameter
        dropout: Dropout rate for LoRA
    
    Returns:
        model: Modified model with LoRA layers
        lora_params: List of LoRA parameters
    """
    lora_params = []
    
    # Get device from model
    device = next(model.parameters()).device
    
    # Get the base transformer model (unwrap if using prefix wrapper)
    if hasattr(model, 'base_model'):
        base_model = model.base_model
        # Add prefix embeddings to trainable params
        lora_params.append(model.prefix_embeddings)
        print(f"  Added prefix embeddings to trainable params")
    else:
        base_model = model
    
    # Iterate through transformer encoder layers
    for layer_idx, layer in enumerate(base_model.transformer_encoder.layers):
        # Replace output projection in self-attention
        if hasattr(layer.self_attn, 'out_proj'):
            original_linear = layer.self_attn.out_proj
            lora_linear = LoRALinear(original_linear, rank=rank, alpha=alpha, dropout=dropout).to(device)
            layer.self_attn.out_proj = lora_linear
            lora_params.extend(list(lora_linear.lora.parameters()))
            print(f"  Added LoRA to layer {layer_idx} self_attn.out_proj")
        
        # Replace feedforward layers
        if hasattr(layer, 'linear1'):
            original_linear1 = layer.linear1
            lora_linear1 = LoRALinear(original_linear1, rank=rank, alpha=alpha, dropout=dropout).to(device)
            layer.linear1 = lora_linear1
            lora_params.extend(list(lora_linear1.lora.parameters()))
            print(f"  Added LoRA to layer {layer_idx} linear1")
        
        if hasattr(layer, 'linear2'):
            original_linear2 = layer.linear2
            lora_linear2 = LoRALinear(original_linear2, rank=rank, alpha=alpha, dropout=dropout).to(device)
            layer.linear2 = lora_linear2
            lora_params.extend(list(lora_linear2.lora.parameters()))
            print(f"  Added LoRA to layer {layer_idx} linear2")
    
    # Optionally add LoRA to the prediction head
    for name, module in base_model.joint_head.named_children():
        if isinstance(module, nn.Linear):
            original_linear = module
            lora_linear = LoRALinear(original_linear, rank=rank, alpha=alpha, dropout=dropout).to(device)
            # Replace in the sequential
            idx = int(name) if name.isdigit() else list(base_model.joint_head._modules.keys()).index(name)
            base_model.joint_head[idx] = lora_linear
            lora_params.extend(list(lora_linear.lora.parameters()))
            print(f"  Added LoRA to joint_head.{name}")
    
    return model, lora_params


class LoRAFineTuner:
    """Fine-tuning class using LoRA with prefix tokens"""
    
    def __init__(self, base_model_path, lora_config, training_config):
        self.lora_config = lora_config
        self.training_config = training_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load base model
        print(f"\nLoading base model from: {base_model_path}")
        checkpoint = torch.load(base_model_path, map_location=self.device, weights_only=False)
        
        if 'model' in checkpoint:
            base_model = checkpoint['model'].to(self.device)
        elif 'model_state_dict' in checkpoint:
            config = checkpoint['config']
            base_model = ViolinFingeringTransformer(config).to(self.device)
            base_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise KeyError("Checkpoint does not contain 'model' or 'model_state_dict'")
        
        self.base_config = checkpoint.get('config', {})
        print(f"Base model loaded successfully")
        
        # Define prefix token mapping
        # 3 levels x 3 shift styles = 9 combinations
        self.level_to_idx = {'beginner': 0, 'medium': 1, 'advanced': 2}
        self.shift_to_idx = {'less shifting': 0, 'normal shifting': 1, 'more shifting': 2}
        self.n_prefix_tokens = 9
        
        print(f"\nPrefix token mapping:")
        for level, level_idx in self.level_to_idx.items():
            for shift, shift_idx in self.shift_to_idx.items():
                prefix_idx = level_idx * 3 + shift_idx
                print(f"  {level} + {shift} -> token {prefix_idx}")
        
        # Wrap base model with prefix support
        print(f"\nAdding prefix token support ({self.n_prefix_tokens} tokens)...")
        self.model = PrefixViolinFingeringTransformer(
            base_model, 
            n_prefix_tokens=self.n_prefix_tokens,
            d_model=self.base_config['d_model']
        ).to(self.device)
        
        # Add LoRA adapters
        print(f"\nAdding LoRA adapters (rank={lora_config['rank']}, alpha={lora_config['alpha']})...")
        self.model, self.lora_params = add_lora_to_model(
            self.model,
            rank=lora_config['rank'],
            alpha=lora_config['alpha'],
            dropout=lora_config['dropout']
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nParameter Summary:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters (LoRA + Prefix): {trainable_params:,}")
        print(f"  Trainable percentage: {100 * trainable_params / total_params:.2f}%")
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer (only LoRA parameters and prefix embeddings)
        self.optimizer = torch.optim.AdamW(
            self.lora_params,
            lr=training_config['learning_rate'],
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=training_config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=training_config['scheduler_t0'],
            T_mult=1,
            eta_min=training_config['learning_rate'] * 0.01
        )
    
    def load_piece_labels(self, labels_path):
        """Load piece-level labels from CSV"""
        print(f"\nLoading piece labels from: {labels_path}")
        df = pd.read_csv(labels_path)
        
        # Create mapping from filename to (level, shift) tuple
        piece_labels = {}
        for _, row in df.iterrows():
            filename = row['file']
            level = row['level_label']
            shift = row['shift_label']
            piece_labels[filename] = (level, shift)
        
        print(f"Loaded labels for {len(piece_labels)} pieces")
        
        # Print distribution
        level_counts = {}
        shift_counts = {}
        for level, shift in piece_labels.values():
            level_counts[level] = level_counts.get(level, 0) + 1
            shift_counts[shift] = shift_counts.get(shift, 0) + 1
        
        print(f"\nLabel distribution:")
        print(f"  Levels: {level_counts}")
        print(f"  Shifts: {shift_counts}")
        
        return piece_labels
    
    def get_prefix_label(self, filename):
        """Convert filename to prefix token index"""
        if filename not in self.piece_labels:
            print(f"Warning: {filename} not in piece_labels, using default (beginner, less shifting)")
            return 0  # Default to beginner + less shifting
        
        level, shift = self.piece_labels[filename]
        level_idx = self.level_to_idx.get(level, 0)
        shift_idx = self.shift_to_idx.get(shift, 0)
        prefix_idx = level_idx * 3 + shift_idx
        return prefix_idx
    
    def load_data(self, dataset_dir):
        """Load and process data from CSV files"""
        print("\nLoading fine-tuning data...")
        files = [x for x in os.listdir(dataset_dir) if x.endswith('csv')]
        corpus = {}
        
        for file in files:
            file_path = os.path.join(dataset_dir, file)
            with open(file_path) as f:
                corpus[file] = np.genfromtxt(
                    f, 
                    delimiter=',', 
                    names=True, 
                    dtype=[('int'), ('float'), ('float'), ('int'), ('int'), ('int'), ('int')]
                )
        
        return corpus
    
    def segment_corpus(self, corpus, max_len):
        """Segment sequences into fixed-length chunks with prefix labels"""
        def _segment_sequence(sequence, max_len, pitch_for_invalid):
            dt = sequence.dtype
            N = len(sequence)
            segments = []
            valid_lens = []

            padding_template = (pitch_for_invalid, -1, 0, 0, 0, 0, 0)

            for idx in range(N):
                valid_len = min(idx + 1, max_len)
                pad_len = max_len - valid_len
                padding = np.array([padding_template] * pad_len, dtype=dt) if pad_len > 0 else np.empty((0,), dtype=dt)
                start_idx = max(0, idx - max_len + 1)
                notes = sequence[start_idx: idx + 1]
                segment = np.concatenate([padding, notes])
                assert len(segment) == max_len
                segments.append(segment)
                valid_lens.append(len(notes)) 

            segments = np.stack(segments, axis=0)
            return segments, valid_lens

        corpus_seg = {}
        for key, sequence in corpus.items():
            segments, valid_lens = _segment_sequence(
                sequence, 
                max_len, 
                self.base_config['pitch_for_invalid_note']
            )
            
            # Get prefix label for this piece
            prefix_label = self.get_prefix_label(key)
            
            corpus_seg[key] = {
                'segments': segments,
                'lens': valid_lens,
                'prefix_label': prefix_label
            }
        
        total_segments = sum([v['segments'].shape[0] for v in corpus_seg.values()])
        print(f'Total number of segments: {total_segments}')
        return corpus_seg
    
    def create_dataloaders(self, corpus_seg, train_files=None, test_files=None):
        """Create training and testing dataloaders with balanced sampling"""
        # Filter for vio1_ through vio10_ files
        corpus_vio1 = {k: v for k, v in corpus_seg.items() 
                      if any(f'vio{i}_' in k for i in range(1, 11))}
        
        if train_files is None and test_files is None:
            # Create balanced dataset based on piece labels
            print("\nCreating balanced dataset...")
            
            # Group pieces by level
            beginner_pieces = []
            medium_pieces = []
            advanced_pieces = []
            
            for filename, data in corpus_vio1.items():
                if filename in self.piece_labels:
                    level, shift = self.piece_labels[filename]
                    if level == 'beginner':
                        beginner_pieces.append(filename)
                    elif level == 'medium':
                        medium_pieces.append(filename)
                    elif level == 'advanced':
                        advanced_pieces.append(filename)
            
            print(f"\nAvailable pieces by level:")
            print(f"  Beginner: {len(beginner_pieces)} pieces")
            print(f"  Medium: {len(medium_pieces)} pieces")
            print(f"  Advanced: {len(advanced_pieces)} pieces")
            
            # Select 14 medium pieces randomly to balance with beginner and advanced
            import random
            random.seed(42)  # For reproducibility
            selected_medium = random.sample(medium_pieces, min(14, len(medium_pieces)))
            
            # All beginner and advanced pieces
            selected_beginner = beginner_pieces
            selected_advanced = advanced_pieces
            
            print(f"\nSelected for balanced dataset:")
            print(f"  Beginner: {len(selected_beginner)} pieces")
            print(f"  Medium: {len(selected_medium)} pieces")
            print(f"  Advanced: {len(selected_advanced)} pieces")
            print(f"  Total: {len(selected_beginner) + len(selected_medium) + len(selected_advanced)} pieces")
            
            # Split each category into train (80%) and test (20%)
            def split_pieces(pieces, train_ratio=0.8):
                pieces_copy = pieces[:]
                random.shuffle(pieces_copy)
                split_idx = int(len(pieces_copy) * train_ratio)
                return pieces_copy[:split_idx], pieces_copy[split_idx:]
            
            train_beginner, test_beginner = split_pieces(selected_beginner)
            train_medium, test_medium = split_pieces(selected_medium)
            train_advanced, test_advanced = split_pieces(selected_advanced)
            
            training_keys = train_beginner + train_medium + train_advanced
            test_keys = test_beginner + test_medium + test_advanced
            
            print(f"\nTrain/Test split:")
            print(f"  Train - Beginner: {len(train_beginner)}, Medium: {len(train_medium)}, Advanced: {len(train_advanced)}")
            print(f"  Test  - Beginner: {len(test_beginner)}, Medium: {len(test_medium)}, Advanced: {len(test_advanced)}")
            print(f"  Total train: {len(training_keys)}, Total test: {len(test_keys)}")
            
        else:
            # Use provided train/test files
            if train_files is None:
                training_keys = [
                    key for key in corpus_vio1.keys() 
                    if any(x in key for x in ['bach', 'mozart', 'beeth', 'mend', 'flower', 'wind'])
                ]
            else:
                training_keys = [key for key in corpus_vio1.keys() if any(x in key for x in train_files)]
            
            if test_files is None:
                test_keys = [key for key in corpus_vio1.keys() if key not in training_keys]
            else:
                test_keys = [key for key in corpus_vio1.keys() if any(x in key for x in test_files)]
        
        train_data = [v for k, v in corpus_vio1.items() if k in training_keys]
        test_data = [v for k, v in corpus_vio1.items() if k in test_keys]
        
        # Concatenate segments and create prefix label arrays
        train_segments = np.concatenate([x['segments'] for x in train_data], axis=0)
        train_lens = np.array([l for x in train_data for l in x['lens']])
        train_prefix = np.array([x['prefix_label'] for x in train_data for _ in x['lens']])
        
        test_segments = np.concatenate([x['segments'] for x in test_data], axis=0)
        test_lens = np.array([l for x in test_data for l in x['lens']])
        test_prefix = np.array([x['prefix_label'] for x in test_data for _ in x['lens']])
        
        print(f'\nFinal data shapes:')
        print(f'  Training data: {train_segments.shape}')
        print(f'  Testing data: {test_segments.shape}')
        
        # Print prefix token distribution
        from collections import Counter
        train_prefix_dist = Counter(train_prefix)
        test_prefix_dist = Counter(test_prefix)
        
        print(f'\nPrefix token distribution in training set:')
        for prefix_idx, count in sorted(train_prefix_dist.items()):
            print(f'  Token {prefix_idx}: {count} segments')
        
        print(f'\nPrefix token distribution in test set:')
        for prefix_idx, count in sorted(test_prefix_dist.items()):
            print(f'  Token {prefix_idx}: {count} segments')
        
        train_dataset = PrefixViolinFingeringDataset(train_segments, train_lens, train_prefix)
        test_dataset = PrefixViolinFingeringDataset(test_segments, test_lens, test_prefix)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.training_config['batch_size'], 
            shuffle=True, 
            num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.training_config['batch_size'], 
            shuffle=False, 
            num_workers=0
        )
        
        return train_loader, test_loader
    
    def prepare_autoregressive_inputs(self, batch):
        """Prepare inputs for autoregressive prediction"""
        joint_labels = batch['joint_label']
        seq_len = joint_labels.size(1)
        mid_idx = seq_len // 2
        
        prev_fingerings = joint_labels.clone()
        prev_fingerings[:, mid_idx:] = self.model.no_fingering_token
        target = joint_labels[:, mid_idx]
        
        return prev_fingerings, target, mid_idx
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            pitch = batch['pitch'].to(self.device)
            start = batch['start'].to(self.device)
            duration = batch['duration'].to(self.device)
            beat_type = batch['beat_type'].to(self.device)
            lengths = batch['length'].to(self.device)
            prefix_labels = batch['prefix_label'].to(self.device)
            
            prev_fingerings, target, mid_idx = self.prepare_autoregressive_inputs(batch)
            prev_fingerings = prev_fingerings.to(self.device)
            target = target.to(self.device)
            
            seq_len = pitch.size(1)
            padding_mask = torch.arange(seq_len, device=self.device)[None, :] >= lengths[:, None]
            
            # Forward pass with prefix labels
            joint_logits = self.model(
                pitch, start, duration, beat_type, 
                prev_fingerings, prefix_labels, padding_mask
            )
            
            # Compute loss
            loss = self.criterion(joint_logits, target)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.lora_params, self.training_config['grad_clip'])
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        self.scheduler.step()
        return total_loss / len(train_loader)
    
    @torch.no_grad()
    def evaluate(self, test_loader):
        """Evaluate on test set"""
        self.model.eval()
        
        all_string_preds = []
        all_string_targets = []
        all_finger_preds = []
        all_finger_targets = []
        all_joint_preds = []
        all_joint_targets = []
        
        total_loss = 0
        
        for batch in test_loader:
            pitch = batch['pitch'].to(self.device)
            start = batch['start'].to(self.device)
            duration = batch['duration'].to(self.device)
            beat_type = batch['beat_type'].to(self.device)
            lengths = batch['length'].to(self.device)
            prefix_labels = batch['prefix_label'].to(self.device)
            
            prev_fingerings, target, mid_idx = self.prepare_autoregressive_inputs(batch)
            prev_fingerings = prev_fingerings.to(self.device)
            target = target.to(self.device)
            
            string_target = batch['string'][:, mid_idx].to(self.device)
            finger_target = batch['finger'][:, mid_idx].to(self.device)
            
            seq_len = pitch.size(1)
            padding_mask = torch.arange(seq_len, device=self.device)[None, :] >= lengths[:, None]
            
            joint_logits = self.model(
                pitch, start, duration, beat_type,
                prev_fingerings, prefix_labels, padding_mask
            )
            loss = self.criterion(joint_logits, target)
            total_loss += loss.item()
            
            joint_pred = torch.argmax(joint_logits, dim=-1)
            string_pred = joint_pred // 5
            finger_pred = joint_pred % 5
            
            all_joint_preds.extend(joint_pred.cpu().numpy())
            all_joint_targets.extend(target.cpu().numpy())
            all_string_preds.extend(string_pred.cpu().numpy())
            all_string_targets.extend(string_target.cpu().numpy())
            all_finger_preds.extend(finger_pred.cpu().numpy())
            all_finger_targets.extend(finger_target.cpu().numpy())
        
        joint_p, joint_r, joint_f, _ = precision_recall_fscore_support(
            all_joint_targets, all_joint_preds, average='micro'
        )
        string_p, string_r, string_f, _ = precision_recall_fscore_support(
            all_string_targets, all_string_preds, average='micro'
        )
        finger_p, finger_r, finger_f, _ = precision_recall_fscore_support(
            all_finger_targets, all_finger_preds, average='micro'
        )
        
        return {
            'loss': total_loss / len(test_loader),
            'joint_f1': joint_f,
            'string_f1': string_f,
            'finger_f1': finger_f
        }
    
    def finetune(self, dataset_dir, labels_path, save_dir, train_files=None, test_files=None):
        """Main fine-tuning loop"""
        # Load piece labels
        self.piece_labels = self.load_piece_labels(labels_path)
        
        corpus = self.load_data(dataset_dir)
        corpus_seg = self.segment_corpus(corpus, self.base_config['seq_len'])
        train_loader, test_loader = self.create_dataloaders(corpus_seg, train_files, test_files)
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        best_joint_f1 = 0
        patience_counter = 0
        
        for epoch in range(self.training_config['n_epochs']):
            print(f"\n{'='*60}\nEpoch {epoch+1}/{self.training_config['n_epochs']}\n{'='*60}")
            
            train_loss = self.train_epoch(train_loader, epoch)
            print(f"\nTraining Loss: {train_loss:.4f}")
            
            test_metrics = self.evaluate(test_loader)
            print(f"Testing - Loss: {test_metrics['loss']:.4f}, "
                  f"Joint F1: {test_metrics['joint_f1']:.4f}, "
                  f"String F1: {test_metrics['string_f1']:.4f}, "
                  f"Finger F1: {test_metrics['finger_f1']:.4f}")
            
            if test_metrics['joint_f1'] > best_joint_f1:
                best_joint_f1 = test_metrics['joint_f1']
                patience_counter = 0
                save_path = os.path.join(save_dir, f'lora_prefix_best_model_{timestamp}.pt')
                
                # Save LoRA parameters separately (more efficient)
                lora_state_dict = {
                    name: param for name, param in self.model.named_parameters() 
                    if param.requires_grad
                }
                
                torch.save({
                    'epoch': epoch,
                    'lora_state_dict': lora_state_dict,
                    'full_model_state_dict': self.model.state_dict(),
                    'model': self.model,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'test_metrics': test_metrics,
                    'base_config': self.base_config,
                    'lora_config': self.lora_config,
                    'training_config': self.training_config,
                    'level_to_idx': self.level_to_idx,
                    'shift_to_idx': self.shift_to_idx,
                    'n_prefix_tokens': self.n_prefix_tokens
                }, save_path)
                print(f"✓ Saved best LoRA model (Joint F1: {best_joint_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.training_config['patience']:
                    print(f"\nEarly stopping after {epoch+1} epochs")
                    break


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='LoRA Fine-tuning with Prefix Tokens for Violin Fingering Model')
    parser.add_argument('--base-model', type=str, required=True,
                        help='Path to base model checkpoint')
    parser.add_argument('--dataset-dir', type=str, default='./TNUA_violin_fingering_dataset',
                        help='Directory containing CSV files')
    parser.add_argument('--labels-path', type=str, default='./piece_labels.csv',
                        help='Path to piece labels CSV file')
    parser.add_argument('--save-dir', type=str, default='./lora_prefix_models',
                        help='Directory to save LoRA fine-tuned models')
    parser.add_argument('--rank', type=int, default=8,
                        help='LoRA rank (default: 8)')
    parser.add_argument('--alpha', type=int, default=16,
                        help='LoRA alpha (default: 16)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='LoRA dropout (default: 0.1)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (default: 20)')
    
    args = parser.parse_args()
    
    lora_config = {
        'rank': args.rank,
        'alpha': args.alpha,
        'dropout': args.dropout
    }
    
    training_config = {
        'batch_size': args.batch_size,
        'n_epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': 1e-5,
        'grad_clip': 1.0,
        'patience': args.patience,
        'scheduler_t0': 10
    }
    
    print("="*60)
    print("LoRA Fine-tuning with Prefix Tokens Configuration")
    print("="*60)
    print(f"Base model: {args.base_model}")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Labels: {args.labels_path}")
    print(f"Save directory: {args.save_dir}")
    print(f"\nLoRA Configuration:")
    print(f"  Rank: {lora_config['rank']}")
    print(f"  Alpha: {lora_config['alpha']}")
    print(f"  Dropout: {lora_config['dropout']}")
    print(f"\nTraining Configuration:")
    print(f"  Batch size: {training_config['batch_size']}")
    print(f"  Learning rate: {training_config['learning_rate']}")
    print(f"  Epochs: {training_config['n_epochs']}")
    print(f"  Patience: {training_config['patience']}")
    print("="*60)
    
    finetuner = LoRAFineTuner(args.base_model, lora_config, training_config)
    finetuner.finetune(args.dataset_dir, args.labels_path, args.save_dir)


if __name__ == '__main__':
    main()