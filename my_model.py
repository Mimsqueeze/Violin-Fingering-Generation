import os
import numpy as np
import pickle
from sklearn.metrics import precision_recall_fscore_support
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

class ViolinFingeringDataset(Dataset):
    """Dataset for violin fingering data from CSV files"""
    
    def __init__(self, segments, lens):
        self.segments = segments
        self.lens = lens
        
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        segment = self.segments[idx]
        length = self.lens[idx]
        
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
            'length': length
        }


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ViolinFingeringTransformer(nn.Module):
    """Autoregressive Transformer model for violin fingering prediction"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings for note features
        self.pitch_embedding = nn.Embedding(config['n_pitch_classes'], config['embedding_dim'])
        self.beat_embedding = nn.Embedding(config['n_beat_classes'], config['embedding_dim'])
        
        # Embedding for previous fingerings (joint string+finger)
        # Add 1 for a special "no fingering" token (used for the last position)
        self.fingering_embedding = nn.Embedding(config['n_joint_classes'] + 1, config['embedding_dim'])
        self.no_fingering_token = config['n_joint_classes']  # Index for "no fingering"
        
        # Project continuous features
        self.start_projection = nn.Linear(1, config['embedding_dim'])
        self.duration_projection = nn.Linear(1, config['embedding_dim'])
        
        # Input projection - now includes fingering embedding
        self.input_projection = nn.Linear(config['embedding_dim'] * 5, config['d_model'])
        self.input_norm = nn.LayerNorm(config['d_model'])
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(config['d_model'], dropout=config['dropout'])
        
        # Transformer encoder with causal mask
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=config['d_model'],
            nhead=config['n_heads'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, config['n_layers'])
        
        # Joint prediction head (string+finger) for the last position only
        self.joint_head = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model'] // 2),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['d_model'] // 2, config['n_joint_classes'])
        )
        
        self.lowest_pitch = config['lowest_pitch']
        self.pitch_for_invalid_note = config['pitch_for_invalid_note']
        
        # Load fingering mask
        self.register_buffer('fingering_mask', self._load_fingering_mask(config['fingering_mask_path']))
        
    def _load_fingering_mask(self, mask_path):
        """Load and convert fingering mask to tensor"""
        with open(mask_path, 'rb') as f:
            fingering_table = pickle.load(f)
        
        # Create mask tensor: [n_pitches, n_joint_classes]
        n_pitches = 46
        n_joint = 25
        
        mask = torch.zeros(n_pitches, n_joint, dtype=torch.bool)
        
        for pitch in range(55, 101):
            pitch_idx = pitch - 55
            valid_fingerings = fingering_table.get(pitch, [])
            
            for string, finger in valid_fingerings:
                joint_idx = string * 5 + finger
                mask[pitch_idx, joint_idx] = True
        
        return mask
        
    def forward(self, pitch, start, duration, beat_type, prev_fingerings, padding_mask=None):
        """
        Args:
            pitch: [batch, seq_len]
            start: [batch, seq_len]
            duration: [batch, seq_len]
            beat_type: [batch, seq_len]
            prev_fingerings: [batch, seq_len] - previous fingerings (with no_fingering_token at middle position)
            padding_mask: [batch, seq_len] boolean mask (True for padding positions)
        """
        batch_size, seq_len = pitch.shape
        mid_idx = seq_len // 2
        
        # Adjust pitch for embedding
        pitch_adjusted = (pitch - self.lowest_pitch).clamp(0, self.config['n_pitch_classes'] - 1)
        
        # Embed all features including previous fingerings
        pitch_emb = self.pitch_embedding(pitch_adjusted)
        beat_emb = self.beat_embedding(beat_type)
        fingering_emb = self.fingering_embedding(prev_fingerings)
        
        # Project continuous features
        start_emb = self.start_projection(start.unsqueeze(-1))
        duration_emb = self.duration_projection(duration.unsqueeze(-1))
        
        # Concatenate all embeddings (now includes previous fingerings)
        x = torch.cat([pitch_emb, start_emb, duration_emb, beat_emb, fingering_emb], dim=-1)
        
        # Project to model dimension
        x = self.input_projection(x)
        x = self.input_norm(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding 
        # The middle position can attend to all positions (including future ones)
        # Future positions just have no_fingering_token for their fingering embeddings
        transformer_output = self.transformer_encoder(
            x, 
            src_key_padding_mask=padding_mask
        )
        
        # Only predict for the MIDDLE position
        mid_position_output = transformer_output[:, mid_idx, :]  # [batch, d_model]
        
        # Predict joint string+finger for middle position
        joint_logits = self.joint_head(mid_position_output)  # [batch, n_joint_classes]
        
        # Apply fingering mask based on the middle pitch
        mid_pitch = pitch[:, mid_idx]  # [batch]
        joint_mask = self.create_joint_mask_single(mid_pitch)  # [batch, n_joint_classes]
        joint_logits = joint_logits.masked_fill(~joint_mask, -1000)
        
        return joint_logits
    
    def create_joint_mask_single(self, pitches):
        """
        Create mask for valid string+finger combinations for single pitches
        
        Args:
            pitches: [batch]
        Returns:
            mask: [batch, n_joint_classes]
        """
        device = pitches.device
        
        # Clamp pitches to valid range (55-100)
        pitch_clamped = pitches.clamp(55, 100)
        pitch_idx = pitch_clamped - 55
        
        # Get mask for each pitch
        mask = self.fingering_mask[pitch_idx]  # [batch, n_joint]
        
        return mask


class ViolinFingeringTrainer:
    """Training class for the autoregressive violin fingering model"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = ViolinFingeringTransformer(config).to(self.device)
        
        # Loss function (no reduction needed since we predict single position)
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config['scheduler_t0'],
            T_mult=1,
            eta_min=config['learning_rate'] * 0.01
        )
        
    def load_data(self, dataset_dir):
        """Load and process data from CSV files"""
        print("Loading training data...")
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
        """Segment sequences into fixed-length chunks"""
        def _segment_sequence(sequence, max_len, pitch_for_invalid):
            dt = sequence.dtype
            N = len(sequence)
            segments = []
            valid_lens = []

            padding_template = (pitch_for_invalid, -1, 0, 0, 0, 0, 0)

            # Loop over all positions to create overlapping segments
            for idx in range(N):
                valid_len = min(idx + 1, max_len)

                # Compute padding at the start
                pad_len = max_len - valid_len
                padding = np.array([padding_template] * pad_len, dtype=dt) if pad_len > 0 else np.empty((0,), dtype=dt)

                # Take last valid_len notes ending at idx
                start_idx = max(0, idx - max_len + 1)
                notes = sequence[start_idx: idx + 1]

                # Combine padding + notes
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
                self.config['pitch_for_invalid_note']
            )
            corpus_seg[key] = {
                'segments': segments,
                'lens': valid_lens
            }
        
        total_segments = sum([v['segments'].shape[0] for v in corpus_seg.values()])
        print(f'Total number of segments: {total_segments}')
        return corpus_seg
    
    def create_dataloaders(self, corpus_seg):
        """Create training and testing dataloaders"""
        corpus_vio1 = {k: v for k, v in corpus_seg.items()}
        
        training_keys = [
            key for key in corpus_vio1.keys() 
            if any(x in key for x in ['bach', 'mozart', 'beeth', 'mend', 'flower', 'wind'])
        ]
        
        train_data = [v for k, v in corpus_vio1.items() if k in training_keys]
        test_data = [v for k, v in corpus_vio1.items() if k not in training_keys]
        
        train_segments = np.concatenate([x['segments'] for x in train_data], axis=0)
        train_lens = np.array([l for x in train_data for l in x['lens']])
        test_segments = np.concatenate([x['segments'] for x in test_data], axis=0)
        test_lens = np.array([l for x in test_data for l in x['lens']])
        
        print(f'Training data shape: {train_segments.shape}')
        print(f'Testing data shape: {test_segments.shape}')
        
        train_dataset = ViolinFingeringDataset(train_segments, train_lens)
        test_dataset = ViolinFingeringDataset(test_segments, test_lens)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=0)
        
        return train_loader, test_loader
    
    def prepare_autoregressive_inputs(self, batch):
        """
        Prepare inputs for autoregressive prediction.
        For positions 0 to mid_idx-1: use ground truth fingerings
        For position mid_idx (middle): use special no_fingering_token
        For positions mid_idx+1 to end: use special no_fingering_token (masked out)
        """
        joint_labels = batch['joint_label']
        seq_len = joint_labels.size(1)
        mid_idx = seq_len // 2
        
        # Create previous fingerings tensor
        prev_fingerings = joint_labels.clone()
        
        # Mask out the middle position and all positions after it
        prev_fingerings[:, mid_idx:] = self.model.no_fingering_token
        
        # Target is only the middle position
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
            
            # Prepare autoregressive inputs
            prev_fingerings, target, mid_idx = self.prepare_autoregressive_inputs(batch)
            prev_fingerings = prev_fingerings.to(self.device)
            target = target.to(self.device)
            
            seq_len = pitch.size(1)
            padding_mask = torch.arange(seq_len, device=self.device)[None, :] >= lengths[:, None]
            
            # Forward pass - only predicts middle position
            joint_logits = self.model(pitch, start, duration, beat_type, prev_fingerings, padding_mask)
            
            # Compute loss only on middle position
            loss = self.criterion(joint_logits, target)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
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
            
            # Prepare autoregressive inputs
            prev_fingerings, target, mid_idx = self.prepare_autoregressive_inputs(batch)
            prev_fingerings = prev_fingerings.to(self.device)
            target = target.to(self.device)
            
            # Also get individual string/finger targets for middle position
            string_target = batch['string'][:, mid_idx].to(self.device)
            finger_target = batch['finger'][:, mid_idx].to(self.device)
            
            seq_len = pitch.size(1)
            padding_mask = torch.arange(seq_len, device=self.device)[None, :] >= lengths[:, None]
            
            # Forward pass
            joint_logits = self.model(pitch, start, duration, beat_type, prev_fingerings, padding_mask)
            
            # Compute loss
            loss = self.criterion(joint_logits, target)
            total_loss += loss.item()
            
            # Get predictions
            joint_pred = torch.argmax(joint_logits, dim=-1)
            
            # Convert joint predictions back to string and finger
            string_pred = joint_pred // 5
            finger_pred = joint_pred % 5
            
            # Collect predictions and targets (only middle position)
            all_joint_preds.extend(joint_pred.cpu().numpy())
            all_joint_targets.extend(target.cpu().numpy())
            all_string_preds.extend(string_pred.cpu().numpy())
            all_string_targets.extend(string_target.cpu().numpy())
            all_finger_preds.extend(finger_pred.cpu().numpy())
            all_finger_targets.extend(finger_target.cpu().numpy())
        
        # Compute metrics
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
    
    def train(self, dataset_dir, save_dir):
        """Main training loop"""
        corpus = self.load_data(dataset_dir)
        corpus_seg = self.segment_corpus(corpus, self.config['seq_len'])
        train_loader, test_loader = self.create_dataloaders(corpus_seg)
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        best_joint_f1 = 0
        patience_counter = 0
        
        for epoch in range(self.config['n_epochs']):
            print(f"\n{'='*60}\nEpoch {epoch+1}/{self.config['n_epochs']}\n{'='*60}")
            
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
                save_path = os.path.join(save_dir, f'best_model_{timestamp}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'model': self.model,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'test_metrics': test_metrics,
                    'config': self.config
                }, save_path)
                print(f"✓ Saved best model (Joint F1: {best_joint_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.config['patience']:
                    print(f"\nEarly stopping after {epoch+1} epochs")
                    break


def main():
    config = {
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 8,
        'dim_feedforward': 256,
        'embedding_dim': 64,
        'dropout': 0.1,
        'seq_len': 32,
        'n_pitch_classes': 47,
        'n_beat_classes': 7,
        'n_joint_classes': 25,
        'lowest_pitch': 55,
        'pitch_for_invalid_note': 101,
        'batch_size': 32,
        'n_epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'grad_clip': 1.0,
        'patience': 10,
        'scheduler_t0': 10,
        'dataset_dir': './TNUA_violin_fingering_dataset',
        'save_dir': './model_transformer',
        'fingering_mask_path': './fingering_masks/violin_fingerings.pkl'
    }
    
    trainer = ViolinFingeringTrainer(config)
    trainer.train(config['dataset_dir'], config['save_dir'])

if __name__ == '__main__':
    main()