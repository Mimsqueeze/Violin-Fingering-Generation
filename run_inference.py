# run_inference.py

import numpy as np
from music21 import stream, note, metadata, tempo, instrument, articulations, meter, duration as m21_duration
from pathlib import Path
import torch
import datetime
from my_model import *
from lora_finetune import *

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def load_model(checkpoint_path, level='beginner', shift_style='less shifting'):
    """
    Load model from checkpoint. Supports both regular and LoRA models.
    
    Args:
        checkpoint_path: Path to model checkpoint
        level: Skill level for LoRA models ('beginner', 'medium', 'advanced')
        shift_style: Shifting style for LoRA models ('less shifting', 'normal shifting', 'more shifting')
    
    Returns:
        model, device, config, is_lora, prefix_idx
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Check if this is a LoRA model
    is_lora = 'lora_config' in checkpoint or 'lora_state_dict' in checkpoint
    
    if is_lora:
        print(f"\n{'='*60}")
        print("LoRA Model Detected")
        print(f"{'='*60}")
        
        # Get LoRA-specific info
        level_to_idx = checkpoint.get('level_to_idx', {'beginner': 0, 'medium': 1, 'advanced': 2})
        shift_to_idx = checkpoint.get('shift_to_idx', {'less shifting': 0, 'normal shifting': 1, 'more shifting': 2})
        
        # Calculate prefix index
        level_idx = level_to_idx.get(level)
        shift_idx = shift_to_idx.get(shift_style)
        prefix_idx = level_idx * 3 + shift_idx
        
        print(f"Style Configuration:")
        print(f"  Level: {level} (index {level_idx})")
        print(f"  Shifting: {shift_style} (index {shift_idx})")
        print(f"  Prefix Token: {prefix_idx}")
        print(f"{'='*60}\n")
        
        # Load the full model (includes base model + LoRA + prefix)
        if 'model' in checkpoint:
            model = checkpoint['model'].to(device)
        else:
            raise KeyError("LoRA checkpoint must contain 'model'")
        
        model.eval()
        config = checkpoint.get('base_config', checkpoint.get('config', {}))
        
    else:
        print(f"\n{'='*60}")
        print("Regular Model Detected (No LoRA)")
        print(f"{'='*60}\n")
        
        prefix_idx = None
        
        # Load regular model
        if 'model' in checkpoint:
            model = checkpoint['model'].to(device)
        elif 'model_state_dict' in checkpoint:
            config = checkpoint['config']
            model = ViolinFingeringTransformer(config).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise KeyError("Checkpoint does not contain 'model' or 'model_state_dict'")
        
        model.eval()
        config = checkpoint.get('config', {})
    
    epoch = checkpoint.get('epoch', 'unknown')
    test_metrics = checkpoint.get('test_metrics', {})
    
    print(f"Model loaded successfully from epoch {epoch}")
    if test_metrics:
        print(f"Test metrics: {test_metrics}")
    
    return model, device, config, is_lora, prefix_idx

def predict_autoregressive(model, device, pitch, start, duration, beat_type, seq_len=32, 
                          is_lora=False, prefix_idx=None):
    """
    Run autoregressive inference on a sequence.
    Uses a sliding window approach where we predict one note at a time.
    
    Args:
        model: The loaded model
        device: torch device
        pitch: numpy array of pitches [N]
        start: numpy array of start times [N]
        duration: numpy array of durations [N]
        beat_type: numpy array of beat types [N]
        seq_len: sequence length the model was trained on (default 32)
        is_lora: whether this is a LoRA model
        prefix_idx: prefix token index for LoRA models
    
    Returns:
        List of dictionaries with predictions
    """
    N = len(pitch)
    string_predictions = []
    finger_predictions = []
    
    # Get special token for "no fingering"
    if is_lora:
        no_fingering_token = model.base_model.no_fingering_token
        config = model.base_model.config
    else:
        no_fingering_token = model.no_fingering_token
        config = model.config
    
    mid_idx = seq_len // 2
    
    print(f"\nPredicting fingerings autoregressively for {N} notes...")
    print(f"Sequence length: {seq_len}, Middle index: {mid_idx}")
    if is_lora:
        print(f"Using prefix token: {prefix_idx}")
    
    with torch.no_grad():
        for i in range(N):
            # Create a window of seq_len notes centered around position i
            # We want to predict the fingering at position i (the middle of the window)
            
            # Determine the start and end of the window
            start_idx = max(0, i - mid_idx)
            end_idx = min(N, i + (seq_len - mid_idx))
            
            # Extract window data
            window_pitch = pitch[start_idx:end_idx]
            window_start = start[start_idx:end_idx]
            window_duration = duration[start_idx:end_idx]
            window_beat = beat_type[start_idx:end_idx]
            
            # Pad if necessary
            current_len = len(window_pitch)
            pad_left = max(0, mid_idx - i)
            pad_right = max(0, seq_len - current_len - pad_left)
            
            if pad_left > 0 or pad_right > 0:
                # Use padding values
                pitch_for_invalid = config.get('pitch_for_invalid_note', 101)
                window_pitch = np.pad(window_pitch, (pad_left, pad_right), constant_values=pitch_for_invalid)
                window_start = np.pad(window_start, (pad_left, pad_right), constant_values=-1)
                window_duration = np.pad(window_duration, (pad_left, pad_right), constant_values=0)
                window_beat = np.pad(window_beat, (pad_left, pad_right), constant_values=0)
            
            # Create fingering input
            # For positions before mid_idx in the window: use predicted fingerings
            # For mid_idx and after: use no_fingering_token
            window_fingerings = np.full(seq_len, no_fingering_token, dtype=np.int64)
            
            # Fill in previously predicted fingerings
            for j in range(mid_idx):
                actual_idx = start_idx + j
                if 0 <= actual_idx < i:  # Only use already predicted fingerings
                    # Convert string and finger to joint label
                    joint_label = string_predictions[actual_idx] * 5 + finger_predictions[actual_idx]
                    window_fingerings[j] = joint_label
            
            # Convert to tensors
            pitch_tensor = torch.LongTensor(window_pitch).unsqueeze(0).to(device)
            start_tensor = torch.FloatTensor(window_start).unsqueeze(0).to(device)
            duration_tensor = torch.FloatTensor(window_duration).unsqueeze(0).to(device)
            beat_tensor = torch.LongTensor(window_beat).unsqueeze(0).to(device)
            fingering_tensor = torch.LongTensor(window_fingerings).unsqueeze(0).to(device)
            
            # Predict (returns logits for middle position only)
            if is_lora:
                # LoRA model needs prefix_labels
                prefix_tensor = torch.LongTensor([prefix_idx]).to(device)
                joint_logits = model(
                    pitch_tensor,
                    start_tensor,
                    duration_tensor,
                    beat_tensor,
                    fingering_tensor,
                    prefix_tensor
                )
            else:
                # Regular model
                joint_logits = model(
                    pitch_tensor,
                    start_tensor,
                    duration_tensor,
                    beat_tensor,
                    fingering_tensor
                )
            
            # Get prediction
            joint_pred = torch.argmax(joint_logits, dim=-1).item()
            
            # Decompose joint prediction
            string_pred = joint_pred // 5
            finger_pred = joint_pred % 5
            
            string_predictions.append(string_pred)
            finger_predictions.append(finger_pred)
            
            if (i + 1) % 50 == 0:
                print(f"  Predicted {i + 1}/{N} notes...")
    
    # Format results
    results = [
        {
            'pitch': int(p),
            'start': float(s),
            'duration': float(d),
            'beat_type': int(bt),
            'string': int(st),
            'finger': int(fing)
        }
        for p, s, d, bt, st, fing in zip(
            pitch, start, duration, beat_type,
            string_predictions, finger_predictions
        )
    ]
    
    print(f"  Predicted all {N} notes!")
    return results

def duration_to_type_and_dots(duration_value):
    """
    Convert a duration value to note type and number of dots.
    Returns (type_string, num_dots, quarterLength, is_triplet) tuple.
    """
    tolerance = 0.01
    
    # Map of duration -> (type, dots)
    duration_map = {
        # Dotted notes
        6.0: ('whole', 1), 3.0: ('half', 1), 1.5: ('quarter', 1),
        0.75: ('eighth', 1), 0.375: ('16th', 1), 0.1875: ('32nd', 1),
        # Regular notes
        4.0: ('whole', 0), 2.0: ('half', 0), 1.0: ('quarter', 0),
        0.5: ('eighth', 0), 0.25: ('16th', 0), 0.125: ('32nd', 0), 0.0625: ('64th', 0),
        # Double dotted
        7.0: ('whole', 2), 3.5: ('half', 2), 1.75: ('quarter', 2), 0.875: ('eighth', 2),
    }
    
    # Triplet map
    triplet_map = {
        2.6667: ('whole', 0), 1.3333: ('half', 0), 0.6667: ('quarter', 0),
        0.3333: ('eighth', 0), 0.1667: ('16th', 0), 0.0833: ('32nd', 0),
    }
    
    # Check for triplets first
    for dur_val, (note_type, dots) in triplet_map.items():
        if abs(duration_value - dur_val) < tolerance:
            return note_type, dots, dur_val, True
    
    # Check for regular/dotted notes
    for dur_val, (note_type, dots) in duration_map.items():
        if abs(duration_value - dur_val) < tolerance:
            return note_type, dots, dur_val, False
    
    # Find closest standard duration
    all_durations = list(duration_map.keys()) + list(triplet_map.keys())
    closest_dur = min(all_durations, key=lambda x: abs(x - duration_value))
    note_type, dots, actual_dur, is_triplet = duration_to_type_and_dots(closest_dur)
    return note_type, dots, actual_dur, is_triplet

def sequence_to_musicxml(sequence, filename, include_fingerings=False, time_signature='4/4'):
    """
    Converts a predicted sequence to MusicXML for sheet music.
    
    Args:
        sequence: List of note dictionaries with pitch, start, duration, beat_type, string, finger
        filename: Output MusicXML filename
        include_fingerings: If True, displays string+finger notation above notes (e.g., A-2)
        time_signature: Time signature string like '4/4', '3/4', etc.
    """
    score = stream.Score()
    score.append(metadata.Metadata(title="Violin Fingering Prediction"))

    violin_part = stream.Part()
    violin_part.append(instrument.Violin())
    violin_part.append(tempo.MetronomeMark(number=80))

    string_map = {0: 'null', 1: 'G', 2: 'D', 3: 'A', 4: 'E'}
    finger_map = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4'}

    # Parse time signature
    numerator, denominator = map(int, time_signature.split('/'))
    beats_per_measure = numerator * (4.0 / denominator)
    
    # Separate pickup notes (start < 0) from main notes
    pickup_notes = [n for n in sequence if n['start'] < 0]
    main_notes = [n for n in sequence if n['start'] >= 0]
    
    # Process pickup measure
    if pickup_notes:
        pickup_notes = sorted(pickup_notes, key=lambda x: x['start'])
        pickup_measure = stream.Measure(number=0)
        pickup_duration = sum(n['duration'] for n in pickup_notes)
        pickup_measure.timeSignature = meter.TimeSignature(time_signature)
        
        for n in pickup_notes:
            n_obj = note.Note(n['pitch'])
            note_type, num_dots, actual_duration, is_triplet = duration_to_type_and_dots(n['duration'])
            
            if is_triplet:
                n_obj.duration = m21_duration.Duration(type=note_type)
                n_obj.duration.appendTuplet(m21_duration.Tuplet(3, 2))
            else:
                n_obj.duration.type = note_type
                n_obj.quarterLength = actual_duration
                n_obj.duration.dots = num_dots

            if include_fingerings:
                string_name = string_map.get(n['string'], '')
                finger_name = finger_map.get(n['finger'], str(n['finger']))
                if string_name and string_name != 'null':
                    fingering_text = f"{string_name}-{finger_name}"
                    n_obj.articulations.append(articulations.Fingering(fingering_text))

            pickup_measure.append(n_obj)
        
        pickup_measure.padAsAnacrusis()
        violin_part.append(pickup_measure)
        print(f"Created pickup measure with duration {pickup_duration:.2f} quarter notes")
    
    # Process main notes into measures
    if main_notes:
        current_measure = stream.Measure(number=1)
        if not pickup_notes:
            current_measure.timeSignature = meter.TimeSignature(time_signature)
        
        current_position = 0.0
        measure_number = 1
        
        for n in main_notes:
            n_obj = note.Note(n['pitch'])
            note_type, num_dots, actual_duration, is_triplet = duration_to_type_and_dots(n['duration'])
            
            if is_triplet:
                n_obj.duration = m21_duration.Duration(type=note_type)
                n_obj.duration.appendTuplet(m21_duration.Tuplet(3, 2))
            else:
                n_obj.duration.type = note_type
                n_obj.quarterLength = actual_duration
                n_obj.duration.dots = num_dots

            if include_fingerings:
                string_name = string_map.get(n['string'], '')
                finger_name = finger_map.get(n['finger'], str(n['finger']))
                if string_name and string_name != 'null':
                    fingering_text = f"{string_name}-{finger_name}"
                    n_obj.articulations.append(articulations.Fingering(fingering_text))

            # Check if adding this note would exceed measure length
            if current_position + actual_duration > beats_per_measure + 0.01:
                violin_part.append(current_measure)
                measure_number += 1
                current_measure = stream.Measure(number=measure_number)
                current_position = 0.0
            
            current_measure.append(n_obj)
            current_position += actual_duration
        
        # Append the last measure
        if len(current_measure.notes) > 0:
            violin_part.append(current_measure)

    score.append(violin_part)
    score.write('musicxml', filename)
    print(f"Saved MusicXML: {filename}")

def load_csv(csv_path):
    """Load data from CSV file"""
    print(f"Loading CSV: {csv_path}")
    with open(csv_path) as f:
        data = np.genfromtxt(
            f, 
            delimiter=',', 
            names=True, 
            dtype=[('int'), ('float'), ('float'), ('int'), ('int'), ('int'), ('int')]
        )
    
    pitch = data['pitch']
    start = data['start']
    duration = data['duration']
    beat_type = data['beat_type']
    string = data['string']
    position = data['position']
    finger = data['finger']
    
    print(f"Loaded {len(pitch)} notes from CSV")
    print(f"Duration range: {duration.min():.3f} - {duration.max():.3f} quarter notes")
    
    pickup_count = np.sum(start < 0)
    if pickup_count > 0:
        print(f"Found {pickup_count} pickup notes (start < 0)")
    
    return pitch, start, duration, beat_type, string, position, finger

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Violin Fingering Autoregressive Inference (Regular and LoRA Models)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to CSV file with note data (optional)')
    parser.add_argument('--output-dir', type=str, default='./musicxml_output',
                        help='Output directory for MusicXML files')
    parser.add_argument('--output-name', type=str, default='prediction',
                        help='Base name for output files')
    parser.add_argument('--time-signature', type=str, default='4/4',
                        help='Time signature (e.g., 4/4, 3/4, 6/8)')
    parser.add_argument('--seq-len', type=int, default=32,
                        help='Sequence length used during training (default: 32)')
    
    # LoRA-specific arguments
    parser.add_argument('--level', type=str, default='medium',
                        choices=['beginner', 'medium', 'advanced'],
                        help='Skill level for LoRA models (default: medium)')
    parser.add_argument('--shift', type=str, default='normal shifting',
                        choices=['less shifting', 'normal shifting', 'more shifting'],
                        help='Shifting style for LoRA models (default: normal shifting)')
    
    args = parser.parse_args()
    
    # Load model (automatically detects if LoRA or regular)
    model, device, config, is_lora, prefix_idx = load_model(
        args.checkpoint, 
        level=args.level, 
        shift_style=args.shift
    )
    
    # Get sequence length from config if available
    seq_len = config.get('seq_len', args.seq_len)
    print(f"Using sequence length: {seq_len}")

    # Load data from CSV or use example
    ground_truth_sequence = None
    if args.csv:
        pitch, start, duration, beat_type, gt_string, gt_position, gt_finger = load_csv(args.csv)
        
        ground_truth_sequence = [
            {
                'pitch': int(p), 'start': float(s), 'duration': float(d),
                'beat_type': int(bt), 'string': int(st), 'finger': int(fing)
            }
            for p, s, d, bt, st, fing in zip(pitch, start, duration, beat_type, gt_string, gt_finger)
        ]
    else:
        print("\nNo CSV provided, using example sequence...")
        pitch = np.array([67, 60, 62, 64, 65, 67, 69])
        start = np.array([-1.0, 0.0, 1.5, 3.0, 4.0, 5.5, 7.0])
        duration = np.array([1.0, 1.5, 1.5, 1.0, 1.5, 1.5, 2.0])
        beat_type = np.array([3, 3, 3, 3, 3, 3, 2])

    # Run autoregressive predictions
    print("\nRunning autoregressive inference...")
    predictions = predict_autoregressive(
        model, device, pitch, start, duration, beat_type, 
        seq_len=seq_len, is_lora=is_lora, prefix_idx=prefix_idx
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Display results
    print(f"\n{'='*60}")
    print(f"Predictions:")
    print(f"{'='*60}")
    for j, pred in enumerate(predictions):
        string_name = {0: 'null', 1: 'G', 2: 'D', 3: 'A', 4: 'E'}.get(pred['string'], '?')
        note_type, num_dots, actual_dur, is_triplet = duration_to_type_and_dots(pred['duration'])
        dotted_str = " (dotted)" if num_dots > 0 else ""
        triplet_str = " (triplet)" if is_triplet else ""
        pickup_str = " [PICKUP]" if pred['start'] < 0 else ""
        
        if ground_truth_sequence:
            gt = ground_truth_sequence[j]
            gt_string_name = {0: 'null', 1: 'G', 2: 'D', 3: 'A', 4: 'E'}.get(gt['string'], '?')
            match_str = "✓" if (pred['string'] == gt['string'] and pred['finger'] == gt['finger']) else "✗"
            print(f"  Note {j}: Pitch={pred['pitch']}, Start={pred['start']:.2f}, "
                  f"Duration={pred['duration']:.2f} ({note_type}{dotted_str}{triplet_str}){pickup_str}")
            print(f"    Pred: String={string_name} ({pred['string']}), Finger={pred['finger']}")
            print(f"    GT:   String={gt_string_name} ({gt['string']}), Finger={gt['finger']} {match_str}")
        else:
            print(f"  Note {j}: Pitch={pred['pitch']}, Start={pred['start']:.2f}, "
                  f"Duration={pred['duration']:.2f} ({note_type}{dotted_str}{triplet_str}){pickup_str}, "
                  f"String={string_name} ({pred['string']}), Finger={pred['finger']}")
    
    # Save MusicXML files
    print(f"\n{'='*60}")
    print("Saving prediction MusicXML files...")
    
    # Add style info to output name if LoRA
    if is_lora:
        output_name = f"{args.output_name}_{args.level}_{args.shift.replace(' ', '_')}"
    else:
        output_name = args.output_name
    
    sequence_to_musicxml(
        predictions, 
        output_dir / f'{output_name}_{timestamp}_prediction.musicxml', 
        include_fingerings=True,
        time_signature=args.time_signature
    )
    
    # Save ground truth files if available
    if ground_truth_sequence:
        print("Saving ground truth MusicXML files...")
        sequence_to_musicxml(
            ground_truth_sequence, 
            output_dir / f'{output_name}_{timestamp}_blank.musicxml', 
            include_fingerings=False,
            time_signature=args.time_signature
        )
        sequence_to_musicxml(
            ground_truth_sequence, 
            output_dir / f'{output_name}_{timestamp}_groundtruth.musicxml', 
            include_fingerings=True,
            time_signature=args.time_signature
        )
        
        # Calculate accuracy
        total_notes = len(predictions)
        string_matches = sum(1 for p, gt in zip(predictions, ground_truth_sequence) if p['string'] == gt['string'])
        finger_matches = sum(1 for p, gt in zip(predictions, ground_truth_sequence) if p['finger'] == gt['finger'])
        full_matches = sum(1 for p, gt in zip(predictions, ground_truth_sequence) 
                          if p['string'] == gt['string'] and p['finger'] == gt['finger'])
        
        print(f"\n{'='*60}")
        print("Accuracy Metrics:")
        print(f"  String accuracy:   {string_matches}/{total_notes} ({100*string_matches/total_notes:.1f}%)")
        print(f"  Finger accuracy:   {finger_matches}/{total_notes} ({100*finger_matches/total_notes:.1f}%)")
        print(f"  Full match:        {full_matches}/{total_notes} ({100*full_matches/total_notes:.1f}%)")
    
    print(f"\n{'='*60}")
    print(f"Inference complete! Output saved to {output_dir}")
    if is_lora:
        print(f"Style: {args.level} + {args.shift}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()