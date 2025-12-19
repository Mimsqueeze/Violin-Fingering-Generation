import os
import pandas as pd
import numpy as np

dataset_dir = './TNUA_violin_fingering_dataset'  # adjust if needed
output_file = './position_statistics.txt'
labels_file = './piece_labels.csv'

files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]

# Store per-piece data for summary
all_position_props = {}
all_change_rates = []
piece_data = []

# Counters for majority positions
majority_pos1 = 0
majority_pos3 = 0

with open(output_file, 'w') as f_out:
    for file in files:
        path = os.path.join(dataset_dir, file)
        df = pd.read_csv(path)
        
        if 'position' not in df.columns:
            f_out.write(f"{file} does not have a 'position' column.\n\n")
            continue
        
        positions = df['position'].values
        total_notes = len(positions)
        
        # Count number of position changes
        position_changes = (positions[1:] != positions[:-1]).sum()
        position_change_rate = position_changes / total_notes if total_notes > 0 else 0.0
        all_change_rates.append(position_change_rate)
        
        # Compute position proportions
        position_counts = pd.Series(positions).value_counts(normalize=True).sort_index()
        prop_pos1 = position_counts.get(1, 0.0)  # proportion of position 1
        all_position_props.setdefault(1, []).append(prop_pos1)
        
        # Save per-piece data for labeling
        piece_data.append({
            'file': file,
            'prop_pos1': prop_pos1,
            'position_change_rate': position_change_rate
        })
        
        for pos, prop in position_counts.items():
            all_position_props.setdefault(pos, []).append(prop)
        
        f_out.write(f"File: {file}\n")
        for pos, prop in position_counts.items():
            f_out.write(f"position {pos}: {prop:.3f}\n")
        f_out.write(f"Number of position changes: {position_changes}\n")
        f_out.write(f"Position change rate: {position_change_rate:.3f}\n\n")
        
        # Determine majority position
        majority_position = position_counts.idxmax()
        if majority_position == 1:
            majority_pos1 += 1
        elif majority_position == 3:
            majority_pos3 += 1

# Convert piece data to DataFrame
df_pieces = pd.DataFrame(piece_data)

# Compute quartiles for labeling
q_pos1_lower = df_pieces['prop_pos1'].quantile(0.10)
q_pos1_upper = df_pieces['prop_pos1'].quantile(0.90)

q_shift_lower = df_pieces['position_change_rate'].quantile(0.10)
q_shift_upper = df_pieces['position_change_rate'].quantile(0.90)

# Assign labels
def assign_level_label(x):
    if x >= q_pos1_upper:
        return 'beginner'
    elif x <= q_pos1_lower:
        return 'advanced'
    else:
        return 'medium'

def assign_shift_label(x):
    if x >= q_shift_upper:
        return 'more shifting'
    elif x <= q_shift_lower:
        return 'less shifting'
    else:
        return 'normal shifting'

df_pieces['level_label'] = df_pieces['prop_pos1'].apply(assign_level_label)
df_pieces['shift_label'] = df_pieces['position_change_rate'].apply(assign_shift_label)

# Save labels to CSV
df_pieces.to_csv(labels_file, index=False)
print(f"Piece labels saved to {labels_file}")

# --- Optional: append to statistics file ---
with open(output_file, 'a') as f_out:
    f_out.write("\nPiece Labels:\n")
    for _, row in df_pieces.iterrows():
        f_out.write(f"{row['file']}: Level={row['level_label']}, Shifting={row['shift_label']}\n")
