"""
Generate validity masks for violin fingerings.
Modified to use string indices 0-4 to match the dataset encoding.
"""

import numpy as np
import json
import pickle
from pathlib import Path


class ViolinFingeringGenerator:
    """Generate all valid violin fingerings for MIDI pitches"""
    
    def __init__(self):
        # Open string pitches (MIDI) - using 0-4 encoding
        self.open_strings = {
            0: None,  # null string
            1: 55,    # G string (G3)
            2: 62,    # D string (D4)
            3: 69,    # A string (A4)
            4: 76     # E string (E5)
        }
        
        # String names for display
        self.string_names = {
            0: 'null',
            1: 'G',
            2: 'D',
            3: 'A',
            4: 'E'
        }
        
        # Maximum reachable pitch on each string (approximate)
        self.string_max_pitch = {
            1: 76,   # G string can reach E5
            2: 83,   # D string can reach B5
            3: 91,   # A string can reach G6
            4: 100   # E string can reach E7
        }
    
    def generate_all_fingerings_for_pitch(self, pitch):
        """
        Generate all valid (string, finger) combinations for a given pitch.
        String indices: 0=null, 1=G, 2=D, 3=A, 4=E
        """
        valid_fingerings = []
        
        # Try all real strings (1-4, excluding null string 0)
        for string in [1, 2, 3, 4]:
            open_pitch = self.open_strings[string]
            
            # Skip if pitch is below open string
            if pitch < open_pitch:
                continue
            
            # Skip if pitch is beyond practical range of this string
            if pitch > self.string_max_pitch[string]:
                continue
            
            # Calculate semitones above open string
            semitones_above_open = pitch - open_pitch
            
            # Finger 0: Can only play open string (0 semitones above)
            if semitones_above_open == 0:
                valid_fingerings.append((string, 0))
            
            # Fingers 1-4: Can play any pitch above the open string
            if semitones_above_open > 0:
                for finger in [1, 2, 3, 4]:
                    if finger <= semitones_above_open:
                        valid_fingerings.append((string, finger))
        
        return valid_fingerings
    
    def generate_mask_table(self, pitch_range=(55, 101), save_txt=True, save_json=True, save_pickle=True):
        """Generate complete fingering table for all pitches."""
        fingering_table = {}
        
        print("=" * 80)
        print("VIOLIN FINGERING MASK GENERATOR (String indices 0-4)")
        print("=" * 80)
        print(f"\nGenerating fingerings for MIDI pitches {pitch_range[0]} to {pitch_range[1]-1}")
        print(f"(G3 to E7)\n")
        
        for pitch in range(pitch_range[0], pitch_range[1]):
            fingerings = self.generate_all_fingerings_for_pitch(pitch)
            fingering_table[pitch] = fingerings
            
            # Convert MIDI to note name for display
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            note_name = note_names[pitch % 12]
            octave = (pitch // 12) - 1
            
            print(f"Pitch {pitch:3d} ({note_name:2s}{octave}): {len(fingerings):2d} fingerings")
            for string, finger in fingerings:
                string_name = self.string_names[string]
                print(f"  String {string} ({string_name}), finger {finger}")
        
        print(f"\n{'=' * 80}")
        print("SUMMARY")
        print("=" * 80)
        total_fingerings = sum(len(f) for f in fingering_table.values())
        print(f"Total pitches: {len(fingering_table)}")
        print(f"Total valid fingerings: {total_fingerings}")
        print(f"Average fingerings per pitch: {total_fingerings / len(fingering_table):.1f}")
        
        # Save to files
        output_dir = Path("./fingering_masks")
        output_dir.mkdir(exist_ok=True)
        
        if save_txt:
            txt_path = output_dir / "violin_fingerings.txt"
            with open(txt_path, 'w') as f:
                f.write("VIOLIN FINGERING LOOKUP TABLE (String 0-4 encoding)\n")
                f.write("=" * 80 + "\n\n")
                
                for pitch in sorted(fingering_table.keys()):
                    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                    note_name = note_names[pitch % 12]
                    octave = (pitch // 12) - 1
                    
                    fingerings = fingering_table[pitch]
                    f.write(f"{pitch:3d} ({note_name:2s}{octave}): ")
                    
                    fingering_strs = []
                    for string, finger in fingerings:
                        string_name = self.string_names[string]
                        fingering_strs.append(f"({string}={string_name},{finger})")
                    
                    f.write(", ".join(fingering_strs))
                    f.write("\n")
            
            print(f"\n✓ Saved text file: {txt_path}")
        
        if save_json:
            json_table = {
                str(pitch): [list(f) for f in fingerings]
                for pitch, fingerings in fingering_table.items()
            }
            
            json_path = output_dir / "violin_fingerings.json"
            with open(json_path, 'w') as f:
                json.dump(json_table, f, indent=2)
            
            print(f"✓ Saved JSON file: {json_path}")
        
        if save_pickle:
            pickle_path = output_dir / "violin_fingerings.pkl"
            with open(pickle_path, 'wb') as f:
                pickle.dump(fingering_table, f)
            
            print(f"✓ Saved pickle file: {pickle_path}")
        
        print(f"\n{'=' * 80}\n")
        
        return fingering_table


def main():
    """Main function to generate fingering masks"""
    generator = ViolinFingeringGenerator()
    
    # Generate complete table
    fingering_table = generator.generate_mask_table(
        pitch_range=(55, 101),
        save_txt=True,
        save_json=True,
        save_pickle=True
    )
    
    print("\nEXAMPLES:")
    print("=" * 80)
    
    example_pitches = [
        (55, "G3 - Lowest note (open G string)"),
        (69, "A4 - Concert A (440 Hz)"),
        (76, "E5 - Highest open string"),
    ]
    
    for pitch, description in example_pitches:
        if pitch in fingering_table:
            fingerings = fingering_table[pitch]
            print(f"\n{description} (MIDI {pitch}):")
            print(f"  {len(fingerings)} ways to play:")
            for string, finger in fingerings:
                string_name = generator.string_names[string]
                print(f"    - String {string} ({string_name}), finger {finger}")


if __name__ == "__main__":
    main()