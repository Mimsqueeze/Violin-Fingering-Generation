import sys
from pathlib import Path
import pandas as pd
from music21 import converter, note, chord

# Beat type mapping
BEAT_TYPE_MAP = {
    'whole': 1,
    'half': 2,
    'quarter': 3,
    'eighth': 4,
    '16th': 5,
    '32nd': 6,
}

def get_beat_type(n):
    return BEAT_TYPE_MAP.get(n.duration.type, 0)

def extract_notes_from_part(part):
    rows = []

    for el in part.flat.notes:
        if isinstance(el, note.Note):
            rows.append(make_row(el))
        elif isinstance(el, chord.Chord):
            for n in el.notes:
                rows.append(make_row(n, chord_offset=el.offset))

    return rows

def make_row(n, chord_offset=None):
    offset = chord_offset if chord_offset is not None else n.offset

    return {
        "pitch": n.pitch.midi,
        "start": float(offset),
        "duration": float(n.duration.quarterLength),
        "beat_type": get_beat_type(n),
        "string": 0,
        "position": 0,
        "finger": 0,
    }

def convert_mxl_to_csv(mxl_path, out_csv):
    score = converter.parse(mxl_path)

    # Select the topmost part only
    if score.parts:
        part = score.parts[0]
    else:
        # Fallback: treat whole score as a single part
        part = score

    rows = extract_notes_from_part(part)
    df = pd.DataFrame(rows)

    # Enforce pitch range if desired
    df = df[(df.pitch >= 55) & (df.pitch <= 100)]

    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python mxl_to_csv.py <file.mxl | directory>")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    if input_path.is_file():
        convert_mxl_to_csv(input_path, input_path.with_suffix(".csv"))

    elif input_path.is_dir():
        for mxl_file in input_path.glob("*.mxl"):
            convert_mxl_to_csv(mxl_file, mxl_file.with_suffix(".csv"))

    else:
        print("Invalid path")

if __name__ == "__main__":
    main()
