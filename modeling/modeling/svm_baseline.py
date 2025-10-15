from pathlib import Path
import pandas as pd

# Always start path relative to this script file
BASE_DIR = Path(__file__).resolve().parent.parent   # go up to repo root
DATA_PATH = BASE_DIR / "data" / "processed" / "subjects_processed.csv"

df = pd.read_csv(DATA_PATH)

