"""
Data utility functions 
"""

from pathlib import Path
import pandas as pd


def load_csv(path):
    """Load CSV file."""
    return pd.read_csv(path)


def save_csv(df, path):
    """Save DataFrame to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def remove_columns(df, columns):
    """Remove unwanted columns if present."""
    cols = [c for c in columns if c in df.columns]
    return df.drop(columns=cols)


def print_shape(df, name):
    print(f"{name} Shape: {df.shape}")