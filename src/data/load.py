import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Args:
        path: Path to the CSV file
        
    Returns:j
        pd.DataFrame: Loaded data
    """
    print(f"Attempting to load data from {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("DataFrame is empty")
    print(f"Successfully loaded data with shape {df.shape}")
    return df