from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(
    processed_dir: Path,
    input_file: str = "train_ordered.csv",
    train_output_file: str = "train_split.csv",
    test_output_file: str = "test_split.csv",
    summary_output_file: str = "dataset_split_summary.csv",
    label_col: str = "isFraud",
    test_size: float = 0.20,
    random_state: int = 42,
):
    """
    Split the processed training dataset into stratified train/test subsets.

    Args:
        processed_dir: Path to processed data directory.
        input_file: Input ordered dataset file.
        train_output_file: Output training subset file.
        test_output_file: Output testing subset file.
        summary_output_file: Output split summary file.
        label_col: Target label column.
        test_size: Ratio of testing subset.
        random_state: Random seed.

    Returns:
        train_df, test_df, summary_df
    """

    input_path = processed_dir / input_file
    train_output_path = processed_dir / train_output_file
    test_output_path = processed_dir / test_output_file
    summary_output_path = processed_dir / summary_output_file

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print("=" * 70)
    print("Step 1: Loading ordered dataset...")
    df = pd.read_csv(input_path)
    print(f"Input shape: {df.shape}")

    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in dataset.")

    print("=" * 70)
    print("Step 2: Applying stratified 80/20 split...")
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_col],
        random_state=random_state
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"Train subset shape: {train_df.shape}")
    print(f"Test subset shape : {test_df.shape}")

    print("=" * 70)
    print("Step 3: Counting class distribution...")

    total_legit = int((df[label_col] == 0).sum())
    total_fraud = int((df[label_col] == 1).sum())

    train_legit = int((train_df[label_col] == 0).sum())
    train_fraud = int((train_df[label_col] == 1).sum())

    test_legit = int((test_df[label_col] == 0).sum())
    test_fraud = int((test_df[label_col] == 1).sum())

    summary_df = pd.DataFrame({
        "Subset": ["Full Dataset", "Training Set", "Testing Set"],
        "Total Rows": [len(df), len(train_df), len(test_df)],
        "Legitimate Transactions": [total_legit, train_legit, test_legit],
        "Fraudulent Transactions": [total_fraud, train_fraud, test_fraud]
    })

    print(summary_df)

    print("=" * 70)
    print("Step 4: Saving split datasets...")
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)
    summary_df.to_csv(summary_output_path, index=False)

    print(f"Training set saved to: {train_output_path}")
    print(f"Testing set saved to : {test_output_path}")
    print(f"Summary saved to     : {summary_output_path}")

    return train_df, test_df, summary_df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    processed_dir = project_root / "data" / "processed"

    train_df, test_df, summary_df = split_dataset(
        processed_dir=processed_dir,
        input_file="train_ordered.csv",
        train_output_file="train_split.csv",
        test_output_file="test_split.csv",
        summary_output_file="dataset_split_summary.csv",
        label_col="isFraud",
        test_size=0.20,
        random_state=42
    )

    print("=" * 70)
    print("Dataset split completed successfully.")