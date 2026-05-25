from pathlib import Path
import pandas as pd


def apply_temporal_ordering(
    processed_dir: Path,
    input_file: str = "train_transformed.csv",
    output_file: str = "train_ordered.csv",
    time_column: str = "TransactionDT",
) -> pd.DataFrame:
    """
    Apply temporal ordering to the transformed dataset by sorting records
    in ascending order of TransactionDT.

    Args:
        processed_dir: Directory containing processed data files.
        input_file: Input transformed dataset file name.
        output_file: Output ordered dataset file name.
        time_column: Time-related column used for ordering.

    Returns:
        pd.DataFrame: Temporally ordered dataframe.
    """

    input_path = processed_dir / input_file
    output_path = processed_dir / output_file

    if not input_path.exists():
        raise FileNotFoundError(f"Input transformed file not found: {input_path}")

    print("=" * 70)
    print("Step 1: Loading transformed dataset...")
    df = pd.read_csv(input_path)

    original_rows, original_cols = df.shape
    print(f"Original shape: {df.shape}")

    if time_column not in df.columns:
        raise KeyError(f"Time column '{time_column}' not found in dataset.")

    print("=" * 70)
    print(f"Step 2: Sorting records by '{time_column}' in ascending order...")
    df = df.sort_values(by=time_column, ascending=True).reset_index(drop=True)

    print("Temporal ordering completed.")

    print("=" * 70)
    print("Step 3: Final validation...")
    final_rows, final_cols = df.shape
    print(f"Final shape: {df.shape}")
    print(f"Original rows   : {original_rows}")
    print(f"Original columns: {original_cols}")
    print(f"Final rows      : {final_rows}")
    print(f"Final columns   : {final_cols}")

    print("=" * 70)
    print("Step 4: Saving temporally ordered dataset...")
    df.to_csv(output_path, index=False)
    print(f"Ordered dataset saved to: {output_path}")

    return df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    processed_dir = project_root / "data" / "processed"

    ordered_df = apply_temporal_ordering(
        processed_dir=processed_dir,
        input_file="train_transformed.csv",
        output_file="train_ordered.csv",
        time_column="TransactionDT",
    )

    print("=" * 70)
    print("Temporal ordering completed successfully.")
    print(ordered_df[[ "TransactionID", "TransactionDT" ]].head())