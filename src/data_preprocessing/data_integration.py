from pathlib import Path
import pandas as pd


#输出整合后的训练集train_merge到data/processed

def integrate_train_data(
    raw_dir: Path,
    processed_dir: Path,
    transaction_file: str = "train_transaction_590540.csv",
    identity_file: str = "train_identity.csv",
    output_file: str = "train_merged.csv",
) -> pd.DataFrame:
    """
    Integrate train_transaction and train_identity using TransactionID.

    Args:
        raw_dir: Directory containing raw CSV files.
        processed_dir: Directory to save processed merged data.
        transaction_file: Main transaction table filename.
        identity_file: Supplementary identity table filename.
        output_file: Output merged filename.

    Returns:
        pd.DataFrame: Merged training dataframe.
    """

    transaction_path = raw_dir / transaction_file
    identity_path = raw_dir / identity_file
    output_path = processed_dir / output_file

    if not transaction_path.exists():
        raise FileNotFoundError(f"Transaction file not found: {transaction_path}")

    if not identity_path.exists():
        raise FileNotFoundError(f"Identity file not found: {identity_path}")

    processed_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Step 1: Loading raw training tables...")
    train_transaction = pd.read_csv(transaction_path)
    train_identity = pd.read_csv(identity_path)

    print(f"train_transaction shape: {train_transaction.shape}")
    print(f"train_identity shape   : {train_identity.shape}")

    if "TransactionID" not in train_transaction.columns:
        raise KeyError("TransactionID not found in train_transaction.")

    if "TransactionID" not in train_identity.columns:
        raise KeyError("TransactionID not found in train_identity.")

    print("=" * 70)
    print("Step 2: Checking TransactionID duplicates...")
    train_txn_dup = train_transaction["TransactionID"].duplicated().sum()
    train_id_dup = train_identity["TransactionID"].duplicated().sum()

    print(f"Duplicated TransactionID in train_transaction: {train_txn_dup}")
    print(f"Duplicated TransactionID in train_identity   : {train_id_dup}")

    print("=" * 70)
    print("Step 3: Merging tables with left join on TransactionID...")
    train_merged = train_transaction.merge(
        train_identity,
        on="TransactionID",
        how="left"
    )

    print(f"Merged train shape: {train_merged.shape}")

    print("=" * 70)
    print("Step 4: Basic merge validation...")
    print(f"Rows before merge : {len(train_transaction)}")
    print(f"Rows after merge  : {len(train_merged)}")

    matched_identity = train_merged["DeviceType"].notna().sum() if "DeviceType" in train_merged.columns else "N/A"
    print(f"Rows with matched identity info: {matched_identity}")

    print("=" * 70)
    print("Step 5: Saving merged dataset...")
    train_merged.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")

    return train_merged


if __name__ == "__main__":
    # Project root = .../Fraud_Detection Code
    project_root = Path(__file__).resolve().parents[2]

    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"

    merged_df = integrate_train_data(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        transaction_file="train_transaction_590540.csv",
        identity_file="train_identity.csv",
        output_file="train_merged.csv",
    )

    print("=" * 70)
    print("Integration completed successfully.")
    print(merged_df.head())