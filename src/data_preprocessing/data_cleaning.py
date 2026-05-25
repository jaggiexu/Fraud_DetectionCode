from pathlib import Path
import pandas as pd


def clean_merged_data(
    processed_dir: Path,
    input_file: str = "train_merged.csv",
    output_file: str = "train_cleaned.csv",
    missing_report_file: str = "missing_value_report.csv",
    missing_threshold: float = 0.90,
) -> pd.DataFrame:
    """
    Clean the merged IEEE-CIS training dataset.

    Cleaning strategy:
    1. Drop columns with missing ratio > 90%
    2. Fill missing values in remaining columns
       - numeric columns: median
       - categorical/object columns: 'Unknown'
    3. Remove duplicated rows
    4. Save cleaned dataset and missing-value report
    """

    input_path = processed_dir / input_file
    output_path = processed_dir / output_file
    report_path = processed_dir / missing_report_file

    if not input_path.exists():
        raise FileNotFoundError(f"Input merged file not found: {input_path}")

    print("=" * 70)
    print("Step 1: Loading merged dataset...")
    df = pd.read_csv(input_path)

    original_rows, original_cols = df.shape
    print(f"Original shape: {df.shape}")

    print("=" * 70)
    print("Step 2: Calculating missing value ratio...")
    missing_ratio = df.isnull().mean().sort_values(ascending=False)

    missing_report = pd.DataFrame({
        "column": missing_ratio.index,
        "missing_ratio": missing_ratio.values,
        "missing_count": df.isnull().sum()[missing_ratio.index].values
    })
    missing_report.to_csv(report_path, index=False)
    print(f"Missing value report saved to: {report_path}")

    print("=" * 70)
    print(f"Step 3: Dropping columns with missing ratio > {missing_threshold:.0%} ...")
    cols_to_drop = missing_ratio[missing_ratio > missing_threshold].index.tolist()

    protected_cols = ["TransactionID", "isFraud"]
    cols_to_drop = [col for col in cols_to_drop if col not in protected_cols]

    print(f"Number of columns to drop: {len(cols_to_drop)}")
    df = df.drop(columns=cols_to_drop)

    print(f"Shape after dropping high-missing columns: {df.shape}")

    print("=" * 70)
    print("Step 4: Filling missing values in remaining columns...")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    for col in numeric_cols:
        if df[col].isnull().any():
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)

    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna("Unknown")

    remaining_missing = df.isnull().sum().sum()
    print(f"Total remaining missing values after filling: {remaining_missing}")

    print("=" * 70)
    print("Step 5: Removing duplicated rows...")
    before_dedup_rows = len(df)
    df = df.drop_duplicates()
    after_dedup_rows = len(df)

    print(f"Duplicated rows removed: {before_dedup_rows - after_dedup_rows}")
    print(f"Shape after removing duplicates: {df.shape}")

    print("=" * 70)
    print("Step 6: Final validation...")
    if "isFraud" not in df.columns:
        raise KeyError("Target column 'isFraud' is missing after cleaning.")

    final_rows, final_cols = df.shape

    print(f"Original rows   : {original_rows}")
    print(f"Original columns: {original_cols}")
    print(f"Final rows      : {final_rows}")
    print(f"Final columns   : {final_cols}")
    print(f"Fraud label distribution:\n{df['isFraud'].value_counts(dropna=False)}")

    print("=" * 70)
    print("Step 7: Saving cleaned dataset...")
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to: {output_path}")

    return df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    processed_dir = project_root / "data" / "processed"

    cleaned_df = clean_merged_data(
        processed_dir=processed_dir,
        input_file="train_merged.csv",
        output_file="train_cleaned.csv",
        missing_report_file="missing_value_report.csv",
        missing_threshold=0.90
    )

    print("=" * 70)
    print("Data cleaning completed successfully.")
    print(cleaned_df.head())