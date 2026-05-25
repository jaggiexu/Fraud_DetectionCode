from pathlib import Path
import pandas as pd


def transform_features(
    processed_dir: Path,
    input_file: str = "train_cleaned.csv",
    output_file: str = "train_transformed.csv",
) -> pd.DataFrame:
    """
    Transform cleaned dataset into a structured feature set for baseline experiments.

    Strategy:
    1. Load cleaned dataset
    2. Identify numeric and categorical columns
    3. Encode categorical columns into numeric codes
    4. Save transformed dataset

    Args:
        processed_dir: Directory containing processed data files
        input_file: Cleaned input dataset
        output_file: Transformed output dataset

    Returns:
        pd.DataFrame: Transformed dataframe
    """

    input_path = processed_dir / input_file
    output_path = processed_dir / output_file

    if not input_path.exists():
        raise FileNotFoundError(f"Input cleaned file not found: {input_path}")

    print("=" * 70)
    print("Step 1: Loading cleaned dataset...")
    df = pd.read_csv(input_path)
    print(f"Input shape: {df.shape}")

    print("=" * 70)
    print("Step 2: Identifying column types...")
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    print(f"Number of numeric columns    : {len(numeric_cols)}")
    print(f"Number of categorical columns: {len(categorical_cols)}")

    print("=" * 70)
    print("Step 3: Encoding categorical features...")
    for col in categorical_cols:
        df[col] = df[col].astype("category").cat.codes

    print("Categorical encoding completed.")

    print("=" * 70)
    print("Step 4: Final validation...")
    print(f"Output shape: {df.shape}")
    print(f"Remaining missing values: {df.isnull().sum().sum()}")

    print("=" * 70)
    print("Step 5: Saving transformed dataset...")
    df.to_csv(output_path, index=False)
    print(f"Transformed dataset saved to: {output_path}")

    return df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    processed_dir = project_root / "data" / "processed"

    transformed_df = transform_features(
        processed_dir=processed_dir,
        input_file="train_cleaned.csv",
        output_file="train_transformed.csv",
    )

    print("=" * 70)
    print("Feature transformation completed successfully.")
    print(transformed_df.head())