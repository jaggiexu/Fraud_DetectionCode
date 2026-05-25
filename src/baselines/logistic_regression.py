from pathlib import Path
import warnings
import time

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# 计算分类结果中的 FPR、FNR 以及混淆矩阵四个基本值
def calculate_fpr_fnr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return fpr, fnr, tn, fp, fn, tp


# 读取已经完成划分的训练集和测试集
# 核心输入文件默认为 train_split.csv 和 test_split.csv
def load_data(processed_dir, train_file, test_file, label_col="isFraud"):
    train_path = processed_dir / train_file
    test_path = processed_dir / test_file

    print("=" * 70)
    print("Step 1: Loading split datasets...")
    print(f"Training file: {train_path}")
    print(f"Testing file : {test_path}")

    load_start = time.time()

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    load_end = time.time()

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape : {test_df.shape}")
    print(f"Data loading time: {load_end - load_start:.2f} seconds")

    if label_col not in train_df.columns or label_col not in test_df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset.")

    return train_df, test_df


# 分离特征和标签
# 同时去掉不适合直接用于训练的标识列，例如 TransactionID
def prepare_features(train_df, test_df, label_col="isFraud"):
    print("=" * 70)
    print("Step 2: Preparing features and labels...")

    prep_start = time.time()

    drop_cols = [label_col]

    optional_drop_cols = ["TransactionID"]
    for col in optional_drop_cols:
        if col in train_df.columns:
            drop_cols.append(col)

    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    y_train = train_df[label_col]

    X_test = test_df.drop(columns=drop_cols, errors="ignore")
    y_test = test_df[label_col]

    prep_end = time.time()

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape : {X_test.shape}")
    print("y_train distribution:")
    print(y_train.value_counts())
    print("y_test distribution:")
    print(y_test.value_counts())
    print(f"Feature preparation time: {prep_end - prep_start:.2f} seconds")

    return X_train, y_train, X_test, y_test


# 训练 Logistic Regression 模型
# 关键设置：
# 1. StandardScaler：对特征进行标准化，适合 LR
# 2. class_weight='balanced'：缓解欺诈检测中的类别不平衡问题
# 3. max_iter=1000：避免迭代次数不足导致不收敛
def train_logistic_regression(X_train, y_train, random_state=42):
    print("=" * 70)
    print("Step 3: Training Logistic Regression model...")

    train_start = time.time()

    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="liblinear",
            random_state=random_state
        ))
    ])

    lr_pipeline.fit(X_train, y_train)

    train_end = time.time()

    print("Logistic Regression training completed.")
    print(f"Model training time: {train_end - train_start:.2f} seconds")

    return lr_pipeline


# 使用测试集对模型进行评估
# 输出结果包括：Recall、F1、AUC、FNR、FPR
def evaluate_model(model, X_test, y_test):
    print("=" * 70)
    print("Step 4: Evaluating model...")

    eval_start = time.time()

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    fpr, fnr, tn, fp, fn, tp = calculate_fpr_fnr(y_test, y_pred)

    metrics_dict = {
        "Model": "Logistic Regression",
        "Recall": recall,
        "F1": f1,
        "AUC": auc,
        "FNR": fnr,
        "FPR": fpr,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }

    metrics_df = pd.DataFrame([metrics_dict])

    eval_end = time.time()

    print("Evaluation Results:")
    print(metrics_df)
    print(f"Evaluation time: {eval_end - eval_start:.2f} seconds")

    return metrics_df, y_pred, y_prob


# 保存输出结果
# 包括：
# 1. 指标表 lr_metrics.csv
# 2. 预测样例 lr_predictions_sample.csv
def save_results(output_dir, metrics_df, test_df, y_pred, y_prob):
    print("=" * 70)
    print("Step 5: Saving results...")

    save_start = time.time()

    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "lr_metrics.csv"
    pred_path = output_dir / "lr_predictions_sample.csv"

    metrics_df.to_csv(metrics_path, index=False)

    prediction_df = test_df.copy()
    prediction_df["pred_label"] = y_pred
    prediction_df["pred_prob"] = y_prob

    prediction_df.head(1000).to_csv(pred_path, index=False)

    save_end = time.time()

    print(f"Metrics saved to     : {metrics_path}")
    print(f"Predictions saved to : {pred_path}")
    print(f"Result saving time: {save_end - save_start:.2f} seconds")


# 整体运行主流程
# 按顺序完成：读取数据 -> 准备特征 -> 训练模型 -> 模型评估 -> 保存结果
def run_logistic_regression_baseline(
    processed_dir,
    output_dir,
    train_file="train_split.csv",
    test_file="test_split.csv",
    label_col="isFraud",
    random_state=42
):
    total_start = time.time()

    train_df, test_df = load_data(
        processed_dir=processed_dir,
        train_file=train_file,
        test_file=test_file,
        label_col=label_col
    )

    X_train, y_train, X_test, y_test = prepare_features(
        train_df=train_df,
        test_df=test_df,
        label_col=label_col
    )

    model = train_logistic_regression(
        X_train=X_train,
        y_train=y_train,
        random_state=random_state
    )

    metrics_df, y_pred, y_prob = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test
    )

    save_results(
        output_dir=output_dir,
        metrics_df=metrics_df,
        test_df=test_df,
        y_pred=y_pred,
        y_prob=y_prob
    )

    total_end = time.time()

    print("=" * 70)
    print("Logistic Regression baseline completed successfully.")
    print(f"Total running time: {total_end - total_start:.2f} seconds")

    return model, metrics_df


# 程序入口
# 默认从项目根目录下读取 data/processed 中的数据，并将结果输出到 output/tables
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    processed_dir = project_root / "data" / "processed"
    output_dir = project_root / "output" / "tables"

    model, metrics_df = run_logistic_regression_baseline(
        processed_dir=processed_dir,
        output_dir=output_dir,
        train_file="train_split.csv",
        test_file="test_split.csv",
        label_col="isFraud",
        random_state=42
    )