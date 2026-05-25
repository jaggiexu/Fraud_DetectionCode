from pathlib import Path
import warnings
import time

import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import (
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# 计算 FPR、FNR 以及混淆矩阵四个基本值
def calculate_fpr_fnr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return fpr, fnr, tn, fp, fn, tp


# 读取已经划分好的训练集和测试集
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


# 分离特征和标签，并删除不适合直接训练的标识列
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


# 根据训练集类别分布计算 scale_pos_weight
# 该参数用于增强模型对少数类 fraud 样本的关注
def calculate_scale_pos_weight(y_train):
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    if positive_count == 0:
        raise ValueError("No positive fraud samples found in y_train.")

    scale_pos_weight = negative_count / positive_count

    print("=" * 70)
    print("Class imbalance information:")
    print(f"Negative samples: {negative_count}")
    print(f"Positive samples: {positive_count}")
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")

    return scale_pos_weight


# 训练 XGBoost 模型
# 关键说明：
# 1. XGBoost 适合处理结构化表格数据
# 2. scale_pos_weight 用于缓解类别不平衡
# 3. tree_method='hist' 用于加速 CPU 训练
# 4. n_jobs=-1 使用全部 CPU 核心
def train_xgboost(X_train, y_train, random_state=42):
    print("=" * 70)
    print("Step 3: Training XGBoost model...")

    train_start = time.time()

    scale_pos_weight = calculate_scale_pos_weight(y_train)

    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1
    )

    xgb_model.fit(X_train, y_train)

    train_end = time.time()

    print("XGBoost training completed.")
    print(f"Model training time: {train_end - train_start:.2f} seconds")

    return xgb_model


# 使用测试集评估模型性能
# 输出 Recall、F1、AUC、FNR、FPR 以及混淆矩阵结果
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
        "Model": "XGBoost",
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


# 保存实验结果
# 包括：
# 1. 模型整体指标表
# 2. 测试集前 1000 条预测样例
def save_results(output_dir, metrics_df, test_df, y_pred, y_prob):
    print("=" * 70)
    print("Step 5: Saving results...")

    save_start = time.time()

    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "xgboost_metrics.csv"
    pred_path = output_dir / "xgboost_predictions_sample.csv"

    metrics_df.to_csv(metrics_path, index=False)

    prediction_df = test_df.copy()
    prediction_df["pred_label"] = y_pred
    prediction_df["pred_prob"] = y_prob

    prediction_df.head(1000).to_csv(pred_path, index=False)

    save_end = time.time()

    print(f"Metrics saved to     : {metrics_path}")
    print(f"Predictions saved to : {pred_path}")
    print(f"Result saving time: {save_end - save_start:.2f} seconds")


# 整体运行流程
# 顺序为：读取数据 -> 特征准备 -> 类别权重计算 -> 模型训练 -> 模型评估 -> 保存结果
def run_xgboost_baseline(
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

    model = train_xgboost(
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
    print("XGBoost baseline completed successfully.")
    print(f"Total running time: {total_end - total_start:.2f} seconds")

    return model, metrics_df


# 程序入口
# 默认从 data/processed 中读取划分后的数据，并将结果保存到 output/tables
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    processed_dir = project_root / "data" / "processed"
    output_dir = project_root / "output" / "tables"

    model, metrics_df = run_xgboost_baseline(
        processed_dir=processed_dir,
        output_dir=output_dir,
        train_file="train_split.csv",
        test_file="test_split.csv",
        label_col="isFraud",
        random_state=42
    )