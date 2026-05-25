from pathlib import Path
import warnings
import time

import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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


# 训练线性 SVM 模型
# 关键说明：
# 1. SVM 对特征尺度敏感，因此需要标准化
# 2. 使用 LinearSVC 以适应大规模数据
# 3. 使用 CalibratedClassifierCV 生成概率输出，便于计算 AUC
# 4. class_weight='balanced' 用于缓解类别不平衡
def train_svm(X_train, y_train, random_state=42):
    print("=" * 70)
    print("Step 3: Training SVM model...")

    train_start = time.time()

    base_svm = LinearSVC(
        class_weight="balanced",
        max_iter=3000,
        random_state=random_state
    )

    svm_model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", CalibratedClassifierCV(
            estimator=base_svm,
            method="sigmoid",
            cv=3
        ))
    ])

    svm_model.fit(X_train, y_train)

    train_end = time.time()

    print("SVM training completed.")
    print(f"Model training time: {train_end - train_start:.2f} seconds")

    return svm_model


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
        "Model": "SVM",
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

    metrics_path = output_dir / "svm_metrics.csv"
    pred_path = output_dir / "svm_predictions_sample.csv"

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
# 顺序为：读取数据 -> 特征准备 -> 模型训练 -> 模型评估 -> 保存结果
def run_svm_baseline(
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

    model = train_svm(
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
    print("SVM baseline completed successfully.")
    print(f"Total running time: {total_end - total_start:.2f} seconds")

    return model, metrics_df


# 程序入口
# 默认从 data/processed 中读取划分后的数据，并将结果保存到 output/tables
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    processed_dir = project_root / "data" / "processed"
    output_dir = project_root / "output" / "tables"

    model, metrics_df = run_svm_baseline(
        processed_dir=processed_dir,
        output_dir=output_dir,
        train_file="train_split.csv",
        test_file="test_split.csv",
        label_col="isFraud",
        random_state=42
    )