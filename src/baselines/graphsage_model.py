from pathlib import Path
import warnings
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import networkx as nx

from sklearn.metrics import (
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# 选择运行设备，优先使用 GPU，没有 GPU 则使用 CPU
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"Using device: {device}")
    return device


# 设置随机种子，尽量保证实验结果可复现
def set_seed(random_state=42):
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)


# 计算 FPR、FNR 以及混淆矩阵中的 TN、FP、FN、TP
def calculate_fpr_fnr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return fpr, fnr, tn, fp, fn, tp


# 读取已经完成划分的训练集和测试集
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


# 准备节点特征和标签
# 每一条交易记录被视为一个节点，isFraud 作为节点标签
def prepare_node_features(train_df, test_df, label_col="isFraud"):
    print("=" * 70)
    print("Step 2: Preparing node features and labels...")

    prep_start = time.time()

    drop_cols = [label_col]

    optional_drop_cols = ["TransactionID"]
    for col in optional_drop_cols:
        if col in train_df.columns:
            drop_cols.append(col)

    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    y_train = train_df[label_col].astype(int)

    X_test = test_df.drop(columns=drop_cols, errors="ignore")
    y_test = test_df[label_col].astype(int)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.long)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.long)

    prep_end = time.time()

    print(f"X_train shape: {tuple(X_train.shape)}")
    print(f"X_test shape : {tuple(X_test.shape)}")
    print("y_train distribution:")
    print(pd.Series(y_train.numpy()).value_counts())
    print("y_test distribution:")
    print(pd.Series(y_test.numpy()).value_counts())
    print(f"Feature preparation time: {prep_end - prep_start:.2f} seconds")

    return X_train, y_train, X_test, y_test


# 自动选择可用于构图的交易实体字段
# 如果字段存在，就用共享实体关系构建交易节点之间的边
def select_graph_columns(df):
    candidate_cols = [
        "card1", "card2", "card3", "card4", "card5", "card6",
        "addr1", "addr2",
        "P_emaildomain", "R_emaildomain",
        "DeviceType", "DeviceInfo",
        "ProductCD"
    ]

    graph_cols = [col for col in candidate_cols if col in df.columns]

    print("=" * 70)
    print("Step 3: Selecting graph construction columns...")

    if len(graph_cols) == 0:
        print("No entity columns found. Fallback to TransactionDT-based chain graph.")
    else:
        print(f"Graph construction columns: {graph_cols}")

    return graph_cols


# 根据共享实体字段构建交易图
# 为了避免完整图过大，同一个实体组内只连接相邻交易节点
def build_edge_index(df, graph_cols, max_edges_per_group=200):
    print("=" * 70)
    print("Step 4: Building transaction graph...")

    graph_start = time.time()

    num_nodes = len(df)
    edge_set = set()

    if len(graph_cols) == 0:
        if "TransactionDT" in df.columns:
            sorted_indices = df.sort_values("TransactionDT").index.tolist()
        else:
            sorted_indices = list(range(num_nodes))

        for i in range(len(sorted_indices) - 1):
            src = sorted_indices[i]
            dst = sorted_indices[i + 1]
            edge_set.add((src, dst))
            edge_set.add((dst, src))
    else:
        for col in graph_cols:
            grouped = df.groupby(col).indices

            for _, indices in grouped.items():
                indices = list(indices)

                if len(indices) <= 1:
                    continue

                if len(indices) > max_edges_per_group:
                    indices = indices[:max_edges_per_group]

                for i in range(len(indices) - 1):
                    src = indices[i]
                    dst = indices[i + 1]
                    edge_set.add((src, dst))
                    edge_set.add((dst, src))

    for i in range(num_nodes):
        edge_set.add((i, i))

    edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()

    graph_end = time.time()

    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {edge_index.shape[1]}")
    print(f"Graph construction time: {graph_end - graph_start:.2f} seconds")

    return edge_index


# 构建 GraphSAGE 使用的均值聚合邻接矩阵
# 这里实现的是按目标节点归一化的 mean aggregation
def build_mean_adj(edge_index, num_nodes, device):
    src, dst = edge_index

    values = torch.ones(src.size(0), dtype=torch.float32)

    degree = torch.zeros(num_nodes, dtype=torch.float32)
    degree.scatter_add_(0, dst, values)

    norm_values = values / (degree[dst] + 1e-12)

    adj_index = torch.stack([dst, src], dim=0)

    adj = torch.sparse_coo_tensor(
        adj_index,
        norm_values,
        size=(num_nodes, num_nodes)
    )

    return adj.coalesce().to(device)


# 定义 GraphSAGE 单层
# 核心思想：分别变换自身特征和邻居均值特征，然后进行融合
class GraphSAGELayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphSAGELayer, self).__init__()

        self.self_linear = nn.Linear(in_features, out_features)
        self.neighbor_linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        neighbor_mean = torch.sparse.mm(adj, x)

        self_part = self.self_linear(x)
        neighbor_part = self.neighbor_linear(neighbor_mean)

        out = self_part + neighbor_part

        return out


# 定义两层 GraphSAGE 模型
# 第一层学习节点隐藏表示，第二层输出 fraud / legitimate 二分类 logits
class GraphSAGEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2, dropout=0.5):
        super(GraphSAGEModel, self).__init__()

        self.sage1 = GraphSAGELayer(input_dim, hidden_dim)
        self.sage2 = GraphSAGELayer(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.sage1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.sage2(x, adj)

        return x


# 训练 GraphSAGE 模型
# 使用 class weight 缓解 fraud 样本较少的问题
def train_graphsage_model(
    X_train,
    y_train,
    train_adj,
    input_dim,
    device,
    epochs=50,
    hidden_dim=64,
    learning_rate=0.001,
    weight_decay=5e-4
):
    print("=" * 70)
    print("Step 5: Training GraphSAGE model...")

    train_start = time.time()
    loss_history = []

    model = GraphSAGEModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=2,
        dropout=0.5
    ).to(device)

    X_train = X_train.to(device)
    y_train = y_train.to(device)

    class_counts = torch.bincount(y_train)
    class_weights = class_counts.sum() / (2.0 * class_counts.float())
    class_weights = class_weights.to(device)

    print("Class weight information:")
    print(f"Class counts: {class_counts.detach().cpu().numpy()}")
    print(f"Class weights: {class_weights.detach().cpu().numpy()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    for epoch in range(1, epochs + 1):
        model.train()

        optimizer.zero_grad()
        logits = model(X_train, train_adj)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}")

    train_end = time.time()

    print("GraphSAGE training completed.")
    print(f"Model training time: {train_end - train_start:.2f} seconds")

    return model, loss_history


# 使用测试图评估 GraphSAGE 模型
# 输出 Recall、F1、AUC、FNR、FPR
def evaluate_graphsage_model(model, X_test, y_test, test_adj, device):
    print("=" * 70)
    print("Step 6: Evaluating GraphSAGE model...")

    eval_start = time.time()

    model.eval()

    X_test = X_test.to(device)
    y_test_device = y_test.to(device)

    with torch.no_grad():
        logits = model(X_test, test_adj)
        prob = F.softmax(logits, dim=1)[:, 1]
        pred = torch.argmax(logits, dim=1)

    y_true = y_test_device.cpu().numpy()
    y_pred = pred.cpu().numpy()
    y_prob = prob.cpu().numpy()

    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    fpr, fnr, tn, fp, fn, tp = calculate_fpr_fnr(y_true, y_pred)

    metrics_dict = {
        "Model": "GraphSAGE",
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


# 绘制 GraphSAGE 训练损失曲线
# 该图用于观察模型训练过程中 loss 是否下降
def plot_loss_curve(loss_history, figure_dir):
    figure_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("GraphSAGE Training Loss Curve")
    plt.grid(True)

    figure_path = figure_dir / "graphsage_loss_curve.png"
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Loss curve saved to : {figure_path}")


# 绘制交易图抽样子图
# 完整交易图过大，因此这里只抽取部分节点和边作为结构展示
def plot_transaction_subgraph(edge_index, figure_dir, max_nodes=80, max_edges=200):
    figure_dir.mkdir(parents=True, exist_ok=True)

    edge_array = edge_index.cpu().numpy().T

    sampled_edges = []
    sampled_nodes = set()

    for src, dst in edge_array:
        if src == dst:
            continue

        sampled_edges.append((int(src), int(dst)))
        sampled_nodes.add(int(src))
        sampled_nodes.add(int(dst))

        if len(sampled_nodes) >= max_nodes or len(sampled_edges) >= max_edges:
            break

    graph = nx.Graph()
    graph.add_edges_from(sampled_edges)

    if graph.number_of_nodes() == 0:
        print("No valid edges found for subgraph visualization.")
        return

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph, seed=42)

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=80,
        alpha=0.8
    )

    nx.draw_networkx_edges(
        graph,
        pos,
        width=0.6,
        alpha=0.5
    )

    plt.title("Sampled Transaction Subgraph for GraphSAGE")
    plt.axis("off")

    figure_path = figure_dir / "graphsage_transaction_subgraph.png"
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Transaction subgraph saved to : {figure_path}")


# 保存 GraphSAGE 实验结果
# 包括整体指标表和前 1000 条预测样例
def save_results(output_dir, metrics_df, test_df, y_pred, y_prob):
    print("=" * 70)
    print("Step 7: Saving results...")

    save_start = time.time()

    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "graphsage_metrics.csv"
    pred_path = output_dir / "graphsage_predictions_sample.csv"

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
# 顺序为：读取数据 -> 节点特征准备 -> 图构建 -> GraphSAGE训练 -> 评估 -> 画图 -> 保存结果
def run_graphsage_baseline(
    processed_dir,
    output_dir,
    train_file="train_split.csv",
    test_file="test_split.csv",
    label_col="isFraud",
    random_state=42
):
    total_start = time.time()

    set_seed(random_state)
    device = get_device()

    train_df, test_df = load_data(
        processed_dir=processed_dir,
        train_file=train_file,
        test_file=test_file,
        label_col=label_col
    )

    X_train, y_train, X_test, y_test = prepare_node_features(
        train_df=train_df,
        test_df=test_df,
        label_col=label_col
    )

    graph_cols = select_graph_columns(train_df)

    train_edge_index = build_edge_index(
        df=train_df,
        graph_cols=graph_cols,
        max_edges_per_group=200
    )

    test_edge_index = build_edge_index(
        df=test_df,
        graph_cols=graph_cols,
        max_edges_per_group=200
    )

    train_adj = build_mean_adj(
        edge_index=train_edge_index,
        num_nodes=X_train.shape[0],
        device=device
    )

    test_adj = build_mean_adj(
        edge_index=test_edge_index,
        num_nodes=X_test.shape[0],
        device=device
    )

    model, loss_history = train_graphsage_model(
        X_train=X_train,
        y_train=y_train,
        train_adj=train_adj,
        input_dim=X_train.shape[1],
        device=device,
        epochs=50,
        hidden_dim=64,
        learning_rate=0.001,
        weight_decay=5e-4
    )

    metrics_df, y_pred, y_prob = evaluate_graphsage_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        test_adj=test_adj,
        device=device
    )

    figure_dir = output_dir.parent / "figures"

    plot_loss_curve(
        loss_history=loss_history,
        figure_dir=figure_dir
    )

    plot_transaction_subgraph(
        edge_index=train_edge_index,
        figure_dir=figure_dir,
        max_nodes=80,
        max_edges=200
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
    print("GraphSAGE baseline completed successfully.")
    print(f"Total running time: {total_end - total_start:.2f} seconds")

    return model, metrics_df


# 程序入口
# 默认读取 data/processed 下的划分数据，并将结果保存到 output/tables 和 output/figures
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    processed_dir = project_root / "data" / "processed"
    output_dir = project_root / "output" / "tables"

    model, metrics_df = run_graphsage_baseline(
        processed_dir=processed_dir,
        output_dir=output_dir,
        train_file="train_split.csv",
        test_file="test_split.csv",
        label_col="isFraud",
        random_state=42
    )