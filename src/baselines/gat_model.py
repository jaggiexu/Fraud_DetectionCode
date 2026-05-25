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


# 定义 GAT 单层
# 核心思想：对每条边计算 attention 权重，再按目标节点聚合邻居信息
class GATLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        heads=4,
        concat=True,
        dropout=0.5,
        negative_slope=0.2
    ):
        super(GATLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        self.linear = nn.Linear(in_features, out_features * heads, bias=False)

        self.att_src = nn.Parameter(torch.Tensor(heads, out_features))
        self.att_dst = nn.Parameter(torch.Tensor(heads, out_features))

        if concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_features))
        else:
            self.bias = nn.Parameter(torch.Tensor(out_features))

        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.reset_parameters()

    # 初始化 GAT 层中的可学习参数
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        src, dst = edge_index

        h = self.linear(x)
        h = h.view(num_nodes, self.heads, self.out_features)

        h_src = h[src]
        h_dst = h[dst]

        attention_score = (h_src * self.att_src).sum(dim=-1) + (h_dst * self.att_dst).sum(dim=-1)
        attention_score = self.leaky_relu(attention_score)

        attention_score = torch.clamp(attention_score, min=-10, max=10)
        attention_exp = torch.exp(attention_score)

        attention_sum = torch.zeros(
            num_nodes,
            self.heads,
            device=x.device,
            dtype=torch.float32
        )

        attention_sum.index_add_(0, dst, attention_exp)
        attention_alpha = attention_exp / (attention_sum[dst] + 1e-16)
        attention_alpha = F.dropout(attention_alpha, p=self.dropout, training=self.training)

        message = h_src * attention_alpha.unsqueeze(-1)

        out = torch.zeros(
            num_nodes,
            self.heads,
            self.out_features,
            device=x.device,
            dtype=torch.float32
        )

        out.index_add_(0, dst, message)

        if self.concat:
            out = out.reshape(num_nodes, self.heads * self.out_features)
        else:
            out = out.mean(dim=1)

        out = out + self.bias

        return out


# 定义两层 GAT 模型
# 第一层使用多头 attention 学习邻居权重，第二层输出二分类 logits
class GATModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=32,
        output_dim=2,
        heads=4,
        dropout=0.5
    ):
        super(GATModel, self).__init__()

        self.dropout = dropout

        self.gat1 = GATLayer(
            in_features=input_dim,
            out_features=hidden_dim,
            heads=heads,
            concat=True,
            dropout=dropout
        )

        self.gat2 = GATLayer(
            in_features=hidden_dim * heads,
            out_features=output_dim,
            heads=1,
            concat=False,
            dropout=dropout
        )

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)

        return x


# 训练 GAT 模型
# 使用 class weight 缓解 fraud 样本较少的问题
def train_gat_model(
    X_train,
    y_train,
    train_edge_index,
    input_dim,
    device,
    epochs=50,
    hidden_dim=32,
    heads=4,
    learning_rate=0.001,
    weight_decay=5e-4
):
    print("=" * 70)
    print("Step 5: Training GAT model...")

    train_start = time.time()
    loss_history = []

    model = GATModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=2,
        heads=heads,
        dropout=0.5
    ).to(device)

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    train_edge_index = train_edge_index.to(device)

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
        logits = model(X_train, train_edge_index)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}")

    train_end = time.time()

    print("GAT training completed.")
    print(f"Model training time: {train_end - train_start:.2f} seconds")

    return model, loss_history


# 使用测试图评估 GAT 模型
# 输出 Recall、F1、AUC、FNR、FPR
def evaluate_gat_model(model, X_test, y_test, test_edge_index, device):
    print("=" * 70)
    print("Step 6: Evaluating GAT model...")

    eval_start = time.time()

    model.eval()

    X_test = X_test.to(device)
    y_test_device = y_test.to(device)
    test_edge_index = test_edge_index.to(device)

    with torch.no_grad():
        logits = model(X_test, test_edge_index)
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
        "Model": "GAT",
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


# 绘制 GAT 训练损失曲线
# 该图用于观察模型训练过程中 loss 是否下降
def plot_loss_curve(loss_history, figure_dir):
    figure_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("GAT Training Loss Curve")
    plt.grid(True)

    figure_path = figure_dir / "gat_loss_curve.png"
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

    plt.title("Sampled Transaction Subgraph for GAT")
    plt.axis("off")

    figure_path = figure_dir / "gat_transaction_subgraph.png"
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Transaction subgraph saved to : {figure_path}")


# 保存 GAT 实验结果
# 包括整体指标表和前 1000 条预测样例
def save_results(output_dir, metrics_df, test_df, y_pred, y_prob):
    print("=" * 70)
    print("Step 7: Saving results...")

    save_start = time.time()

    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "gat_metrics.csv"
    pred_path = output_dir / "gat_predictions_sample.csv"

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
# 顺序为：读取数据 -> 节点特征准备 -> 图构建 -> GAT训练 -> 评估 -> 画图 -> 保存结果
def run_gat_baseline(
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

    model, loss_history = train_gat_model(
        X_train=X_train,
        y_train=y_train,
        train_edge_index=train_edge_index,
        input_dim=X_train.shape[1],
        device=device,
        epochs=50,
        hidden_dim=32,
        heads=4,
        learning_rate=0.001,
        weight_decay=5e-4
    )

    metrics_df, y_pred, y_prob = evaluate_gat_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        test_edge_index=test_edge_index,
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
    print("GAT baseline completed successfully.")
    print(f"Total running time: {total_end - total_start:.2f} seconds")

    return model, metrics_df


# 程序入口
# 默认读取 data/processed 下的划分数据，并将结果保存到 output/tables 和 output/figures
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    processed_dir = project_root / "data" / "processed"
    output_dir = project_root / "output" / "tables"

    model, metrics_df = run_gat_baseline(
        processed_dir=processed_dir,
        output_dir=output_dir,
        train_file="train_split.csv",
        test_file="test_split.csv",
        label_col="isFraud",
        random_state=42
    )