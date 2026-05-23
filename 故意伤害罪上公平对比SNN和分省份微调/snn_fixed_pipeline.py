import argparse
import math
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


TARGET_COL = "有期徒刑"
CASE_ID_COL = "案号"
HIDDEN_PREFIX = "hidden2_"
RESULT_COLS = [
    "prediction_linear",
    "prediction_normal1",
    "prediction_normal2",
    "precision_linear",
    "precision_normal2",
]


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SaturatedActivation(torch.autograd.Function):
    LOWER = 6.0
    UPPER = 36.0
    EPSILON = 0.01

    @staticmethod
    def forward(ctx, input_tensor):
        output = input_tensor.clamp(
            min=SaturatedActivation.LOWER,
            max=SaturatedActivation.UPPER,
        )
        ctx.save_for_backward(input_tensor)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (input_tensor,) = ctx.saved_tensors
        mask = (
            (input_tensor > SaturatedActivation.LOWER)
            & (input_tensor < SaturatedActivation.UPPER)
        )
        grad_input = grad_output * torch.where(
            mask,
            torch.tensor(1.0, device=input_tensor.device),
            torch.tensor(SaturatedActivation.EPSILON, device=input_tensor.device),
        )
        return grad_input


class Saturate(nn.Module):
    def forward(self, x):
        return SaturatedActivation.apply(x)


class DualHiddenNet(nn.Module):
    def __init__(self, input_dim, hidden1=64, hidden2=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.fc_out = nn.Linear(hidden2, 1)
        self.saturate = Saturate()

    def forward(self, x, return_hidden2=False):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        hidden2 = x
        out = self.saturate(self.fc_out(x))
        if return_hidden2:
            return out, hidden2
        return out


@dataclass
class FixedAlgorithmResult:
    theta_linear: np.ndarray
    theta_normal1: np.ndarray
    theta_normal2: np.ndarray
    prediction_linear: np.ndarray
    prediction_normal1: np.ndarray
    prediction_normal2: np.ndarray
    precision_linear: np.ndarray
    precision_normal2: np.ndarray
    mean_precision_linear: float
    mean_precision_normal2: float
    subset_mean_precision_linear: float | None
    subset_mean_precision_normal2: float | None
    subset_start_case_id: str | None


def relative_abs_loss(output, target):
    return torch.mean(torch.abs(output - target) / target)


def compute_precision(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    precision = np.zeros_like(y_true, dtype=np.float64)
    for idx, true_val in enumerate(y_true):
        diff = abs(true_val - y_pred[idx])
        if diff / true_val < 0.2 or diff < 2:
            error_val = 0.0
        else:
            error_val = diff / true_val
        precision[idx] = 1.0 - error_val
    return precision


def projection_box_qp(theta_candidate, q_matrix, lower, upper):
    theta_candidate = np.asarray(theta_candidate, dtype=np.float64)
    q_matrix = np.asarray(q_matrix, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    if np.all(theta_candidate >= lower) and np.all(theta_candidate <= upper):
        return theta_candidate

    sym_q = 0.5 * (q_matrix + q_matrix.T)

    def objective(x):
        delta = x - theta_candidate
        return 0.5 * float(delta.T @ sym_q @ delta)

    def gradient(x):
        delta = x - theta_candidate
        return sym_q @ delta

    bounds = list(zip(lower, upper))
    start = np.clip(theta_candidate, lower, upper)
    result = minimize(
        objective,
        start,
        jac=gradient,
        method="L-BFGS-B",
        bounds=bounds,
    )
    if not result.success:
        return np.clip(theta_candidate, lower, upper)
    return result.x


def s_minor(x, lower=6.0, upper=36.0, meanu=0.0, standard=5.0):
    x = np.asarray(x, dtype=np.float64)
    term1 = lower * norm.cdf(lower - x, loc=meanu, scale=standard)
    term2 = upper * (1.0 - norm.cdf(upper - x, loc=meanu, scale=standard))
    term3 = (
        np.exp(-((lower - x - meanu) ** 2) / (2.0 * standard**2))
        - np.exp(-((upper - x - meanu) ** 2) / (2.0 * standard**2))
    ) * standard / math.sqrt(2.0 * math.pi)
    term4 = (x + meanu) * (
        norm.cdf(upper, loc=x + meanu, scale=standard)
        - norm.cdf(lower, loc=x + meanu, scale=standard)
    )
    return term1 + term2 + term3 + term4


def normalize_hidden_df(hidden_df):
    hidden_cols = [
        col
        for col in hidden_df.columns
        if isinstance(col, str) and col.startswith(HIDDEN_PREFIX)
    ]
    if not hidden_cols:
        raise ValueError("hidden 输入文件里没有找到 hidden2_* 特征列。")

    required_cols = [CASE_ID_COL, TARGET_COL]
    missing_cols = [col for col in required_cols if col not in hidden_df.columns]
    if missing_cols:
        raise ValueError(f"hidden 输入文件缺少必要列: {missing_cols}")

    def hidden_index(col_name):
        try:
            return int(col_name.split("_", 1)[1])
        except (IndexError, ValueError):
            return 10**9

    hidden_cols = sorted(hidden_cols, key=hidden_index)
    clean_df = hidden_df[[CASE_ID_COL, *hidden_cols, TARGET_COL]].copy()
    clean_df = clean_df.dropna(subset=hidden_cols + [TARGET_COL]).reset_index(drop=True)
    return clean_df


def run_fixed_algorithm(hidden_df, subset_case_id=None):
    hidden_df = normalize_hidden_df(hidden_df)
    hidden_cols = [col for col in hidden_df.columns if col.startswith(HIDDEN_PREFIX)]
    x_data = hidden_df[hidden_cols].to_numpy(dtype=np.float64)
    y_data = hidden_df[TARGET_COL].to_numpy(dtype=np.float64)
    case_ids = hidden_df[CASE_ID_COL].astype(str).to_numpy()

    n_samples, hidden_dim = x_data.shape
    phi_dim = 1 + hidden_dim
    meanu = 0.0
    standard = 5.0
    minor_lower = 6.0
    minor_upper = 36.0
    sigma_val = 25.0

    lower_bound = -5.0 * np.ones(phi_dim)
    upper_bound = 5.0 * np.ones(phi_dim)

    p_linear = np.eye(phi_dim)
    theta_linear = np.zeros(phi_dim)
    prediction_linear = np.zeros(n_samples)

    for idx in range(n_samples):
        phi = np.concatenate(([1.0], x_data[idx]))
        inter = float(phi @ theta_linear)
        gain = 1.0 / (1.0 + phi @ p_linear @ phi)
        theta_linear = theta_linear + gain * (p_linear @ phi) * (y_data[idx] - inter)
        p_linear = p_linear - gain * np.outer(p_linear @ phi, phi @ p_linear)
        prediction_linear[idx] = inter

    p_normal1 = np.eye(phi_dim)
    p_normal2 = np.eye(phi_dim)
    q_normal1 = np.eye(phi_dim)
    q_normal2 = np.eye(phi_dim)
    theta_normal1 = theta_linear.copy()
    theta_normal2 = theta_linear.copy()
    prediction_normal1 = np.zeros(n_samples)
    prediction_normal2 = np.zeros(n_samples)

    for idx in range(n_samples):
        phi = np.concatenate(([1.0], x_data[idx]))

        inter1 = float(phi @ theta_normal1)
        inter2 = float(phi @ theta_normal2)
        inter3 = float(phi @ lower_bound)
        inter4 = float(phi @ upper_bound)

        beta_21 = norm.cdf(36.0 - inter3, loc=meanu, scale=standard) - norm.cdf(
            6.0 - inter3, loc=meanu, scale=standard
        )
        beta_22 = norm.cdf(36.0 - inter4, loc=meanu, scale=standard) - norm.cdf(
            6.0 - inter4, loc=meanu, scale=standard
        )
        beta2 = min(beta_21, beta_22)

        if abs(inter1 - inter2) < 1e-10:
            beta4 = norm.cdf(36.0 - inter2, loc=meanu, scale=standard) - norm.cdf(
                6.0 - inter2, loc=meanu, scale=standard
            )
        else:
            beta4 = float(
                (s_minor(inter1, minor_lower, minor_upper, meanu, standard)
                 - s_minor(inter2, minor_lower, minor_upper, meanu, standard))
                / (inter1 - inter2)
            )

        a1 = 1.0 / (1.0 + beta2**2 * (phi @ p_normal1 @ phi))
        theta1_candidate = theta_normal1 + a1 * (p_normal1 @ phi) * beta2 * (
            y_data[idx] - s_minor(inter1, minor_lower, minor_upper, meanu, standard)
        )
        p_normal1 = p_normal1 - a1 * beta2**2 * np.outer(
            p_normal1 @ phi,
            phi @ p_normal1,
        )
        q_normal1 = q_normal1 + beta2**2 * np.outer(phi, phi)

        a2 = 1.0 / (sigma_val + beta4**2 * (phi @ p_normal2 @ phi))
        theta2_candidate = theta_normal2 + a2 * (p_normal2 @ phi) * beta4 * (
            y_data[idx] - s_minor(inter2, minor_lower, minor_upper, meanu, standard)
        )
        p_normal2 = p_normal2 - a2 * beta4**2 * np.outer(
            p_normal2 @ phi,
            phi @ p_normal2,
        )
        q_normal2 = q_normal2 + beta4**2 * np.outer(phi, phi)

        theta_normal1 = projection_box_qp(
            theta1_candidate,
            q_normal1,
            lower_bound,
            upper_bound,
        )
        theta_normal2 = projection_box_qp(
            theta2_candidate,
            q_normal2,
            lower_bound,
            upper_bound,
        )

        prediction_normal1[idx] = s_minor(
            inter1,
            minor_lower,
            minor_upper,
            meanu,
            standard,
        )
        prediction_normal2[idx] = s_minor(
            inter2,
            minor_lower,
            minor_upper,
            meanu,
            standard,
        )

    precision_linear = compute_precision(y_data, prediction_linear)
    precision_normal2 = compute_precision(y_data, prediction_normal2)

    subset_mean_precision_linear = None
    subset_mean_precision_normal2 = None
    subset_start_value = None

    if subset_case_id is not None:
        match_idx = np.where(case_ids == subset_case_id)[0]
        if match_idx.size > 0:
            start_idx = int(match_idx[0])
            subset_mean_precision_linear = float(np.mean(precision_linear[start_idx:]))
            subset_mean_precision_normal2 = float(np.mean(precision_normal2[start_idx:]))
            subset_start_value = subset_case_id

    return FixedAlgorithmResult(
        theta_linear=theta_linear,
        theta_normal1=theta_normal1,
        theta_normal2=theta_normal2,
        prediction_linear=prediction_linear,
        prediction_normal1=prediction_normal1,
        prediction_normal2=prediction_normal2,
        precision_linear=precision_linear,
        precision_normal2=precision_normal2,
        mean_precision_linear=float(np.mean(precision_linear)),
        mean_precision_normal2=float(np.mean(precision_normal2)),
        subset_mean_precision_linear=subset_mean_precision_linear,
        subset_mean_precision_normal2=subset_mean_precision_normal2,
        subset_start_case_id=subset_start_value,
    )


def train_snn_and_export_hidden(
    all_data_path,
    province_data_path,
    hidden_output_path,
    hidden1=64,
    hidden2=32,
    batch_size=64,
    learning_rate=0.001,
    epochs=200,
    seed=42,
):
    set_random_seed(seed)
    df_all = pd.read_excel(all_data_path)
    ignore_cols = [CASE_ID_COL, "省份/直辖市", "判决时间", TARGET_COL]
    feature_cols = [col for col in df_all.columns if col not in ignore_cols]

    x_all = df_all[feature_cols].to_numpy(dtype=np.float32)
    y_all = df_all[TARGET_COL].to_numpy(dtype=np.float32).reshape(-1, 1)

    valid_mask = ~np.isnan(x_all).any(axis=1) & ~np.isnan(y_all).any(axis=1)
    df_all_valid = df_all.loc[valid_mask].reset_index(drop=True)
    x_all = x_all[valid_mask]
    y_all = y_all[valid_mask]

    split_idx = int(0.8 * len(x_all))
    x_train = x_all[:split_idx]
    x_val = x_all[split_idx:]
    y_train = y_all[:split_idx]
    y_val = y_all[split_idx:]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(x_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
    )

    model = DualHiddenNet(x_train.shape[1], hidden1=hidden1, hidden2=hidden2)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = relative_abs_loss(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val_t)
            val_loss = relative_abs_loss(val_pred, y_val_t).item()

        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}, "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )

    df_sd = pd.read_excel(province_data_path)
    x_sd = df_sd[feature_cols].to_numpy(dtype=np.float32)
    valid_mask_sd = ~np.isnan(x_sd).any(axis=1)
    df_sd_valid = df_sd.loc[valid_mask_sd].reset_index(drop=True)
    x_sd = x_sd[valid_mask_sd]
    y_sd = df_sd_valid[TARGET_COL].to_numpy(dtype=np.float32)

    x_sd_scaled = scaler.transform(x_sd)
    x_sd_t = torch.tensor(x_sd_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        _, hidden2_out = model(x_sd_t, return_hidden2=True)
    hidden_np = hidden2_out.cpu().numpy()

    hidden_cols = [f"hidden2_{idx + 1}" for idx in range(hidden_np.shape[1])]
    hidden_df = pd.DataFrame(hidden_np, columns=hidden_cols)
    hidden_df.insert(0, CASE_ID_COL, df_sd_valid[CASE_ID_COL].to_numpy())
    hidden_df[TARGET_COL] = y_sd
    hidden_df.to_excel(hidden_output_path, index=False)

    df_test = df_all_valid.iloc[split_idx:].copy()
    df_sd_test = df_test[df_test["省份/直辖市"] == "山东省"].copy()

    province_holdout_precision = None
    province_first_case = None
    if not df_sd_test.empty:
        x_sd_test = df_sd_test[feature_cols].to_numpy(dtype=np.float32)
        y_sd_test = df_sd_test[TARGET_COL].to_numpy(dtype=np.float32)
        x_sd_test_scaled = scaler.transform(x_sd_test)
        x_sd_test_t = torch.tensor(x_sd_test_scaled, dtype=torch.float32)
        with torch.no_grad():
            pred_sd_test = model(x_sd_test_t).cpu().numpy().reshape(-1)
        precision_sd_test = compute_precision(y_sd_test, pred_sd_test)
        province_holdout_precision = float(np.mean(precision_sd_test))
        province_first_case = str(df_sd_test.iloc[0][CASE_ID_COL])

    return {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "hidden_df": hidden_df,
        "province_holdout_precision": province_holdout_precision,
        "province_first_case": province_first_case,
    }


def build_summary_markdown(
    summary_path,
    all_data_path,
    province_data_path,
    hidden_output_path,
    snn_result,
    fixed_result,
    result_output_path,
    reuse_hidden,
):
    lines = [
        "# SNN + fixed 自适应预测流程说明",
        "",
        "## 这份代码做了什么",
        "",
        "这份 Python 代码把原来的 `SNN参数.py` 和 `adaptive_prediction_fixed.m` 串成了一条完整流程。",
        "它可以先用全国数据训练一个双隐层神经网络，再把山东数据送进这个网络，导出第二个隐层的表示向量；随后只用 hidden2_* 列作为自适应算法输入，执行固定维度递推预测与精度评估。",
        "",
        "## 输入文件",
        "",
        f"- 全国训练数据：`{all_data_path}`",
        f"- 山东数据：`{province_data_path}`",
        "",
        "## 主要步骤",
        "",
        "1. 读取全国数据，删除含缺失值的样本，并按前 80% / 后 20% 切分训练集和验证集。",
        "2. 对特征做标准化，并训练一个双隐层神经网络。",
        "3. 用训练好的网络处理山东数据，提取第二个隐层输出。",
        f"4. 把隐层输出保存为 `{hidden_output_path}`，作为后续自适应算法的输入。",
        "5. 在这些隐层特征上运行 fixed 版本的递推最小二乘/投影自适应算法。",
        "6. hidden 文件只保存案号、hidden2_*、有期徒刑；预测结果另存，避免下次把预测列误当特征。",
        "",
        "## 关键含义",
        "",
        "- 神经网络部分负责从原始案件特征里提取更有表达能力的隐层特征。",
        "- fixed 自适应算法部分负责在山东省数据上继续按顺序微调外层参数。",
        "- 整体上，这是一种“全国样本预训练 + 省级样本递推修正”的两阶段预测方法。",
        "",
        "## 本次脚本输出",
        "",
        f"- 隐层输出文件：`{hidden_output_path}`",
        f"- 预测结果文件：`{result_output_path}`",
        f"- 是否复用已有 hidden 文件：`{reuse_hidden}`",
        f"- 山东省在全国数据后 20% 留出集上的神经网络平均预测精度：`{snn_result['province_holdout_precision']}`",
        f"- fixed 线性算法平均预测精度：`{fixed_result.mean_precision_linear:.4f}`",
        f"- fixed 非线性自适应算法平均预测精度：`{fixed_result.mean_precision_normal2:.4f}`",
    ]

    if snn_result["province_first_case"] is not None:
        lines.append(
            f"- 后 20% 留出集里第一条山东样本案号：`{snn_result['province_first_case']}`"
        )

    if fixed_result.subset_start_case_id is not None:
        lines.extend(
            [
                f"- 从案号 `{fixed_result.subset_start_case_id}` 开始的线性算法平均预测精度："
                f" `{fixed_result.subset_mean_precision_linear:.4f}`",
                f"- 从案号 `{fixed_result.subset_start_case_id}` 开始的非线性自适应算法平均预测精度："
                f" `{fixed_result.subset_mean_precision_normal2:.4f}`",
            ]
        )

    lines.extend(
        [
            "",
            "## 运行方式",
            "",
            "```bash",
            "python3 snn_fixed_pipeline.py",
            "```",
        ]
    )

    with open(summary_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="把 SNN 参数提取和 fixed 自适应算法整合成一个完整 Python 流程。"
    )
    parser.add_argument(
        "--all-data",
        default="merged_data_filtered_minor_0919_6_qwen_all.xlsx",
        help="全国训练数据 Excel 文件路径。",
    )
    parser.add_argument(
        "--province-data",
        default="shandong.xlsx",
        help="省级数据 Excel 文件路径。",
    )
    parser.add_argument(
        "--hidden-output",
        default="shandong_hidden_output_64.xlsx",
        help="导出的第二隐层输出文件路径；默认与 adaptive_prediction_fixed.m 保持一致。",
    )
    parser.add_argument(
        "--result-output",
        default="snn_fixed_pipeline_results.xlsx",
        help="预测结果输出文件路径；不会写回 hidden 文件，避免污染 hidden 特征。",
    )
    parser.add_argument(
        "--summary-md",
        default="snn_fixed_pipeline说明.md",
        help="自动生成的说明 Markdown 文件路径。",
    )
    parser.add_argument(
        "--subset-case-id",
        default="（2021）鲁0704刑初17号",
        help="从该案号开始统计子区间平均精度；若不存在则跳过。",
    )
    parser.add_argument("--hidden1", type=int, default=64)
    parser.add_argument("--hidden2", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reuse-hidden",
        action="store_true",
        help="直接复用 --hidden-output 指定的 hidden 文件，只运行 fixed 自适应算法。",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.reuse_hidden:
        hidden_df = normalize_hidden_df(pd.read_excel(args.hidden_output))
        snn_result = {
            "model": None,
            "scaler": None,
            "feature_cols": None,
            "hidden_df": hidden_df,
            "province_holdout_precision": None,
            "province_first_case": None,
        }
    else:
        snn_result = train_snn_and_export_hidden(
            all_data_path=args.all_data,
            province_data_path=args.province_data,
            hidden_output_path=args.hidden_output,
            hidden1=args.hidden1,
            hidden2=args.hidden2,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            seed=args.seed,
        )

    fixed_result = run_fixed_algorithm(
        snn_result["hidden_df"],
        subset_case_id=args.subset_case_id,
    )

    result_df = snn_result["hidden_df"].copy()
    result_df["prediction_linear"] = fixed_result.prediction_linear
    result_df["prediction_normal1"] = fixed_result.prediction_normal1
    result_df["prediction_normal2"] = fixed_result.prediction_normal2
    result_df["precision_linear"] = fixed_result.precision_linear
    result_df["precision_normal2"] = fixed_result.precision_normal2
    result_df.to_excel(args.result_output, index=False)

    build_summary_markdown(
        summary_path=args.summary_md,
        all_data_path=args.all_data,
        province_data_path=args.province_data,
        hidden_output_path=args.hidden_output,
        snn_result=snn_result,
        fixed_result=fixed_result,
        result_output_path=args.result_output,
        reuse_hidden=args.reuse_hidden,
    )

    print("SNN/fixed 自适应预测流程已完成。")
    if snn_result["province_holdout_precision"] is not None:
        print(
            "山东省数据在全国数据后20%留出集上的神经网络平均预测精度: "
            f"{snn_result['province_holdout_precision']:.4f}"
        )
    print(f"fixed 线性算法平均预测精度: {fixed_result.mean_precision_linear:.4f}")
    print(f"fixed 非线性自适应算法平均预测精度: {fixed_result.mean_precision_normal2:.4f}")
    if fixed_result.subset_start_case_id is not None:
        print(
            f"从案号 {fixed_result.subset_start_case_id} 开始的 fixed 线性算法平均预测精度: "
            f"{fixed_result.subset_mean_precision_linear:.4f}"
        )
        print(
            f"从案号 {fixed_result.subset_start_case_id} 开始的 fixed 非线性自适应算法平均预测精度: "
            f"{fixed_result.subset_mean_precision_normal2:.4f}"
        )
    print(f"干净隐层文件已保存/读取自: {args.hidden_output}")
    print(f"预测结果已保存到: {args.result_output}")
    print(f"说明文档已保存到: {args.summary_md}")


if __name__ == "__main__":
    main()
