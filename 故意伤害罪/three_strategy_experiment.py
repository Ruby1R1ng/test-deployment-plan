# -*- coding: utf-8 -*-
"""
三种部署策略对比实验。

策略一：训练后固定参数，验证/测试时只预测，不更新。
策略二：训练后只保存参数，验证/测试时用新的 Adam 状态继续在线更新。
策略三：训练后保存参数和 Adam 状态，验证/测试时接着训练阶段的 Adam 状态继续在线更新。

评估方式：
需要继续更新的策略采用“先预测，再更新”的流程，避免先看到当前样本标签。
"""

import argparse
import copy
import json
import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


LAMDA = 0
DEVICE = torch.device("cpu")

NOFLEE_LOWER = 6
NOFLEE_UPPER = 36
FLEE_REDUCE_LOWER = 6
FLEE_REDUCE_UPPER = 36
FLEE_LOWER = 6
FLEE_UPPER = 36
FLEE_DEATH_REDUCE_LOWER = 36
FLEE_DEATH_REDUCE_UPPER = 120
FLEE_DEATH_LOWER = 36
FLEE_DEATH_UPPER = 120

FLEEING_STARTPOINT = FLEE_LOWER + (FLEE_UPPER - FLEE_LOWER) * LAMDA
DEATH_FLEEING_STARTPOINT = FLEE_DEATH_LOWER + (FLEE_DEATH_UPPER - FLEE_DEATH_LOWER) * LAMDA


@dataclass
class DatasetTensors:
    a: torch.Tensor
    injuries: torch.Tensor
    prior: torch.Tensor
    phi: torch.Tensor
    y: torch.Tensor
    nn_features: torch.Tensor
    is_1: torch.Tensor
    is_2: torch.Tensor
    is_3: torch.Tensor
    is_4: torch.Tensor
    is_5: torch.Tensor
    case_ids: list

    def __len__(self):
        return int(self.y.shape[0])

    def take(self, start, end):
        return DatasetTensors(
            self.a[start:end],
            self.injuries[start:end],
            self.prior[start:end],
            self.phi[start:end],
            self.y[start:end],
            self.nn_features[start:end],
            self.is_1[start:end],
            self.is_2[start:end],
            self.is_3[start:end],
            self.is_4[start:end],
            self.is_5[start:end],
            self.case_ids[start:end],
        )

    def batch(self, indices):
        return DatasetTensors(
            self.a[indices],
            self.injuries[indices],
            self.prior[indices],
            self.phi[indices],
            self.y[indices],
            self.nn_features[indices],
            self.is_1[indices],
            self.is_2[indices],
            self.is_3[indices],
            self.is_4[indices],
            self.is_5[indices],
            [self.case_ids[int(i)] for i in indices.detach().cpu().tolist()],
        )


class SaturatedClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, lower, upper):
        ctx.save_for_backward(input_tensor, lower, upper)
        return torch.max(lower, torch.min(input_tensor, upper))

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, lower, upper = ctx.saved_tensors
        grad_input = grad_output.clone()

        inside = (input_tensor >= lower) & (input_tensor <= upper)
        below = input_tensor < lower
        above = input_tensor > upper

        grad_input[inside] = grad_output[inside]
        grad_input[below] = grad_output[below] * 0.1
        grad_input[above] = grad_output[above] * 0.1
        return grad_input, None, None


class CustomModel(nn.Module):
    def __init__(self, prior_feature_size, posterior_feature_size, nn_feature_size):
        super().__init__()

        self.theta1 = nn.Parameter(torch.tensor(
            [0.006053874192107, -0.014246674000886],
            dtype=torch.float32,
            device=DEVICE,
        ))
        self.theta2 = nn.Parameter(torch.tensor(
            [
                -0.071460721748652, 0, 0.071999335873278,
                0.041459795735327, -0.089429572096502,
                0.035230012975787, 0, 0.023275538768743,
                0.010928107534989, 0, -0.082097742167234, 0, 0,
            ],
            dtype=torch.float32,
            device=DEVICE,
        ))
        self.theta3 = nn.Parameter(torch.tensor(
            [
                0.628234424950179, 0.508776171673015, 0.199507011046998,
                0.0351936995326325, -0.131493764273348, 0.0391397386851738,
                -0.0499557203236371, -0.0856474377984399, -0.259742209776726,
                0.0415733433518542, -0.000231116326542458, -0.183433904285415,
                0.0933599811207825, 0.0786486168767389, 0.0505572262976934,
                -0.0170173130310525, 0.00100343640699517, -0.0453841678053260,
                0.0729842596077152, -0.0677786868839388, -0.136214961497343,
                -1.55250548703934,
            ],
            dtype=torch.float32,
            device=DEVICE,
        ))

        if prior_feature_size != self.theta2.numel():
            raise ValueError(f"先验特征数量为 {prior_feature_size}，但 theta2 长度为 {self.theta2.numel()}")
        if posterior_feature_size != self.theta3.numel():
            raise ValueError(f"后验特征数量为 {posterior_feature_size}，但 theta3 长度为 {self.theta3.numel()}")

        self.fc1 = nn.Linear(nn_feature_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, a, injuries, prior, phi, nn_features):
        hidden_input1 = a + torch.matmul(injuries, self.theta1)
        hidden_input2 = torch.prod(1 + prior * self.theta2, dim=1)
        hidden_input3 = 1 + torch.matmul(phi, self.theta3)

        x = self.relu(self.fc1(nn_features))
        x = self.relu(self.fc2(x))
        x = self.fc3(x).squeeze(1)

        inter_product = hidden_input1 * hidden_input2 * (hidden_input3 + x)
        inter_product_unbias = hidden_input1 * hidden_input2 * hidden_input3
        return inter_product, inter_product - x, x, inter_product_unbias


class RelativeAbsoluteErrorLoss(nn.Module):
    def forward(self, output, target):
        return torch.mean(torch.abs(output - target) / target)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_feature_lists(table):
    prior_features = [
        "16-18", "12-16", "75+",
        "减轻刑事责任的精神病人", "又聋又哑、盲人",
        "防卫过当", "紧急避险过当",
        "犯罪预备", "犯罪未遂", "犯罪中止",
        "从犯", "胁从犯", "教唆犯",
    ]
    base_features = ["重伤人数", "轻伤人数"]
    other_features = ["省份/直辖市", "案号", "判决时间", "有期徒刑", "减轻"]

    nn_features = [c for c in table.columns if c not in other_features + base_features]
    posterior_features = [
        c for c in table.columns
        if c not in prior_features + base_features + other_features
    ]
    if "缓刑" in posterior_features:
        posterior_features.remove("缓刑")
    posterior_features.append("缓刑")
    return prior_features, base_features, posterior_features, nn_features


def require_columns(table, columns):
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise ValueError("数据缺少必要列：" + ", ".join(missing))


def feature_extract(table, prior_features, base_features, posterior_features, nn_features):
    require_columns(
        table,
        ["案号", "有期徒刑"] + prior_features + base_features + posterior_features + nn_features,
    )

    prison_normal = table["有期徒刑"].values.astype(np.float32)
    prior_data = table.loc[:, prior_features].values.astype(np.float32)
    base_data = table.loc[:, base_features].values.astype(np.float32)
    posterior_data = table.loc[:, posterior_features].values.astype(np.float32)
    nn_data = table.loc[:, nn_features].values.astype(np.float32)

    row_count = len(table)
    a = np.zeros(row_count, dtype=np.float32)
    injuries = np.zeros((row_count, 2), dtype=np.float32)

    is_1 = np.zeros(row_count, dtype=np.float32)
    is_2 = np.zeros(row_count, dtype=np.float32)
    is_3 = np.zeros(row_count, dtype=np.float32)
    is_4 = np.zeros(row_count, dtype=np.float32)
    is_5 = np.zeros(row_count, dtype=np.float32)

    for idx in range(row_count):
        serious_count, minor_count = base_data[idx, :]
        if serious_count > 0:
            a[idx] = DEATH_FLEEING_STARTPOINT
            is_4[idx] = 1
            injuries[idx, :] = np.array([serious_count - 1, minor_count], dtype=np.float32)
        else:
            a[idx] = FLEEING_STARTPOINT
            is_1[idx] = 1
            injuries[idx, :] = np.array([serious_count, minor_count - 1], dtype=np.float32)

    return DatasetTensors(
        a=torch.tensor(a, dtype=torch.float32, device=DEVICE),
        injuries=torch.tensor(injuries, dtype=torch.float32, device=DEVICE),
        prior=torch.tensor(prior_data, dtype=torch.float32, device=DEVICE),
        phi=torch.tensor(posterior_data, dtype=torch.float32, device=DEVICE),
        y=torch.tensor(prison_normal, dtype=torch.float32, device=DEVICE),
        nn_features=torch.tensor(nn_data, dtype=torch.float32, device=DEVICE),
        is_1=torch.tensor(is_1, dtype=torch.float32, device=DEVICE),
        is_2=torch.tensor(is_2, dtype=torch.float32, device=DEVICE),
        is_3=torch.tensor(is_3, dtype=torch.float32, device=DEVICE),
        is_4=torch.tensor(is_4, dtype=torch.float32, device=DEVICE),
        is_5=torch.tensor(is_5, dtype=torch.float32, device=DEVICE),
        case_ids=table["案号"].tolist(),
    )


def split_train_val_test(dataset, train_ratio=0.6, val_ratio=0.2):
    total = len(dataset)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    return dataset.take(0, train_end), dataset.take(train_end, val_end), dataset.take(val_end, total)


def make_bounds(data):
    lower = torch.zeros_like(data.y, device=DEVICE)
    upper = torch.zeros_like(data.y, device=DEVICE)

    mask = data.is_1 == 1
    lower[mask] = NOFLEE_LOWER
    upper[mask] = NOFLEE_UPPER

    mask = data.is_2 == 1
    lower[mask] = FLEE_LOWER
    upper[mask] = FLEE_UPPER

    mask = data.is_3 == 1
    lower[mask] = FLEE_REDUCE_LOWER
    upper[mask] = FLEE_REDUCE_UPPER

    mask = data.is_4 == 1
    lower[mask] = FLEE_DEATH_LOWER
    upper[mask] = FLEE_DEATH_UPPER

    mask = data.is_5 == 1
    lower[mask] = FLEE_DEATH_REDUCE_LOWER
    upper[mask] = FLEE_DEATH_REDUCE_UPPER
    return lower, upper


def predict_raw(model, data):
    inter_product, mechanism, nn_output, inter_product_unbias = model(
        data.a, data.injuries, data.prior, data.phi, data.nn_features
    )
    lower, upper = make_bounds(data)
    output = SaturatedClamp.apply(inter_product, lower, upper)
    return output, nn_output, inter_product_unbias


def compute_precision(output, y):
    precision = 1 - torch.abs(y - output) / y
    return precision


def batch_indices(size, batch_size, shuffle=False):
    if shuffle:
        indices = torch.randperm(size, device=DEVICE)
    else:
        indices = torch.arange(size, device=DEVICE)
    for start in range(0, size, batch_size):
        yield indices[start:start + batch_size]


def train_base_model(model, train_data, epochs, batch_size, lr, alpha_penalty):
    criterion = RelativeAbsoluteErrorLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_precision = []
        epoch_loss = 0.0

        for indices in batch_indices(len(train_data), batch_size, shuffle=False):
            batch = train_data.batch(indices)
            optimizer.zero_grad()
            output, nn_output, _ = predict_raw(model, batch)
            loss = criterion(output, batch.y) + alpha_penalty * torch.abs(torch.mean(nn_output) + 0.0370)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().cpu())
            epoch_precision.append(compute_precision(output.detach(), batch.y).detach().cpu().numpy())

        mean_precision = float(np.concatenate(epoch_precision).mean())
        print(f"Epoch {epoch:03d}/{epochs} | train_precision={mean_precision:.6f} | loss={epoch_loss:.6f}")

    return copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())


@torch.no_grad()
def evaluate_fixed(model, data):
    model.eval()
    output, _, _ = predict_raw(model, data)
    precision = compute_precision(output, data.y)
    return float(precision.mean().cpu())


def evaluate_online(
    model,
    data,
    optimizer,
    batch_size,
    alpha_penalty,
):
    criterion = RelativeAbsoluteErrorLoss()
    all_precision = []

    for indices in batch_indices(len(data), batch_size, shuffle=False):
        batch = data.batch(indices)

        model.eval()
        with torch.no_grad():
            output_before_update, _, _ = predict_raw(model, batch)
            all_precision.append(
                compute_precision(output_before_update, batch.y).detach().cpu().numpy()
            )

        model.train()
        optimizer.zero_grad()
        output_for_update, nn_output, _ = predict_raw(model, batch)
        loss = criterion(output_for_update, batch.y) + alpha_penalty * torch.abs(torch.mean(nn_output) + 0.0370)
        loss.backward()
        optimizer.step()

    return float(np.concatenate(all_precision).mean())


def build_model(prior_features, posterior_features, nn_features):
    return CustomModel(
        prior_feature_size=len(prior_features),
        posterior_feature_size=len(posterior_features),
        nn_feature_size=len(nn_features),
    ).to(DEVICE)


def run_three_strategies(
    base_model_state,
    base_optimizer_state,
    data,
    prior_features,
    posterior_features,
    nn_features,
    lr,
    online_batch_size,
    alpha_penalty,
):
    results = {}

    fixed_model = build_model(prior_features, posterior_features, nn_features)
    fixed_model.load_state_dict(copy.deepcopy(base_model_state))
    results["策略一_固定参数"] = evaluate_fixed(fixed_model, data)

    params_only_model = build_model(prior_features, posterior_features, nn_features)
    params_only_model.load_state_dict(copy.deepcopy(base_model_state))
    params_only_optimizer = torch.optim.Adam(params_only_model.parameters(), lr=lr, betas=(0.9, 0.999))
    results["策略二_仅参数继续更新"] = evaluate_online(
        params_only_model, data, params_only_optimizer, online_batch_size, alpha_penalty
    )

    stateful_model = build_model(prior_features, posterior_features, nn_features)
    stateful_model.load_state_dict(copy.deepcopy(base_model_state))
    stateful_optimizer = torch.optim.Adam(stateful_model.parameters(), lr=lr, betas=(0.9, 0.999))
    stateful_optimizer.load_state_dict(copy.deepcopy(base_optimizer_state))
    for group in stateful_optimizer.param_groups:
        group["lr"] = lr
    results["策略三_参数加优化器状态继续更新"] = evaluate_online(
        stateful_model, data, stateful_optimizer, online_batch_size, alpha_penalty
    )

    return results


def print_result_table(title, results):
    print("\n" + title)
    print("-" * 56)
    for name, value in results.items():
        print(f"{name}: {value:.6f}")
    best_name = max(results, key=results.get)
    print(f"最佳策略: {best_name} | precision={results[best_name]:.6f}")
    return best_name


def main():
    parser = argparse.ArgumentParser(description="三种模型部署策略对比实验")
    parser.add_argument("--data", default="merged_data_filtered_minor_0508_3.xlsx", help="输入 Excel 数据文件")
    parser.add_argument("--epochs", type=int, default=30, help="基础模型训练轮数")
    parser.add_argument("--train-batch-size", type=int, default=245, help="训练 batch size")
    parser.add_argument("--online-batch-size", type=int, default=245, help="验证/测试阶段在线更新 batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="训练和在线更新学习率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--shuffle", action="store_true", help="划分前是否打乱数据；默认按原始顺序切分")
    parser.add_argument("--output-dir", default=os.path.join("outputs", "three_strategy_experiment"), help="结果输出目录")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    table = pd.read_excel(args.data)
    if args.shuffle:
        table = table.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    prior_features, base_features, posterior_features, nn_features = get_feature_lists(table)
    dataset = feature_extract(table, prior_features, base_features, posterior_features, nn_features)
    train_data, val_data, test_data = split_train_val_test(dataset)

    print(f"数据文件: {args.data}")
    print(f"样本划分: train={len(train_data)}, validation={len(val_data)}, test={len(test_data)}")
    print(f"特征数量: prior={len(prior_features)}, posterior={len(posterior_features)}, NN={len(nn_features)}")

    model = build_model(prior_features, posterior_features, nn_features)
    base_model_state, base_optimizer_state = train_base_model(
        model=model,
        train_data=train_data,
        epochs=args.epochs,
        batch_size=args.train_batch_size,
        lr=args.lr,
        alpha_penalty=1.4,
    )

    val_results = run_three_strategies(
        base_model_state=base_model_state,
        base_optimizer_state=base_optimizer_state,
        data=val_data,
        prior_features=prior_features,
        posterior_features=posterior_features,
        nn_features=nn_features,
        lr=args.lr,
        online_batch_size=args.online_batch_size,
        alpha_penalty=1.4,
    )
    best_val_strategy = print_result_table("验证集三种策略结果", val_results)

    test_results = run_three_strategies(
        base_model_state=base_model_state,
        base_optimizer_state=base_optimizer_state,
        data=test_data,
        prior_features=prior_features,
        posterior_features=posterior_features,
        nn_features=nn_features,
        lr=args.lr,
        online_batch_size=args.online_batch_size,
        alpha_penalty=1.4,
    )
    print_result_table("测试集三种策略结果", test_results)

    summary = pd.DataFrame([
        {"dataset": "validation", "strategy": name, "precision": value, "is_best_on_validation": name == best_val_strategy}
        for name, value in val_results.items()
    ] + [
        {"dataset": "test", "strategy": name, "precision": value, "is_best_on_validation": name == best_val_strategy}
        for name, value in test_results.items()
    ])
    summary_path = os.path.join(args.output_dir, "three_strategy_summary.xlsx")
    summary.to_excel(summary_path, index=False)

    state_path = os.path.join(args.output_dir, "base_model_and_adam_state.pth")
    torch.save(
        {
            "model_state_dict": base_model_state,
            "optimizer_state_dict": base_optimizer_state,
            "best_validation_strategy": best_val_strategy,
            "feature_lists": {
                "prior_features": prior_features,
                "base_features": base_features,
                "posterior_features": posterior_features,
                "nn_features": nn_features,
            },
            "args": vars(args),
        },
        state_path,
    )

    json_path = os.path.join(args.output_dir, "three_strategy_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_validation_strategy": best_val_strategy,
                "validation": val_results,
                "test": test_results,
                "split": {
                    "train": len(train_data),
                    "validation": len(val_data),
                    "test": len(test_data),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n验证集最佳策略: {best_val_strategy}")
    print(f"结果表已保存: {summary_path}")
    print(f"基础模型参数和 Adam 状态已保存: {state_path}")
    print(f"JSON 摘要已保存: {json_path}")


if __name__ == "__main__":
    main()
