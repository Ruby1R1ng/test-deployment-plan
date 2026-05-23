# -*- coding: utf-8 -*-
"""
公平比较 SNN+fixed 自适应方法与分省份离线微调方法。

划分规则：
- 每个省份按原始顺序切分，后 20% 固定为测试集。
- 前 80% 再切为 sub-train 和 validation，默认 sub-train 占前 80% 的 75%，
  即整体约 60% / 20% / 20%。
- 两种方法使用完全相同的省份测试集。

控制台和 Excel 同时输出两个精度指标：
- strict_precision: 1 - mean(abs(y - pred) / y)
- tolerant_precision: 与 snn_fixed_pipeline.py 相同，误差小于 20% 或小于 2 个月计为 0。
"""

import argparse
import copy
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
OFFLINE_DIR = ROOT_DIR / "故意伤害罪上公平对比SNN和分省份微调"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(OFFLINE_DIR) not in sys.path:
    sys.path.insert(0, str(OFFLINE_DIR))

from snn_fixed_pipeline import (
    CASE_ID_COL,
    HIDDEN_PREFIX,
    TARGET_COL,
    DualHiddenNet,
    projection_box_qp,
    relative_abs_loss,
    s_minor,
    set_random_seed,
)
from province_offline_finetune_experiment import train_one_epoch
from three_strategy_experiment import (  # noqa: E402
    batch_indices,
    build_model,
    feature_extract,
    get_feature_lists,
    predict_raw,
    set_seed,
    train_base_model,
)


PROVINCE_COL = "省份/直辖市"
TIME_COL = "判决时间"
STRATEGY_PARAMS_ONLY = "策略二_仅参数离线微调"
STRATEGY_PARAMS_ADAM = "策略三_参数加Adam状态离线微调"


def strict_precision_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    return float(np.mean(1.0 - np.abs(y_true - y_pred) / y_true))


def tolerant_precision_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    diff = np.abs(y_true - y_pred)
    relative_error = diff / y_true
    penalized_error = np.where((relative_error < 0.2) | (diff < 2.0), 0.0, relative_error)
    return float(np.mean(1.0 - penalized_error))


def metric_pair(y_true, y_pred):
    return {
        "strict_precision": strict_precision_np(y_true, y_pred),
        "tolerant_precision": tolerant_precision_np(y_true, y_pred),
    }


def split_by_province(table, province_col, time_col, test_ratio, subtrain_ratio, min_samples):
    subtrain_parts = []
    val_parts = []
    support_parts = []
    test_parts = []
    rows = []

    for province, group in table.groupby(province_col, sort=False):
        group = group.copy()
        if time_col in group.columns:
            parsed_time = pd.to_datetime(group[time_col], errors="coerce")
            if parsed_time.notna().any():
                group = (
                    group.assign(_parsed_time=parsed_time)
                    .sort_values("_parsed_time", kind="mergesort")
                    .drop(columns=["_parsed_time"])
                )

        n = len(group)
        test_start = int(n * (1.0 - test_ratio))
        support = group.iloc[:test_start].copy()
        test = group.iloc[test_start:].copy()
        val_start = int(len(support) * subtrain_ratio)
        subtrain = support.iloc[:val_start].copy()
        val = support.iloc[val_start:].copy()
        ok = len(subtrain) >= min_samples and len(val) >= min_samples and len(test) >= min_samples

        if ok:
            subtrain_parts.append(subtrain)
            val_parts.append(val)
            support_parts.append(support)
            test_parts.append(test)

        rows.append({
            "省份": province,
            "总样本数": n,
            "sub_train样本数": len(subtrain),
            "验证样本数": len(val),
            "测试样本数": len(test),
            "是否参与评估": ok,
        })

    def concat(parts):
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=table.columns)

    return concat(subtrain_parts), concat(val_parts), concat(support_parts), concat(test_parts), pd.DataFrame(rows)


def clean_table_for_both_methods(table, province_col):
    ignore_cols = [CASE_ID_COL, province_col, TIME_COL, TARGET_COL]
    snn_feature_cols = [col for col in table.columns if col not in ignore_cols]
    required = [CASE_ID_COL, province_col, TARGET_COL] + snn_feature_cols
    before = len(table)
    clean = table.dropna(subset=required).reset_index(drop=True)
    dropped = before - len(clean)
    return clean, snn_feature_cols, dropped


def train_snn_model(train_df, val_df, feature_cols, args):
    set_random_seed(args.seed)
    scaler = StandardScaler()
    x_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df[TARGET_COL].to_numpy(dtype=np.float32).reshape(-1, 1)
    x_val = val_df[feature_cols].to_numpy(dtype=np.float32)
    y_val = val_df[TARGET_COL].to_numpy(dtype=np.float32).reshape(-1, 1)

    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train_scaled, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=args.snn_batch_size,
        shuffle=True,
    )

    model = DualHiddenNet(len(feature_cols), hidden1=args.snn_hidden1, hidden2=args.snn_hidden2)
    optimizer = optim.Adam(model.parameters(), lr=args.snn_lr)

    x_val_t = torch.tensor(x_val_scaled, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    for epoch in range(1, args.snn_epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = relative_abs_loss(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        if epoch % args.print_every == 0 or epoch == args.snn_epochs:
            model.eval()
            with torch.no_grad():
                val_loss = relative_abs_loss(model(x_val_t), y_val_t).item()
            print(f"SNN epoch {epoch:03d}/{args.snn_epochs} | val_loss={val_loss:.6f}")
    return model, scaler


def build_hidden_df(model, scaler, df, feature_cols):
    x_scaled = scaler.transform(df[feature_cols].to_numpy(dtype=np.float32))
    model.eval()
    with torch.no_grad():
        _, hidden = model(torch.tensor(x_scaled, dtype=torch.float32), return_hidden2=True)
    hidden_np = hidden.cpu().numpy()
    hidden_cols = [f"{HIDDEN_PREFIX}{idx + 1}" for idx in range(hidden_np.shape[1])]
    hidden_df = pd.DataFrame(hidden_np, columns=hidden_cols)
    hidden_df.insert(0, CASE_ID_COL, df[CASE_ID_COL].to_numpy())
    hidden_df[TARGET_COL] = df[TARGET_COL].to_numpy(dtype=np.float64)
    return hidden_df


@dataclass
class FixedAdaptiveState:
    theta_linear: np.ndarray
    p_linear: np.ndarray
    theta_normal1: np.ndarray | None
    theta_normal2: np.ndarray | None
    p_normal1: np.ndarray
    p_normal2: np.ndarray
    q_normal1: np.ndarray
    q_normal2: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray


def init_fixed_state(hidden_dim):
    phi_dim = hidden_dim + 1
    return FixedAdaptiveState(
        theta_linear=np.zeros(phi_dim),
        p_linear=np.eye(phi_dim),
        theta_normal1=None,
        theta_normal2=None,
        p_normal1=np.eye(phi_dim),
        p_normal2=np.eye(phi_dim),
        q_normal1=np.eye(phi_dim),
        q_normal2=np.eye(phi_dim),
        lower_bound=-5.0 * np.ones(phi_dim),
        upper_bound=5.0 * np.ones(phi_dim),
    )


def update_linear(state, phi, y):
    pred = float(phi @ state.theta_linear)
    gain = 1.0 / (1.0 + phi @ state.p_linear @ phi)
    state.theta_linear = state.theta_linear + gain * (state.p_linear @ phi) * (y - pred)
    state.p_linear = state.p_linear - gain * np.outer(state.p_linear @ phi, phi @ state.p_linear)
    return pred


def init_normal_from_linear(state):
    state.theta_normal1 = state.theta_linear.copy()
    state.theta_normal2 = state.theta_linear.copy()


def update_normal(state, phi, y):
    meanu = 0.0
    standard = 5.0
    minor_lower = 6.0
    minor_upper = 36.0
    sigma_val = 25.0

    inter1 = float(phi @ state.theta_normal1)
    inter2 = float(phi @ state.theta_normal2)
    inter3 = float(phi @ state.lower_bound)
    inter4 = float(phi @ state.upper_bound)

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

    a1 = 1.0 / (1.0 + beta2**2 * (phi @ state.p_normal1 @ phi))
    theta1_candidate = state.theta_normal1 + a1 * (state.p_normal1 @ phi) * beta2 * (
        y - s_minor(inter1, minor_lower, minor_upper, meanu, standard)
    )
    state.p_normal1 = state.p_normal1 - a1 * beta2**2 * np.outer(
        state.p_normal1 @ phi,
        phi @ state.p_normal1,
    )
    state.q_normal1 = state.q_normal1 + beta2**2 * np.outer(phi, phi)

    a2 = 1.0 / (sigma_val + beta4**2 * (phi @ state.p_normal2 @ phi))
    theta2_candidate = state.theta_normal2 + a2 * (state.p_normal2 @ phi) * beta4 * (
        y - s_minor(inter2, minor_lower, minor_upper, meanu, standard)
    )
    state.p_normal2 = state.p_normal2 - a2 * beta4**2 * np.outer(
        state.p_normal2 @ phi,
        phi @ state.p_normal2,
    )
    state.q_normal2 = state.q_normal2 + beta4**2 * np.outer(phi, phi)

    state.theta_normal1 = projection_box_qp(
        theta1_candidate,
        state.q_normal1,
        state.lower_bound,
        state.upper_bound,
    )
    state.theta_normal2 = projection_box_qp(
        theta2_candidate,
        state.q_normal2,
        state.lower_bound,
        state.upper_bound,
    )
    return float(s_minor(inter1, minor_lower, minor_upper, meanu, standard)), float(
        s_minor(inter2, minor_lower, minor_upper, meanu, standard)
    )


def evaluate_snn_fixed_for_province(model, scaler, support_df, test_df, feature_cols, update_on_test):
    support_hidden = build_hidden_df(model, scaler, support_df, feature_cols)
    test_hidden = build_hidden_df(model, scaler, test_df, feature_cols)
    hidden_cols = [c for c in support_hidden.columns if c.startswith(HIDDEN_PREFIX)]
    state = init_fixed_state(len(hidden_cols))

    for _, row in support_hidden.iterrows():
        phi = np.concatenate(([1.0], row[hidden_cols].to_numpy(dtype=np.float64)))
        update_linear(state, phi, float(row[TARGET_COL]))

    init_normal_from_linear(state)
    for _, row in support_hidden.iterrows():
        phi = np.concatenate(([1.0], row[hidden_cols].to_numpy(dtype=np.float64)))
        update_normal(state, phi, float(row[TARGET_COL]))

    y_true = test_hidden[TARGET_COL].to_numpy(dtype=np.float64)
    pred_linear = []
    pred_normal2 = []
    for _, row in test_hidden.iterrows():
        phi = np.concatenate(([1.0], row[hidden_cols].to_numpy(dtype=np.float64)))
        y = float(row[TARGET_COL])
        if update_on_test:
            pred_linear.append(update_linear(state, phi, y))
            _, normal2_pred = update_normal(state, phi, y)
            pred_normal2.append(normal2_pred)
        else:
            pred_linear.append(float(phi @ state.theta_linear))
            pred_normal2.append(float(s_minor(phi @ state.theta_normal2)))

    return {
        "linear": {"y_true": y_true, "y_pred": np.asarray(pred_linear, dtype=np.float64)},
        "normal2": {"y_true": y_true, "y_pred": np.asarray(pred_normal2, dtype=np.float64)},
    }


def model_predictions_np(model, data):
    model.eval()
    with torch.no_grad():
        output, _, _ = predict_raw(model, data)
    return data.y.detach().cpu().numpy(), output.detach().cpu().numpy()


def offline_finetune_with_predictions(
    base_model_state,
    base_optimizer_state,
    train_data,
    val_data,
    test_data,
    prior_features,
    posterior_features,
    nn_features,
    args,
    use_saved_adam_state,
):
    model = build_model(prior_features, posterior_features, nn_features)
    model.load_state_dict(copy.deepcopy(base_model_state))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    if use_saved_adam_state:
        optimizer.load_state_dict(copy.deepcopy(base_optimizer_state))
        for group in optimizer.param_groups:
            group["lr"] = args.lr

    y_val, pred_val = model_predictions_np(model, val_data)
    best_val = strict_precision_np(y_val, pred_val)
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, args.finetune_epochs + 1):
        train_one_epoch(model, optimizer, train_data, args.finetune_batch_size, args.alpha_penalty)
        y_val, pred_val = model_predictions_np(model, val_data)
        val_precision = strict_precision_np(y_val, pred_val)
        if val_precision > best_val:
            best_val = val_precision
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    best_model = build_model(prior_features, posterior_features, nn_features)
    best_model.load_state_dict(best_state)
    y_test, pred_test = model_predictions_np(best_model, test_data)
    return best_val, best_epoch, y_test, pred_test


def evaluate_offline_for_province(
    base_model_state,
    base_optimizer_state,
    train_p,
    val_p,
    test_p,
    prior_features,
    base_features,
    posterior_features,
    nn_features,
    args,
):
    train_data = feature_extract(train_p, prior_features, base_features, posterior_features, nn_features)
    val_data = feature_extract(val_p, prior_features, base_features, posterior_features, nn_features)
    test_data = feature_extract(test_p, prior_features, base_features, posterior_features, nn_features)

    y_test = test_data.y.detach().cpu().numpy()

    params_val, params_epoch, _, params_test_pred = offline_finetune_with_predictions(
        base_model_state,
        base_optimizer_state,
        train_data,
        val_data,
        test_data,
        prior_features,
        posterior_features,
        nn_features,
        args,
        use_saved_adam_state=False,
    )
    adam_val, adam_epoch, _, adam_test_pred = offline_finetune_with_predictions(
        base_model_state,
        base_optimizer_state,
        train_data,
        val_data,
        test_data,
        prior_features,
        posterior_features,
        nn_features,
        args,
        use_saved_adam_state=True,
    )

    candidates = {
        STRATEGY_PARAMS_ONLY: {"val": params_val, "epoch": params_epoch, "y_pred": params_test_pred},
        STRATEGY_PARAMS_ADAM: {"val": adam_val, "epoch": adam_epoch, "y_pred": adam_test_pred},
    }
    selected = max(candidates, key=lambda key: candidates[key]["val"])
    return {
        "y_true": y_test,
        "selected_strategy": selected,
        "selected_epoch": candidates[selected]["epoch"],
        "selected_pred": candidates[selected]["y_pred"],
        "params_only_pred": params_test_pred,
        "params_adam_pred": adam_test_pred,
    }


def weighted_metric(result_df, col, weight_col="测试样本数"):
    return float(np.average(result_df[col], weights=result_df[weight_col]))


def print_province_line(row):
    table = pd.DataFrame([
        {
            "方法": "SNN_fixed_linear",
            "strict_precision": row["SNN_fixed_linear_strict_precision"],
            "tolerant_precision": row["SNN_fixed_linear_tolerant_precision"],
        },
        {
            "方法": "SNN_fixed_饱和",
            "strict_precision": row["SNN_fixed_饱和_strict_precision"],
            "tolerant_precision": row["SNN_fixed_饱和_tolerant_precision"],
        },
        {
            "方法": "SNN_fixed_linear_在线更新",
            "strict_precision": row["SNN_fixed_linear_在线更新_strict_precision"],
            "tolerant_precision": row["SNN_fixed_linear_在线更新_tolerant_precision"],
        },
        {
            "方法": "SNN_fixed_饱和_在线更新",
            "strict_precision": row["SNN_fixed_饱和_在线更新_strict_precision"],
            "tolerant_precision": row["SNN_fixed_饱和_在线更新_tolerant_precision"],
        },
        {
            "方法": f"离线微调_验证选择策略({row['离线微调_验证选择策略']})",
            "strict_precision": row["离线微调_选择策略_strict_precision"],
            "tolerant_precision": row["离线微调_选择策略_tolerant_precision"],
        },
    ])
    print(f"{row['省份']} | 测试样本数={row['测试样本数']}")
    print(table.to_string(index=False, formatters={
        "strict_precision": "{:.6f}".format,
        "tolerant_precision": "{:.6f}".format,
    }))


def run(args):
    set_seed(args.seed)
    set_random_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    table = pd.read_excel(args.data)
    table, snn_feature_cols, dropped = clean_table_for_both_methods(table, args.province_col)
    subtrain_df, val_df, support_df, test_df, split_df = split_by_province(
        table,
        province_col=args.province_col,
        time_col=args.time_col,
        test_ratio=args.test_ratio,
        subtrain_ratio=args.subtrain_ratio,
        min_samples=args.min_samples,
    )
    participating = split_df.loc[split_df["是否参与评估"], "省份"].tolist()
    if not participating:
        raise ValueError("没有省份满足最小 sub-train / validation / test 样本数要求。")

    print(f"数据文件: {args.data}")
    print(f"清洗后样本数: {len(table)}; 因缺失删除: {dropped}")
    print(f"省份内切分: sub-train={len(subtrain_df)}, validation={len(val_df)}, test={len(test_df)}")
    print(f"参与评估省份数: {len(participating)}")
    print("SNN fixed 测试阶段同时输出：不更新，以及先预测后用当前标签更新。")

    snn_model, snn_scaler = train_snn_model(subtrain_df, val_df, snn_feature_cols, args)

    prior_features, base_features, posterior_features, nn_features = get_feature_lists(table)
    train_all = feature_extract(subtrain_df, prior_features, base_features, posterior_features, nn_features)
    base_model = build_model(prior_features, posterior_features, nn_features)
    base_model_state, base_optimizer_state = train_base_model(
        model=base_model,
        train_data=train_all,
        epochs=args.base_epochs,
        batch_size=args.train_batch_size,
        lr=args.lr,
        alpha_penalty=args.alpha_penalty,
    )

    rows = []
    prediction_rows = []
    for idx, province in enumerate(participating, start=1):
        print(f"\n[{idx}/{len(participating)}] 处理省份: {province}")
        train_p = subtrain_df[subtrain_df[args.province_col] == province].copy()
        val_p = val_df[val_df[args.province_col] == province].copy()
        support_p = support_df[support_df[args.province_col] == province].copy()
        test_p = test_df[test_df[args.province_col] == province].copy()

        snn_result = evaluate_snn_fixed_for_province(
            snn_model,
            snn_scaler,
            support_p,
            test_p,
            snn_feature_cols,
            update_on_test=False,
        )
        snn_online_result = evaluate_snn_fixed_for_province(
            snn_model,
            snn_scaler,
            support_p,
            test_p,
            snn_feature_cols,
            update_on_test=True,
        )
        offline_result = evaluate_offline_for_province(
            base_model_state,
            base_optimizer_state,
            train_p,
            val_p,
            test_p,
            prior_features,
            base_features,
            posterior_features,
            nn_features,
            args,
        )

        y_true = offline_result["y_true"]
        snn_linear_metrics = metric_pair(snn_result["linear"]["y_true"], snn_result["linear"]["y_pred"])
        snn_normal2_metrics = metric_pair(snn_result["normal2"]["y_true"], snn_result["normal2"]["y_pred"])
        snn_online_linear_metrics = metric_pair(
            snn_online_result["linear"]["y_true"],
            snn_online_result["linear"]["y_pred"],
        )
        snn_online_normal2_metrics = metric_pair(
            snn_online_result["normal2"]["y_true"],
            snn_online_result["normal2"]["y_pred"],
        )
        offline_selected_metrics = metric_pair(y_true, offline_result["selected_pred"])
        offline_params_metrics = metric_pair(y_true, offline_result["params_only_pred"])
        offline_adam_metrics = metric_pair(y_true, offline_result["params_adam_pred"])

        row = {
            "省份": province,
            "sub_train样本数": len(train_p),
            "验证样本数": len(val_p),
            "测试样本数": len(test_p),
            "SNN_fixed_linear_strict_precision": snn_linear_metrics["strict_precision"],
            "SNN_fixed_linear_tolerant_precision": snn_linear_metrics["tolerant_precision"],
            "SNN_fixed_饱和_strict_precision": snn_normal2_metrics["strict_precision"],
            "SNN_fixed_饱和_tolerant_precision": snn_normal2_metrics["tolerant_precision"],
            "SNN_fixed_linear_在线更新_strict_precision": snn_online_linear_metrics["strict_precision"],
            "SNN_fixed_linear_在线更新_tolerant_precision": snn_online_linear_metrics["tolerant_precision"],
            "SNN_fixed_饱和_在线更新_strict_precision": snn_online_normal2_metrics["strict_precision"],
            "SNN_fixed_饱和_在线更新_tolerant_precision": snn_online_normal2_metrics["tolerant_precision"],
            "离线微调_仅参数_strict_precision": offline_params_metrics["strict_precision"],
            "离线微调_仅参数_tolerant_precision": offline_params_metrics["tolerant_precision"],
            "离线微调_Adam状态_strict_precision": offline_adam_metrics["strict_precision"],
            "离线微调_Adam状态_tolerant_precision": offline_adam_metrics["tolerant_precision"],
            "离线微调_验证选择策略": offline_result["selected_strategy"],
            "离线微调_选择策略_best_epoch": offline_result["selected_epoch"],
            "离线微调_选择策略_strict_precision": offline_selected_metrics["strict_precision"],
            "离线微调_选择策略_tolerant_precision": offline_selected_metrics["tolerant_precision"],
            "normal2_strict_减_离线选择": (
                snn_normal2_metrics["strict_precision"] - offline_selected_metrics["strict_precision"]
            ),
            "normal2_tolerant_减_离线选择": (
                snn_normal2_metrics["tolerant_precision"] - offline_selected_metrics["tolerant_precision"]
            ),
        }
        rows.append(row)
        print_province_line(row)

        case_ids = test_p[CASE_ID_COL].tolist()
        for pos, case_id in enumerate(case_ids):
            prediction_rows.append({
                "省份": province,
                "案号": case_id,
                "真实有期徒刑": float(y_true[pos]),
                "SNN_fixed_linear_pred": float(snn_result["linear"]["y_pred"][pos]),
                "SNN_fixed_饱和_pred": float(snn_result["normal2"]["y_pred"][pos]),
                "SNN_fixed_linear_在线更新_pred": float(snn_online_result["linear"]["y_pred"][pos]),
                "SNN_fixed_饱和_在线更新_pred": float(snn_online_result["normal2"]["y_pred"][pos]),
                "离线微调_选择策略": offline_result["selected_strategy"],
                "离线微调_选择策略_pred": float(offline_result["selected_pred"][pos]),
                "离线微调_仅参数_pred": float(offline_result["params_only_pred"][pos]),
                "离线微调_Adam状态_pred": float(offline_result["params_adam_pred"][pos]),
            })

    result_df = pd.DataFrame(rows)
    prediction_df = pd.DataFrame(prediction_rows)

    summary_rows = [
        {"方法": "SNN_fixed_linear", "指标": "strict_precision", "全国测试集加权精度": weighted_metric(result_df, "SNN_fixed_linear_strict_precision")},
        {"方法": "SNN_fixed_linear", "指标": "tolerant_precision", "全国测试集加权精度": weighted_metric(result_df, "SNN_fixed_linear_tolerant_precision")},
        {"方法": "SNN_fixed_饱和", "指标": "strict_precision", "全国测试集加权精度": weighted_metric(result_df, "SNN_fixed_饱和_strict_precision")},
        {"方法": "SNN_fixed_饱和", "指标": "tolerant_precision", "全国测试集加权精度": weighted_metric(result_df, "SNN_fixed_饱和_tolerant_precision")},
        {"方法": "SNN_fixed_linear_在线更新", "指标": "strict_precision", "全国测试集加权精度": weighted_metric(result_df, "SNN_fixed_linear_在线更新_strict_precision")},
        {"方法": "SNN_fixed_linear_在线更新", "指标": "tolerant_precision", "全国测试集加权精度": weighted_metric(result_df, "SNN_fixed_linear_在线更新_tolerant_precision")},
        {"方法": "SNN_fixed_饱和_在线更新", "指标": "strict_precision", "全国测试集加权精度": weighted_metric(result_df, "SNN_fixed_饱和_在线更新_strict_precision")},
        {"方法": "SNN_fixed_饱和_在线更新", "指标": "tolerant_precision", "全国测试集加权精度": weighted_metric(result_df, "SNN_fixed_饱和_在线更新_tolerant_precision")},
        {"方法": "离线微调_仅参数", "指标": "strict_precision", "全国测试集加权精度": weighted_metric(result_df, "离线微调_仅参数_strict_precision")},
        {"方法": "离线微调_仅参数", "指标": "tolerant_precision", "全国测试集加权精度": weighted_metric(result_df, "离线微调_仅参数_tolerant_precision")},
        {"方法": "离线微调_Adam状态", "指标": "strict_precision", "全国测试集加权精度": weighted_metric(result_df, "离线微调_Adam状态_strict_precision")},
        {"方法": "离线微调_Adam状态", "指标": "tolerant_precision", "全国测试集加权精度": weighted_metric(result_df, "离线微调_Adam状态_tolerant_precision")},
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_table = summary_df.pivot(
        index="方法",
        columns="指标",
        values="全国测试集加权精度",
    ).reset_index()

    result_path = os.path.join(args.output_dir, "fair_compare_province_results.xlsx")
    with pd.ExcelWriter(result_path) as writer:
        summary_df.to_excel(writer, sheet_name="national_summary", index=False)
        result_df.to_excel(writer, sheet_name="province_results", index=False)
        split_df.to_excel(writer, sheet_name="split", index=False)
        prediction_df.to_excel(writer, sheet_name="test_predictions", index=False)

    print("\n全国测试集精度对比（按测试样本数加权）")
    print("-" * 80)
    print(summary_table.to_string(index=False, formatters={
        "strict_precision": "{:.6f}".format,
        "tolerant_precision": "{:.6f}".format,
    }))
    print("-" * 80)
    print(
        "SNN_fixed_饱和 - 离线微调_验证选择策略 | strict="
        f"{weighted_metric(result_df, 'normal2_strict_减_离线选择'):+.6f}, "
        "tolerant="
        f"{weighted_metric(result_df, 'normal2_tolerant_减_离线选择'):+.6f}"
    )
    print(f"\n结果已保存: {result_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="公平比较 SNN+fixed 与分省份离线微调")
    parser.add_argument(
        "--data",
        default=str(SCRIPT_DIR / "merged_data_filtered_minor_0919_6_qwen_all.xlsx"),
        help="输入 Excel 数据文件。若要复现实验二脚本默认数据，可传入 ../故意伤害罪上公平对比SNN和分省份微调/merged_data_filtered_minor_0508_3.xlsx",
    )
    parser.add_argument("--province-col", default=PROVINCE_COL)
    parser.add_argument("--time-col", default=TIME_COL)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--subtrain-ratio", type=float, default=0.75, help="前80%训练部分中用于 sub-train 的比例")
    parser.add_argument("--min-samples", type=int, default=1, help="每个省份 sub-train/validation/test 的最小样本数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "outputs" / "fair_compare_snn_fixed_vs_offline_finetune"))

    parser.add_argument("--snn-hidden1", type=int, default=64)
    parser.add_argument("--snn-hidden2", type=int, default=32)
    parser.add_argument("--snn-batch-size", type=int, default=64)
    parser.add_argument("--snn-lr", type=float, default=0.001)
    parser.add_argument("--snn-epochs", type=int, default=200)
    parser.add_argument(
        "--snn-fixed-update-on-test",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    parser.add_argument("--base-epochs", type=int, default=100)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--train-batch-size", type=int, default=245)
    parser.add_argument("--finetune-batch-size", type=int, default=245)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--alpha-penalty", type=float, default=1.4)
    parser.add_argument("--print-every", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
