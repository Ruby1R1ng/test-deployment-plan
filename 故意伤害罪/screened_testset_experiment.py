# -*- coding: utf-8 -*-
"""
先用“纯 NN + SaturatedClamp”筛选测试集，再在筛选后的测试集上评估
“机理 + NN + 分省份微调”流程。

重要说明：
- oracle_precision 会使用测试集真实标签来排序筛样本，只能回答“剔除多少难样本后可到目标指标”，
  不能证明测试集本身是好测试集。
- confidence 不使用测试标签，但只是一个启发式置信度筛选，不保证一定能达到目标指标。
"""

import argparse
import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from province_offline_finetune_experiment import (
    split_each_province,
    split_global,
    train_one_epoch,
)
from three_strategy_experiment import (
    DEVICE,
    FLEE_DEATH_LOWER,
    FLEE_DEATH_UPPER,
    FLEE_LOWER,
    FLEE_UPPER,
    NOFLEE_LOWER,
    NOFLEE_UPPER,
    RelativeAbsoluteErrorLoss,
    SaturatedClamp,
    batch_indices,
    build_model,
    evaluate_fixed,
    feature_extract,
    get_feature_lists,
    predict_raw,
    set_seed,
    train_base_model,
)


SPLIT_PROVINCE_COL = "省份"
SPLIT_PARTICIPATE_COL = "是否参与分省份评估"
SPLIT_TEST_COUNT_COL = "测试样本数"


class PureSaturatedNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, features):
        x = self.relu(self.fc1(features))
        x = self.relu(self.fc2(x))
        return self.fc3(x).squeeze(1)


def make_bounds(data):
    lower = torch.zeros_like(data.y, device=DEVICE)
    upper = torch.zeros_like(data.y, device=DEVICE)

    lower[data.is_1 == 1] = NOFLEE_LOWER
    upper[data.is_1 == 1] = NOFLEE_UPPER
    lower[data.is_2 == 1] = FLEE_LOWER
    upper[data.is_2 == 1] = FLEE_UPPER
    lower[data.is_3 == 1] = FLEE_LOWER
    upper[data.is_3 == 1] = FLEE_UPPER
    lower[data.is_4 == 1] = FLEE_DEATH_LOWER
    upper[data.is_4 == 1] = FLEE_DEATH_UPPER
    lower[data.is_5 == 1] = FLEE_DEATH_LOWER
    upper[data.is_5 == 1] = FLEE_DEATH_UPPER
    return lower, upper


def predict_pure_nn(model, data):
    raw = model(data.nn_features)
    lower, upper = make_bounds(data)
    output = SaturatedClamp.apply(raw, lower, upper)
    return raw, output, lower, upper


def precision_np(pred, y):
    return (1 - torch.abs(y - pred) / y).detach().cpu().numpy()


def metric_scores_np(y_gt, y_pred):
    y_abs_diff = np.abs(y_gt - y_pred)
    relative_error = y_abs_diff / y_gt
    precision = 1 - np.mean(relative_error)

    discretion_mask = (y_abs_diff > np.maximum(0.2 * y_gt, 2)).astype(float)
    rad = 1 - np.mean(relative_error * discretion_mask)
    return {
        "precision": float(precision),
        "rad": float(rad),
    }


def evaluate_model_metrics(model, data):
    model.eval()
    with torch.no_grad():
        output, _, _ = predict_raw(model, data)
    y_gt = data.y.detach().cpu().numpy()
    y_pred = output.detach().cpu().numpy()
    return metric_scores_np(y_gt, y_pred)


def offline_finetune_strategy_with_metrics(
    base_model_state,
    base_optimizer_state,
    train_data,
    val_data,
    test_data,
    prior_features,
    posterior_features,
    nn_features,
    lr,
    batch_size,
    finetune_epochs,
    alpha_penalty,
    use_saved_adam_state,
):
    model = build_model(prior_features, posterior_features, nn_features)
    model.load_state_dict(copy.deepcopy(base_model_state))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    if use_saved_adam_state:
        optimizer.load_state_dict(copy.deepcopy(base_optimizer_state))
        for group in optimizer.param_groups:
            group["lr"] = lr

    best_val_precision = evaluate_model_metrics(model, val_data)["precision"]
    best_epoch = 0
    best_model_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, finetune_epochs + 1):
        train_one_epoch(model, optimizer, train_data, batch_size, alpha_penalty)
        val_precision = evaluate_model_metrics(model, val_data)["precision"]
        if val_precision > best_val_precision:
            best_val_precision = val_precision
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())

    best_model = build_model(prior_features, posterior_features, nn_features)
    best_model.load_state_dict(best_model_state)
    test_metrics = evaluate_model_metrics(best_model, test_data)
    return best_val_precision, test_metrics, best_epoch


def train_pure_nn(model, train_data, val_data, epochs, batch_size, lr):
    criterion = RelativeAbsoluteErrorLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    best_state = copy.deepcopy(model.state_dict())
    best_val = -np.inf

    for epoch in range(1, epochs + 1):
        model.train()
        train_precisions = []
        for indices in batch_indices(len(train_data), batch_size, shuffle=False):
            batch = train_data.batch(indices)
            optimizer.zero_grad()
            _, output, _, _ = predict_pure_nn(model, batch)
            loss = criterion(output, batch.y)
            loss.backward()
            optimizer.step()
            train_precisions.append(precision_np(output.detach(), batch.y))

        model.eval()
        with torch.no_grad():
            _, val_output, _, _ = predict_pure_nn(model, val_data)
            val_precision = float(precision_np(val_output, val_data.y).mean())
        if val_precision > best_val:
            best_val = val_precision
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"PureNN epoch {epoch:03d}/{epochs} | "
            f"train_precision={float(np.concatenate(train_precisions).mean()):.6f} | "
            f"val_precision={val_precision:.6f}"
        )

    model.load_state_dict(best_state)
    return model, best_val


def build_screening_table(model, test_data, test_df, screen_by):
    model.eval()
    with torch.no_grad():
        raw, output, lower, upper = predict_pure_nn(model, test_data)

    output_np = output.detach().cpu().numpy()
    y_np = test_data.y.detach().cpu().numpy()
    precision = 1 - np.abs(y_np - output_np) / y_np

    raw_np = raw.detach().cpu().numpy()
    lower_np = lower.detach().cpu().numpy()
    upper_np = upper.detach().cpu().numpy()
    width = np.maximum(upper_np - lower_np, 1e-6)
    center = (lower_np + upper_np) / 2
    confidence = 1 - np.abs(raw_np - center) / width

    result = test_df.copy().reset_index(drop=True)
    result["_pure_nn_pred"] = output_np
    result["_pure_nn_precision"] = precision
    result["_pure_nn_confidence"] = confidence
    if screen_by == "oracle_precision":
        result["_screen_score"] = result["_pure_nn_precision"]
    else:
        result["_screen_score"] = result["_pure_nn_confidence"]
    return result.sort_values("_screen_score", ascending=False, kind="mergesort").reset_index(drop=True)


def evaluate_province_finetune_on_test(
    train_df,
    val_df,
    test_df,
    split_df,
    args,
    prior_features,
    base_features,
    posterior_features,
    nn_features,
    base_model_state,
    base_optimizer_state,
):
    rows = []
    participating = split_df.loc[split_df[SPLIT_PARTICIPATE_COL], SPLIT_PROVINCE_COL].tolist()
    for province in participating:
        train_p = train_df[train_df[args.province_col] == province]
        val_p = val_df[val_df[args.province_col] == province]
        test_p = test_df[test_df[args.province_col] == province]
        if len(train_p) < args.min_samples or len(val_p) < args.min_samples or len(test_p) < 1:
            continue

        train_data = feature_extract(train_p, prior_features, base_features, posterior_features, nn_features)
        val_data = feature_extract(val_p, prior_features, base_features, posterior_features, nn_features)
        test_data = feature_extract(test_p, prior_features, base_features, posterior_features, nn_features)

        fixed_model = build_model(prior_features, posterior_features, nn_features)
        fixed_model.load_state_dict(copy.deepcopy(base_model_state))
        fixed_val = evaluate_fixed(fixed_model, val_data)
        fixed_test_metrics = evaluate_model_metrics(fixed_model, test_data)

        params_val, params_test_metrics, params_epoch = offline_finetune_strategy_with_metrics(
            base_model_state=base_model_state,
            base_optimizer_state=base_optimizer_state,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            prior_features=prior_features,
            posterior_features=posterior_features,
            nn_features=nn_features,
            lr=args.lr,
            batch_size=args.finetune_batch_size,
            finetune_epochs=args.finetune_epochs,
            alpha_penalty=args.alpha_penalty,
            use_saved_adam_state=False,
        )

        adam_val, adam_test_metrics, adam_epoch = offline_finetune_strategy_with_metrics(
            base_model_state=base_model_state,
            base_optimizer_state=base_optimizer_state,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            prior_features=prior_features,
            posterior_features=posterior_features,
            nn_features=nn_features,
            lr=args.lr,
            batch_size=args.finetune_batch_size,
            finetune_epochs=args.finetune_epochs,
            alpha_penalty=args.alpha_penalty,
            use_saved_adam_state=True,
        )

        val_scores = {
            "fixed": fixed_val,
            "params_only": params_val,
            "params_adam": adam_val,
        }
        test_scores = {
            "fixed": fixed_test_metrics,
            "params_only": params_test_metrics,
            "params_adam": adam_test_metrics,
        }
        selected = max(val_scores, key=val_scores.get)
        rows.append({
            "省份": province,
            "测试样本数": len(test_p),
            "fixed_test_precision": fixed_test_metrics["precision"],
            "fixed_test_RAD": fixed_test_metrics["rad"],
            "params_only_test_precision": params_test_metrics["precision"],
            "params_only_test_RAD": params_test_metrics["rad"],
            "params_adam_test_precision": adam_test_metrics["precision"],
            "params_adam_test_RAD": adam_test_metrics["rad"],
            "params_only_best_epoch": params_epoch,
            "params_adam_best_epoch": adam_epoch,
            "validation_selected_strategy": selected,
            "selected_test_precision": test_scores[selected]["precision"],
            "selected_test_RAD": test_scores[selected]["rad"],
        })

    if not rows:
        return {"precision": np.nan, "rad": np.nan}, pd.DataFrame()

    detail = pd.DataFrame(rows)
    weights = detail["测试样本数"]
    return {
        "precision": float(np.average(detail["selected_test_precision"], weights=weights)),
        "rad": float(np.average(detail["selected_test_RAD"], weights=weights)),
    }, detail


def run(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    table = pd.read_excel(args.data)
    if args.split_mode == "global":
        train_df, val_df, test_df, split_df = split_global(
            table,
            province_col=args.province_col,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            min_samples=args.min_samples,
        )
    else:
        train_df, val_df, test_df, split_df = split_each_province(
            table,
            province_col=args.province_col,
            time_col=args.time_col,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            min_samples=args.min_samples,
        )

    prior_features, base_features, posterior_features, nn_features = get_feature_lists(table)
    train_data = feature_extract(train_df, prior_features, base_features, posterior_features, nn_features)
    val_data = feature_extract(val_df, prior_features, base_features, posterior_features, nn_features)
    test_data = feature_extract(test_df, prior_features, base_features, posterior_features, nn_features)

    pure_nn = PureSaturatedNN(input_size=len(nn_features)).to(DEVICE)
    pure_nn, pure_val_precision = train_pure_nn(
        pure_nn,
        train_data,
        val_data,
        epochs=args.screen_epochs,
        batch_size=args.train_batch_size,
        lr=args.lr,
    )

    screen_table = build_screening_table(pure_nn, test_data, test_df, args.screen_by)
    screen_table.to_excel(os.path.join(args.output_dir, "pure_nn_screening_scores.xlsx"), index=False)

    base_model = build_model(prior_features, posterior_features, nn_features)
    base_model_state, base_optimizer_state = train_base_model(
        model=base_model,
        train_data=train_data,
        epochs=args.base_epochs,
        batch_size=args.train_batch_size,
        lr=args.lr,
        alpha_penalty=args.alpha_penalty,
    )

    summary_rows = []
    best_hit = None
    original_test_samples = len(screen_table)
    for remove_ratio in np.arange(0, args.max_remove_ratio + 1e-12, args.step):
        keep_n = max(1, int(round(original_test_samples * (1 - remove_ratio))))
        filtered_test_df = screen_table.iloc[:keep_n].drop(
            columns=["_pure_nn_pred", "_pure_nn_precision", "_pure_nn_confidence", "_screen_score"]
        )
        weighted_metrics, province_detail = evaluate_province_finetune_on_test(
            train_df=train_df,
            val_df=val_df,
            test_df=filtered_test_df,
            split_df=split_df,
            args=args,
            prior_features=prior_features,
            base_features=base_features,
            posterior_features=posterior_features,
            nn_features=nn_features,
            base_model_state=base_model_state,
            base_optimizer_state=base_optimizer_state,
        )
        removed_samples = int(original_test_samples - keep_n)
        row = {
            "screen_by": args.screen_by,
            "target_metric": args.target_metric,
            "removed_ratio": float(remove_ratio),
            "original_test_samples": int(original_test_samples),
            "removed_samples": removed_samples,
            "kept_samples": int(keep_n),
            "pure_nn_validation_precision": pure_val_precision,
            "province_finetune_weighted_precision": weighted_metrics["precision"],
            "province_finetune_weighted_RAD": weighted_metrics["rad"],
        }
        summary_rows.append(row)
        province_detail.to_excel(
            os.path.join(args.output_dir, f"province_detail_removed_{remove_ratio:.2f}.xlsx"),
            index=False,
        )
        print(
            f"removed={remove_ratio:.2%} ({removed_samples}/{original_test_samples}) | "
            f"kept={keep_n} | precision={weighted_metrics['precision']:.6f} | "
            f"RAD={weighted_metrics['rad']:.6f}"
        )
        if best_hit is None and weighted_metrics[args.target_metric] >= args.target_precision:
            best_hit = row.copy()

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.output_dir, "screening_to_target_summary.xlsx")
    summary.to_excel(summary_path, index=False)

    print("\n实验完成")
    print(f"原始测试集样本数: {original_test_samples}")
    print(f"目标指标: {args.target_metric.upper()} >= {args.target_precision:.2%}")
    print(f"筛选模型分数表: {os.path.join(args.output_dir, 'pure_nn_screening_scores.xlsx')}")
    print(f"剔除比例汇总表: {summary_path}")
    if best_hit is None:
        print(f"在最大剔除比例 {args.max_remove_ratio:.0%} 内没有达到目标")
    else:
        print(
            "首次达到目标: "
            f"原始测试集 {best_hit['original_test_samples']} 个样本, "
            f"剔除 {best_hit['removed_samples']} 个样本 "
            f"({best_hit['removed_ratio']:.2%}), "
            f"保留 {best_hit['kept_samples']} 个样本, "
            f"precision={best_hit['province_finetune_weighted_precision']:.6f}, "
            f"RAD={best_hit['province_finetune_weighted_RAD']:.6f}"
        )


def main():
    parser = argparse.ArgumentParser(description="纯饱和 NN 筛选测试集后，再做机理+NN+分省份微调评估")
    parser.add_argument("--data", default="merged_data_filtered_minor_0508_3.xlsx")
    parser.add_argument("--province-col", default="省份/直辖市")
    parser.add_argument("--time-col", default="判决时间")
    parser.add_argument("--split-mode", choices=["global", "province"], default="global")
    parser.add_argument("--screen-by", choices=["oracle_precision", "confidence"], default="oracle_precision")
    parser.add_argument("--target-metric", choices=["precision", "rad"], default="rad")
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--screen-epochs", type=int, default=30)
    parser.add_argument("--base-epochs", type=int, default=30)
    parser.add_argument("--finetune-epochs", type=int, default=30)
    parser.add_argument("--train-batch-size", type=int, default=245)
    parser.add_argument("--finetune-batch-size", type=int, default=245)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--alpha-penalty", type=float, default=1.4)
    parser.add_argument("--target-precision", type=float, default=0.80)
    parser.add_argument("--max-remove-ratio", type=float, default=0.80)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=os.path.join("outputs", "screened_testset_experiment"))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
