# -*- coding: utf-8 -*-
"""
分省份离线微调实验。

与 province_three_strategy_experiment.py 的在线更新不同，本脚本验证：
每个省份从全国整体模型出发，使用本省训练集多 epoch 微调，
再用本省验证集选择最佳策略 / 最佳 epoch，最后在本省测试集评估。

默认数据划分与 three_strategy_experiment.py 保持一致：
全量数据按原始顺序整体切 Train / Validation / Test = 60% / 20% / 20%。
"""

import argparse
import copy
import os

import numpy as np
import pandas as pd
import torch

from three_strategy_experiment import (
    RelativeAbsoluteErrorLoss,
    batch_indices,
    build_model,
    evaluate_fixed,
    feature_extract,
    get_feature_lists,
    predict_raw,
    set_seed,
    train_base_model,
)


STRATEGY_FIXED = "策略一_固定参数"
STRATEGY_PARAMS_ONLY = "策略二_仅参数离线微调"
STRATEGY_PARAMS_ADAM = "策略三_参数加Adam状态离线微调"


def split_global(table, province_col, train_ratio=0.6, val_ratio=0.2, min_samples=1):
    total = len(table)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    train_df = table.iloc[:train_end].copy().reset_index(drop=True)
    val_df = table.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = table.iloc[val_end:].copy().reset_index(drop=True)

    provinces = sorted(
        set(train_df[province_col].dropna())
        & set(val_df[province_col].dropna())
        & set(test_df[province_col].dropna())
    )
    rows = []
    for province in provinces:
        train_n = int((train_df[province_col] == province).sum())
        val_n = int((val_df[province_col] == province).sum())
        test_n = int((test_df[province_col] == province).sum())
        rows.append({
            "省份": province,
            "总样本数": int((table[province_col] == province).sum()),
            "训练样本数": train_n,
            "验证样本数": val_n,
            "测试样本数": test_n,
            "是否参与分省份评估": train_n >= min_samples and val_n >= min_samples and test_n >= min_samples,
        })
    return train_df, val_df, test_df, pd.DataFrame(rows)


def split_each_province(table, province_col, time_col, train_ratio=0.6, val_ratio=0.2, min_samples=10):
    train_parts = []
    val_parts = []
    test_parts = []
    rows = []

    for province, group in table.groupby(province_col, sort=False):
        group = group.copy()
        if time_col in group.columns:
            parsed_time = pd.to_datetime(group[time_col], errors="coerce")
            group = (
                group.assign(_parsed_time=parsed_time)
                .sort_values("_parsed_time", kind="mergesort")
                .drop(columns=["_parsed_time"])
            )

        n = len(group)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        ok = n >= min_samples and train_end >= 1 and val_end > train_end and val_end < n
        if not ok:
            train_parts.append(group)
            rows.append({
                "省份": province,
                "总样本数": n,
                "训练样本数": n,
                "验证样本数": 0,
                "测试样本数": 0,
                "是否参与分省份评估": False,
            })
            continue

        train_p = group.iloc[:train_end]
        val_p = group.iloc[train_end:val_end]
        test_p = group.iloc[val_end:]
        train_parts.append(train_p)
        val_parts.append(val_p)
        test_parts.append(test_p)
        rows.append({
            "省份": province,
            "总样本数": n,
            "训练样本数": len(train_p),
            "验证样本数": len(val_p),
            "测试样本数": len(test_p),
            "是否参与分省份评估": True,
        })

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame(columns=table.columns)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=table.columns)
    return train_df, val_df, test_df, pd.DataFrame(rows)


def train_one_epoch(model, optimizer, train_data, batch_size, alpha_penalty):
    criterion = RelativeAbsoluteErrorLoss()
    model.train()
    for indices in batch_indices(len(train_data), batch_size, shuffle=False):
        batch = train_data.batch(indices)
        optimizer.zero_grad()
        output, nn_output, _ = predict_raw(model, batch)
        loss = criterion(output, batch.y) + alpha_penalty * torch.abs(torch.mean(nn_output) + 0.0370)
        loss.backward()
        optimizer.step()


def offline_finetune_strategy(
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

    best_val_precision = evaluate_fixed(model, val_data)
    best_epoch = 0
    best_model_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, finetune_epochs + 1):
        train_one_epoch(model, optimizer, train_data, batch_size, alpha_penalty)
        val_precision = evaluate_fixed(model, val_data)
        if val_precision > best_val_precision:
            best_val_precision = val_precision
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())

    best_model = build_model(prior_features, posterior_features, nn_features)
    best_model.load_state_dict(best_model_state)
    test_precision = evaluate_fixed(best_model, test_data)
    return best_val_precision, test_precision, best_epoch


def delta_status(delta, eps=1e-12):
    if delta > eps:
        return "上升"
    if delta < -eps:
        return "下降"
    return "持平"


def run_experiment(args):
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
        split_desc = "全量数据整体 6:2:2 划分"
    else:
        train_df, val_df, test_df, split_df = split_each_province(
            table,
            province_col=args.province_col,
            time_col=args.time_col,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            min_samples=args.min_samples,
        )
        split_desc = "每个省份内部 6:2:2 划分"

    prior_features, base_features, posterior_features, nn_features = get_feature_lists(table)
    train_all = feature_extract(train_df, prior_features, base_features, posterior_features, nn_features)

    print(f"数据文件: {args.data}")
    print(f"划分方式: {split_desc}")
    print(f"整体训练: train={len(train_df)}, validation={len(val_df)}, test={len(test_df)}")
    print(f"参与分省份评估省份数: {int(split_df['是否参与分省份评估'].sum())}")
    print(f"全国模型训练 epoch: {args.base_epochs}; 分省份微调 epoch: {args.finetune_epochs}")

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
    participating = split_df.loc[split_df["是否参与分省份评估"], "省份"].tolist()
    for idx, province in enumerate(participating, start=1):
        print(f"\n[{idx}/{len(participating)}] 处理省份: {province}")
        train_p = train_df[train_df[args.province_col] == province]
        val_p = val_df[val_df[args.province_col] == province]
        test_p = test_df[test_df[args.province_col] == province]

        train_data = feature_extract(train_p, prior_features, base_features, posterior_features, nn_features)
        val_data = feature_extract(val_p, prior_features, base_features, posterior_features, nn_features)
        test_data = feature_extract(test_p, prior_features, base_features, posterior_features, nn_features)

        fixed_model = build_model(prior_features, posterior_features, nn_features)
        fixed_model.load_state_dict(copy.deepcopy(base_model_state))
        fixed_val = evaluate_fixed(fixed_model, val_data)
        fixed_test = evaluate_fixed(fixed_model, test_data)

        params_val, params_test, params_epoch = offline_finetune_strategy(
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

        adam_val, adam_test, adam_epoch = offline_finetune_strategy(
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

        val_results = {
            STRATEGY_FIXED: fixed_val,
            STRATEGY_PARAMS_ONLY: params_val,
            STRATEGY_PARAMS_ADAM: adam_val,
        }
        test_results = {
            STRATEGY_FIXED: fixed_test,
            STRATEGY_PARAMS_ONLY: params_test,
            STRATEGY_PARAMS_ADAM: adam_test,
        }
        best_val_strategy = max(val_results, key=val_results.get)
        selected_test = test_results[best_val_strategy]
        delta = selected_test - fixed_test
        best_test_strategy = max(test_results, key=test_results.get)

        split_info = split_df[split_df["省份"] == province].iloc[0].to_dict()
        rows.append({
            "省份": province,
            "总样本数": split_info["总样本数"],
            "训练样本数": split_info["训练样本数"],
            "验证样本数": split_info["验证样本数"],
            "测试样本数": split_info["测试样本数"],
            "策略一_固定参数_验证精度": fixed_val,
            "策略二_仅参数离线微调_验证精度": params_val,
            "策略三_参数加Adam状态离线微调_验证精度": adam_val,
            "策略二_最佳epoch": params_epoch,
            "策略三_最佳epoch": adam_epoch,
            "验证集最佳策略": best_val_strategy,
            "验证集最佳精度": val_results[best_val_strategy],
            "策略一_固定参数_测试精度": fixed_test,
            "策略二_仅参数离线微调_测试精度": params_test,
            "策略三_参数加Adam状态离线微调_测试精度": adam_test,
            "验证最佳策略对应测试精度": selected_test,
            "相对固定参数提升": delta,
            "是否提升": delta_status(delta),
            "测试集实际最佳策略": best_test_strategy,
            "测试集实际最佳精度": test_results[best_test_strategy],
        })

        print(
            f"验证最佳={best_val_strategy}; "
            f"固定测试={fixed_test:.6f}; "
            f"选中测试={selected_test:.6f}; "
            f"delta={delta:+.6f}; "
            f"策略二epoch={params_epoch}; 策略三epoch={adam_epoch}"
        )

    result_df = pd.DataFrame(rows)
    weighted_fixed = float(np.average(result_df["策略一_固定参数_测试精度"], weights=result_df["测试样本数"]))
    weighted_params_only = float(np.average(result_df["策略二_仅参数离线微调_测试精度"], weights=result_df["测试样本数"]))
    weighted_params_adam = float(np.average(result_df["策略三_参数加Adam状态离线微调_测试精度"], weights=result_df["测试样本数"]))
    weighted_selected = float(np.average(result_df["验证最佳策略对应测试精度"], weights=result_df["测试样本数"]))
    delta_params_only = weighted_params_only - weighted_fixed
    delta_params_adam = weighted_params_adam - weighted_fixed
    weighted_delta = weighted_selected - weighted_fixed

    summary_df = pd.DataFrame([
        {"指标": "划分方式", "值": split_desc},
        {"指标": "策略一_固定参数_加权测试精度", "值": weighted_fixed},
        {"指标": "策略二_仅参数离线微调_加权测试精度", "值": weighted_params_only},
        {"指标": "策略二_相对策略一提升", "值": delta_params_only},
        {"指标": "策略三_参数加Adam状态离线微调_加权测试精度", "值": weighted_params_adam},
        {"指标": "策略三_相对策略一提升", "值": delta_params_adam},
        {"指标": "分省份离线微调选策略后加权测试精度", "值": weighted_selected},
        {"指标": "整体加权提升", "值": weighted_delta},
        {"指标": "上升省份数", "值": int((result_df["是否提升"] == "上升").sum())},
        {"指标": "下降省份数", "值": int((result_df["是否提升"] == "下降").sum())},
        {"指标": "持平省份数", "值": int((result_df["是否提升"] == "持平").sum())},
        {"指标": "验证集策略一胜出省份数", "值": int((result_df["验证集最佳策略"] == STRATEGY_FIXED).sum())},
        {"指标": "验证集策略二胜出省份数", "值": int((result_df["验证集最佳策略"] == STRATEGY_PARAMS_ONLY).sum())},
        {"指标": "验证集策略三胜出省份数", "值": int((result_df["验证集最佳策略"] == STRATEGY_PARAMS_ADAM).sum())},
    ])

    result_path = os.path.join(args.output_dir, "province_offline_finetune_results.xlsx")
    summary_path = os.path.join(args.output_dir, "province_offline_finetune_summary.xlsx")
    model_path = os.path.join(args.output_dir, "base_model_and_adam_state.pth")

    result_df.to_excel(result_path, index=False)
    with pd.ExcelWriter(summary_path) as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        split_df.to_excel(writer, sheet_name="split", index=False)
        result_df.to_excel(writer, sheet_name="province_results", index=False)

    torch.save({
        "model_state_dict": copy.deepcopy(base_model_state),
        "optimizer_state_dict": copy.deepcopy(base_optimizer_state),
        "feature_lists": {
            "prior_features": prior_features,
            "base_features": base_features,
            "posterior_features": posterior_features,
            "nn_features": nn_features,
        },
        "args": vars(args),
    }, model_path)

    print("\n分省份离线微调实验汇总")
    print("-" * 64)
    print("三个策略最终测试集加权精度与相对策略一提升:")
    print(f"{STRATEGY_FIXED}: 精度={weighted_fixed:.6f}, 提升={0.0:+.6f}")
    print(f"{STRATEGY_PARAMS_ONLY}: 精度={weighted_params_only:.6f}, 提升={delta_params_only:+.6f}")
    print(f"{STRATEGY_PARAMS_ADAM}: 精度={weighted_params_adam:.6f}, 提升={delta_params_adam:+.6f}")
    print("-" * 64)
    print(f"验证集选出的最终策略组合: 精度={weighted_selected:.6f}, 相对策略一提升={weighted_delta:+.6f}")
    print(f"上升省份数: {int((result_df['是否提升'] == '上升').sum())}")
    print(f"下降省份数: {int((result_df['是否提升'] == '下降').sum())}")
    print(f"持平省份数: {int((result_df['是否提升'] == '持平').sum())}")
    print("\n验证集最佳策略胜出次数:")
    print(result_df["验证集最佳策略"].value_counts().to_string())
    print("\n相对固定参数提升 Top 5:")
    print(
        result_df.sort_values("相对固定参数提升", ascending=False)
        [["省份", "测试样本数", "验证集最佳策略", "相对固定参数提升"]]
        .head(5)
        .to_string(index=False)
    )
    print("\n相对固定参数下降 Top 5:")
    print(
        result_df.sort_values("相对固定参数提升", ascending=True)
        [["省份", "测试样本数", "验证集最佳策略", "相对固定参数提升"]]
        .head(5)
        .to_string(index=False)
    )
    print(f"\n省份结果: {result_path}")
    print(f"汇总结果: {summary_path}")
    print(f"基础模型和 Adam 状态: {model_path}")


def main():
    parser = argparse.ArgumentParser(description="分省份离线多 epoch 微调实验")
    parser.add_argument("--data", default="merged_data_filtered_minor_0508_3.xlsx", help="输入 Excel 数据文件")
    parser.add_argument("--province-col", default="省份/直辖市", help="省份列名")
    parser.add_argument("--time-col", default="判决时间", help="split-mode=province 时用于省份内部排序")
    parser.add_argument("--split-mode", choices=["global", "province"], default="global", help="global 与 three_strategy_experiment.py 一致")
    parser.add_argument("--train-ratio", type=float, default=0.6, help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--min-samples", type=int, default=1, help="每个省份参与评估所需最小 train/val/test 样本数")
    parser.add_argument("--base-epochs", type=int, default=30, help="全国基础模型训练 epoch")
    parser.add_argument("--finetune-epochs", type=int, default=30, help="每个省份离线微调 epoch")
    parser.add_argument("--train-batch-size", type=int, default=245, help="全国训练 batch size")
    parser.add_argument("--finetune-batch-size", type=int, default=245, help="省份微调 batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--alpha-penalty", type=float, default=1.4, help="NN 输出均值惩罚系数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output-dir", default=os.path.join("outputs", "province_offline_finetune_experiment"), help="输出目录")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
