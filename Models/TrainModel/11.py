"""
改造版训练脚本：支持 MS-DSC / Single-DSC / Standard Conv 三种对照实验
核心改动：
1. 新增 DTANET_VARIANTS 列表来定义三种对照配置
2. 在主训练循环外嵌套一层对不同变体的循环
3. 为每个变体生成独立的结果文件夹
4. 汇总三组结果到对比表格
"""

import os
import sys
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import json
from datetime import datetime

# ============================================================================
# 【核心改动】三组对照配置
# ============================================================================
# 三种 DTANet 变体：MS-DSC（原始） / Single-DSC（隔离多尺度） / Standard Conv（对照）
DTANET_VARIANTS = ["ms_dsc", "single_dsc", "standard"]

# 对应的中文描述（用于报告）
VARIANT_NAMES_CN = {
    "ms_dsc": "多尺度 DSC（MS-DSC，原始设计）",
    "single_dsc": "单尺度 DSC（Single-DSC，隔离多尺度贡献）",
    "standard": "标准卷积（Standard Conv，对照）"
}

# ============================================================================
# 全局训练配置（保持一致，以确保公平对比）
# ============================================================================
DATASET = "DB2"  # "DB2" 或 "DB4"
root_data = "/data/sEMG_DB"  # 根据实际路径修改
subject_range = range(1, 11)  # 例如 DB2 有 10 个被试

# 训练超参数（所有变体使用相同配置）
BATCH_SIZE = 32
EPOCHS = 200
LR = 0.001
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.25
EARLY_STOP_PATIENCE = 20
GRAD_CLIP_NORM = 1.0

# 损失函数与优化器选项
USE_CLASS_BALANCED_LOSS = True
CB_BETA = 0.9999
USE_ADAMW = True
LABEL_SMOOTHING = 0.1

# 后处理选项
USE_POST_PROCESSING = True

# 消融分支（通常为 None，表示不消融）
ABLATED_BRANCH = None

# ============================================================================
# 导入自定义模块
# ============================================================================
# 假设以下模块已在同一项目中定义
from PyTorchModels_improved_with_ablation import (
    ImprovedFeaAndEmg_se_PyTorch,
    ECA1D, ImprovedEMA, MultiScaleTemporal1D, ImprovedLinearResidualBlock
)
from trainandtest import (
    EMGDataset, EarlyStopping, replace_channel_attention_with_identity,
    compute_class_balanced_weights, build_ce_loss, majority_vote_probs,
    plot_and_save_curves, plot_and_save_confusion, get_threeSet
)


# ============================================================================
# 工具函数
# ============================================================================

def create_variant_results_dir(base_results_dir, variant):
    """为每个变体创建独立的结果文件夹"""
    variant_dir = os.path.join(base_results_dir, f"variant_{variant}")
    os.makedirs(variant_dir, exist_ok=True)
    return variant_dir


def train_single_variant(variant, device, base_results_dir):
    """
    训练单个 DTANet 变体

    参数：
    - variant: "ms_dsc" / "single_dsc" / "standard"
    - device: torch device
    - base_results_dir: 基础结果目录

    返回：
    - results_dict: 包含所有受试者的精确度和门控权重统计
    """
    print("\n" + "=" * 80)
    print(f"【开始实验】变体: {variant} ({VARIANT_NAMES_CN[variant]})")
    print("=" * 80)

    # 创建该变体的结果目录
    RESULTS_DIR = create_variant_results_dir(base_results_dir, variant)

    torch.manual_seed(42)
    np.random.seed(42)

    subject_accuracies = []
    all_subject_gate_weights = {}
    variant_results = {
        "variant": variant,
        "variant_cn": VARIANT_NAMES_CN[variant],
        "timestamp": datetime.now().isoformat(),
        "hyperparams": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "dropout_rate": DROPOUT_RATE,
        },
        "subject_results": {}
    }

    for subject_id in subject_range:
        print(f"\n【受试者 {subject_id}】处理中 (变体: {variant})...")
        try:
            emg_file = os.path.join(
                root_data,
                f'data/restimulus/reSegEnhance/{DATASET}_s{subject_id}SegEnhance.h5'
            )
            feature_file = os.path.join(
                root_data,
                f'data/restimulus/Fea/{DATASET}_s{subject_id}feaEnhance.h5'
            )

            if not os.path.exists(emg_file):
                print(f"❌ EMG文件未找到: {emg_file}")
                continue
            if not os.path.exists(feature_file):
                print(f"❌ 特征文件未找到: {feature_file}")
                continue

            # 加载数据
            print("📂 加载数据...")
            with h5py.File(emg_file, 'r') as f:
                emg_data = f['emg'][:]
                labels = f['label'][:]
                rep_arr = f['rep'][:]
            with h5py.File(feature_file, 'r') as f:
                feature_data = f['features'][:]

            # 划分数据集
            rep_vali = 2
            emg_train, emg_vali, emg_test, label_train, label_vali, label_test = \
                get_threeSet(emg_data, labels, rep_arr, rep_vali)
            feature_train, feature_vali, feature_test, _, _, _ = \
                get_threeSet(feature_data, labels, rep_arr, rep_vali)

            print(f"✓ EMG 形状: {emg_train.shape}")
            print(f"✓ 特征形状: {feature_train.shape}")
            print(f"✓ 类别数: {np.max(label_train) + 1}")

        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            continue

        # 构建 DataLoader
        train_dataset = EMGDataset(emg_train, feature_train, label_train)
        val_dataset = EMGDataset(emg_vali, feature_vali, label_vali)
        test_dataset = EMGDataset(emg_test, feature_test, label_test)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # 推断通道数与时间长度
        if emg_train.ndim == 3 and emg_train.shape[-1] in (12, 16):
            emg_channels = int(emg_train.shape[-1])
            time_steps = int(emg_train.shape[1])
        else:
            emg_channels = int(emg_train.shape[1])
            time_steps = int(emg_train.shape[2])

        num_classes = int(np.max(label_train) + 1)

        # ======================== 【核心改动】========================
        # 构建模型时传入 dtanet_variant 参数
        print(f"🔨 构建模型（DTANet 变体: {variant}）...")
        model = ImprovedFeaAndEmg_se_PyTorch(
            emg_channels=emg_channels,
            time_steps=time_steps,
            feature_dim=int(feature_train.shape[1]),
            num_classes=num_classes,
            dropout_rate=DROPOUT_RATE,
            dtanet_variant=variant  # 【关键参数】
        ).to(device)

        # 设置消融分支（如果需要）
        if ABLATED_BRANCH is not None:
            model.set_ablation(ABLATED_BRANCH)

        # 去除通道注意力
        replaced_list = replace_channel_attention_with_identity(model)
        if len(replaced_list) > 0:
            print(f"✓ 已移除 {len(replaced_list)} 个通道注意力模块")

        # 打印模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters()
                               if p.requires_grad)
        print(f"📊 模型参数: 总数={total_params:,} | 可训练={trainable_params:,}")

        # 损失函数与优化器
        class_weights_t = None
        if USE_CLASS_BALANCED_LOSS:
            class_weights_t, _ = compute_class_balanced_weights(
                label_train, num_classes, beta=CB_BETA
            )
            print(f"✓ 启用 Class-Balanced Loss (beta={CB_BETA})")

        criterion = build_ce_loss(
            num_classes=num_classes,
            class_weights_t=class_weights_t,
            label_smoothing=LABEL_SMOOTHING,
            device=device
        )

        if USE_ADAMW:
            optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        else:
            optimizer = Adam(model.parameters(), lr=LR)

        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                      patience=8, min_lr=1e-5, verbose=False)

        early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE, min_delta=0.001)

        # 训练循环
        print(f"🚀 开始训练（{EPOCHS} epochs，变体: {variant})...")
        best_val_acc = 0.0
        best_model_state = None
        train_losses, val_losses, train_accs, val_accs = [], [], [], []
        gate_weights_history = []

        for epoch in range(EPOCHS):
            # 训练
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

            for emg_inputs, fea_inputs, labels in tqdm(
                    train_loader,
                    desc=f"Epoch {epoch + 1}/{EPOCHS}",
                    leave=False
            ):
                emg_inputs = emg_inputs.to(device)
                fea_inputs = fea_inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs, gate_weights = model(
                    fea_inputs, emg_inputs, return_gate_weights=True
                )
                loss = criterion(outputs, labels)
                loss.backward()

                if GRAD_CLIP_NORM is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            train_acc = 100.0 * train_correct / max(1, train_total)
            train_losses.append(train_loss / max(1, len(train_loader)))
            train_accs.append(train_acc)

            # 验证
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for emg_inputs, fea_inputs, labels in val_loader:
                    emg_inputs = emg_inputs.to(device)
                    fea_inputs = fea_inputs.to(device)
                    labels = labels.to(device)
                    outputs, _ = model(fea_inputs, emg_inputs,
                                       return_gate_weights=True)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_acc = 100.0 * val_correct / max(1, val_total)
            val_losses.append(val_loss / max(1, len(val_loader)))
            val_accs.append(val_acc)

            scheduler.step(val_acc)

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch + 1}/{EPOCHS} | "
                      f"Train Loss: {train_losses[-1]:.4f} | "
                      f"Train Acc: {train_acc:.2f}% | "
                      f"Val Loss: {val_losses[-1]:.4f} | "
                      f"Val Acc: {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()

            if early_stopping(val_losses[-1], model):
                print(f"  ⏹ 早停触发！第 {epoch + 1} 轮停止训练")
                break

        # 绘制训练曲线
        curves_path = plot_and_save_curves(
            subject_id, train_losses, val_losses, train_accs, val_accs,
            gate_weights_history, RESULTS_DIR
        )

        # 测试
        print(f"🧪 测试中...")
        model.load_state_dict(best_model_state)
        model.eval()
        all_preds, all_labels, all_gate_weights = [], [], []
        logits_list = []

        with torch.no_grad():
            for emg_inputs, fea_inputs, labels in test_loader:
                emg_inputs = emg_inputs.to(device)
                fea_inputs = fea_inputs.to(device)
                labels = labels.to(device)
                outputs, gate_weights = model(
                    fea_inputs, emg_inputs, return_gate_weights=True
                )
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_gate_weights.append(gate_weights.cpu().numpy())
                logits_list.append(outputs.detach().cpu())

        all_logits = torch.cat(logits_list, dim=0)
        all_labels_np = np.array(all_labels)
        all_preds_np = np.array(all_preds)

        # 多数投票后处理
        if USE_POST_PROCESSING:
            preds_for_metrics = majority_vote_probs(all_logits, k=7)
        else:
            preds_for_metrics = all_preds_np

        test_acc = 100.0 * np.sum(preds_for_metrics == all_labels_np) / len(all_labels_np)
        subject_accuracies.append(test_acc)

        final_gate_weights = np.mean(
            np.concatenate(all_gate_weights, axis=0), axis=0
        )
        all_subject_gate_weights[subject_id] = final_gate_weights

        print(f"✓ 受试者 {subject_id} 测试准确率: {test_acc:.2f}%")
        print(f"  门控权重: 空间={final_gate_weights[0] * 100:.1f}% | "
              f"时域={final_gate_weights[1] * 100:.1f}% | "
              f"时频={final_gate_weights[2] * 100:.1f}%")

        # 混淆矩阵
        cm = confusion_matrix(all_labels_np, preds_for_metrics)
        cm_path = os.path.join(RESULTS_DIR,
                               f'confusion_matrix_subject_{subject_id}.png')
        plot_and_save_confusion(cm, subject_id, cm_path)

        # 分类报告
        report_str = classification_report(all_labels_np, preds_for_metrics,
                                           digits=2)
        report_path = os.path.join(RESULTS_DIR,
                                   f'classification_report_subject_{subject_id}.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_str)

        # 记录受试者结果
        variant_results["subject_results"][subject_id] = {
            "accuracy": float(test_acc),
            "gate_weights": {
                "space": float(final_gate_weights[0]),
                "time_domain": float(final_gate_weights[1]),
                "time_frequency": float(final_gate_weights[2])
            }
        }

    # ======================== 汇总结果 ========================
    if len(subject_accuracies) > 0:
        oa = float(np.mean(subject_accuracies))
        std = float(np.std(subject_accuracies))
        acc_max = float(np.max(subject_accuracies))
        acc_min = float(np.min(subject_accuracies))

        print("\n" + "=" * 80)
        print(f"【变体 {variant} 总体评估】")
        print("=" * 80)
        print(f"处理受试者数: {len(subject_accuracies)}")
        print(f"总体准确度 (OA): {oa:.2f}%")
        print(f"准确度标准差: {std:.2f}%")
        print(f"最高准确度: {acc_max:.2f}%")
        print(f"最低准确度: {acc_min:.2f}%")

        if len(all_subject_gate_weights) > 0:
            gate_arr = np.stack(list(all_subject_gate_weights.values()), axis=0)
            gate_mean = gate_arr.mean(axis=0)
            print(f"平均门控权重: 空间={gate_mean[0] * 100:.1f}% | "
                  f"时域={gate_mean[1] * 100:.1f}% | "
                  f"时频={gate_mean[2] * 100:.1f}%")

        # 写入汇总文件
        summary_path = os.path.join(RESULTS_DIR, "summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"=== 变体: {variant} ({VARIANT_NAMES_CN[variant]}) ===\n\n")
            f.write(f"处理受试者数: {len(subject_accuracies)}\n")
            f.write(f"总体准确度 (OA): {oa:.2f}%\n")
            f.write(f"准确度标准差: {std:.2f}%\n")
            f.write(f"最高准确度: {acc_max:.2f}%\n")
            f.write(f"最低准确度: {acc_min:.2f}%\n\n")
            f.write("各受试者准确度:\n")
            for idx, acc in enumerate(subject_accuracies):
                f.write(f"  受试者 {idx + 1}: {acc:.2f}%\n")

        # 保存 JSON 结果
        variant_results["summary"] = {
            "num_subjects": len(subject_accuracies),
            "overall_accuracy": oa,
            "std": std,
            "max_accuracy": acc_max,
            "min_accuracy": acc_min,
            "all_accuracies": subject_accuracies
        }

        json_path = os.path.join(RESULTS_DIR, "results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(variant_results, f, indent=2, ensure_ascii=False)

        print(f"✓ 结果已保存到 {RESULTS_DIR}")
        return variant_results

    return None


def main():
    """主训练函数：循环三个 DTANet 变体进行对比实验"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n📱 使用设备: {device}")
    print(f"🔧 数据集: {DATASET}")
    print(f"🎯 受试者范围: {list(subject_range)}")
    print(f"⚙️  超参数: batch_size={BATCH_SIZE}, lr={LR}, epochs={EPOCHS}")

    # 创建主结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    BASE_RESULTS_DIR = f"./results_ablation_{DATASET}_{timestamp}"
    os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

    print(f"\n📁 主结果目录: {BASE_RESULTS_DIR}\n")

    # ======================== 【核心改动】========================
    # 循环三个 DTANet 变体
    all_variant_results = {}

    for variant in DTANET_VARIANTS:
        result = train_single_variant(variant, device, BASE_RESULTS_DIR)
        if result is not None:
            all_variant_results[variant] = result

    # ======================== 生成对比总结 ========================
    print("\n" + "=" * 80)
    print("【MS-DSC 消融实验 - 对比总结】")
    print("=" * 80)

    # 对比表格
    comparison_table = []
    for variant in DTANET_VARIANTS:
        if variant in all_variant_results:
            summary = all_variant_results[variant].get("summary", {})
            oa = summary.get("overall_accuracy", -1)
            std = summary.get("std", -1)
            comparison_table.append({
                "变体": VARIANT_NAMES_CN[variant],
                "OA (%)": f"{oa:.2f}",
                "Std (%)": f"{std:.2f}",
                "Max (%)": f"{summary.get('max_accuracy', -1):.2f}",
                "Min (%)": f"{summary.get('min_accuracy', -1):.2f}"
            })

    # 打印对比表
    print("\n对比表格：")
    print("-" * 80)
    if comparison_table:
        keys = comparison_table[0].keys()
        header = " | ".join([str(k) for k in keys])
        print(header)
        print("-" * 80)
        for row in comparison_table:
            values = " | ".join([str(row[k]) for k in keys])
            print(values)

    # 计算改进幅度
    print("\n\n改进幅度分析：")
    print("-" * 80)
    if "ms_dsc" in all_variant_results and "standard" in all_variant_results:
        oa_ms = all_variant_results["ms_dsc"]["summary"]["overall_accuracy"]
        oa_std = all_variant_results["standard"]["summary"]["overall_accuracy"]
        improvement = oa_ms - oa_std
        improvement_pct = (improvement / oa_std) * 100 if oa_std > 0 else 0
        print(f"MS-DSC vs Standard Conv: {improvement:+.2f}% (相对提升 {improvement_pct:+.2f}%)")

    if "ms_dsc" in all_variant_results and "single_dsc" in all_variant_results:
        oa_ms = all_variant_results["ms_dsc"]["summary"]["overall_accuracy"]
        oa_single = all_variant_results["single_dsc"]["summary"]["overall_accuracy"]
        improvement = oa_ms - oa_single
        improvement_pct = (improvement / oa_single) * 100 if oa_single > 0 else 0
        print(f"MS-DSC vs Single-DSC: {improvement:+.2f}% (相对提升 {improvement_pct:+.2f}%)")

    # 保存对比结果为 JSON
    comparison_output = {
        "timestamp": timestamp,
        "dataset": DATASET,
        "variants": all_variant_results,
        "comparison_table": comparison_table
    }

    comparison_json_path = os.path.join(BASE_RESULTS_DIR, "comparison_results.json")
    with open(comparison_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_output, f, indent=2, ensure_ascii=False)

    # 保存对比表为 CSV
    comparison_csv_path = os.path.join(BASE_RESULTS_DIR, "comparison_table.csv")
    with open(comparison_csv_path, "w", encoding="utf-8") as f:
        if comparison_table:
            keys = comparison_table[0].keys()
            f.write(",".join(keys) + "\n")
            for row in comparison_table:
                values = [row[k] for k in keys]
                f.write(",".join(values) + "\n")

    print(f"\n✓ 对比结果已保存到:")
    print(f"  - {comparison_json_path}")
    print(f"  - {comparison_csv_path}")
    print(f"\n✓ 所有结果已保存到: {BASE_RESULTS_DIR}")


if __name__ == "__main__":
    main()
