#!/usr/bin/env python3
"""
Leave-One-Out 跨视角泛化验证实验批量运行脚本

用途：
对单帧重建做7折Leave-One-Out实验，每次用6个视角训练，在第7个held-out视角上评估渲染质量。
可选 --include_baseline 对每个帧额外跑一次"全7视角训练"作为 memorization 基线。

运行方式：
    # 仅 LOO 实验
    python tools/run_leave_one_out.py \
        --frames 100 \
        --config configs/apps/wildtrack_3dgut.yaml \
        --output_root experiments/leave_one_out \
        --n_iterations 15000

    # 包含基线对照
    python tools/run_leave_one_out.py \
        --frames 100 \
        --config configs/apps/wildtrack_3dgut.yaml \
        --output_root experiments/leave_one_out \
        --n_iterations 15000 \
        --include_baseline

输出目录结构：
    experiments/leave_one_out/
    ├── frame_100/
    │   ├── held_C1/
    │   │   └── run/
    │   │       └── Wildtrack-*/            # Hydra 输出
    │   │           ├── ours_15000/renders/  # held-out 视角渲染
    │   │           ├── ckpt_last.pt
    │   │           ├── error.log            # 失败时写入
    │   │           └── train_metrics.json   # 成功时写入
    │   ├── held_C2/
    │   └── ...
    │   └── baseline/                        # --include_baseline 时
    │       └── run/
    │           └── Wildtrack-*/
    └── ...

初始化点云说明：
    当前 wildtrack_3dgut.yaml 使用 initialization: random，即纯随机初始化，
    不依赖任何 SfM 点云。所有 fold 共享相同的随机初始化配置（num_gaussians,
    xyz_min, xyz_max），不存在"不同 fold 用不同初始化点云"的混淆变量。
    如果未来切换到 colmap 或 fused_point_cloud 初始化，需确保所有 fold
    加载同一份预生成点云。
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def _find_hydra_dir(exp_dir: Path) -> Path:
    """查找最新的 Hydra 输出子目录"""
    hydra_dirs = sorted(exp_dir.glob("run/Wildtrack-*"))
    if hydra_dirs:
        return hydra_dirs[-1]
    return exp_dir / "run"  # fallback


def _set_seed(seed: int):
    """设置所有随机种子，确保 LOO fold 间可比"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training(base_config: str, exp_dir: Path, frame_id: int,
                 held_out_camera: str, n_iterations: int,
                 dataset_path: str, seed: int = 42) -> bool:
    """
    运行训练命令（使用 hydra 命令行覆盖）

    Hydra 输出路径约定：
        out_dir=exp_dir, experiment_name=run
        → 实际输出: exp_dir/run/Wildtrack-<timestamp>/

    使用 ++ 前缀强制添加 dataset.held_out_camera 和 dataset.single_frame_id，
    因为原始 yaml 中可能没有这些字段。

    随机种子通过 ++random_seed 传递给 Hydra，确保每个 fold 的初始高斯分布
    仅因视角不同而不同，不受随机初始化差异影响。
    """
    config_path = Path(base_config).resolve()
    # Hydra config-path must be the configs/ root (not configs/apps/) so that
    # defaults like /render:3dgut can resolve render/3dgut.yaml etc.
    # config-name uses the relative path from configs/ root, e.g. apps/wildtrack_3dgut
    configs_root = config_path
    while configs_root.parent.name != "configs" and configs_root.parent != configs_root:
        configs_root = configs_root.parent
    if configs_root.parent.name == "configs":
        configs_root = configs_root.parent
    config_dir = str(configs_root)
    config_name = str(config_path.relative_to(configs_root).with_suffix(''))

    cmd = [
        sys.executable, "train.py",
        f"--config-path={config_dir}",
        f"--config-name={config_name}",
        f"path={dataset_path}",
        f"out_dir={exp_dir}",
        f"experiment_name=run",
        f"n_iterations={n_iterations}",
        f"++dataset.single_frame_id={frame_id}",
        f"++dataset.held_out_camera={held_out_camera}",
        f"++random_seed={seed}",
    ]

    print(f"  运行训练: {' '.join(cmd)}")
    try:
        # 设置环境变量确保 CUDA 确定性
        env = os.environ.copy()
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        env["PYTHONHASHSEED"] = str(seed)

        result = subprocess.run(cmd, capture_output=True, text=True,
                                cwd=str(Path(__file__).parent.parent), env=env)
        hydra_dir = _find_hydra_dir(exp_dir)
        if result.returncode != 0:
            # 写入完整错误日志到 Hydra 输出目录
            error_log = hydra_dir / "error.log"
            error_log.parent.mkdir(parents=True, exist_ok=True)
            with open(error_log, 'w') as f:
                f.write("=== COMMAND ===\n")
                f.write(' '.join(cmd) + '\n\n')
                f.write("=== STDOUT ===\n")
                f.write(result.stdout or '(empty)')
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr or '(empty)')
            print(f"  训练失败! 完整日志: {error_log}")
            return False

        # 成功：解析训练日志中的最终 PSNR
        _save_train_metrics(exp_dir, result.stdout)
        print(f"  训练完成")
        return True

    except Exception as e:
        hydra_dir = _find_hydra_dir(exp_dir)
        error_log = hydra_dir / "error.log"
        error_log.parent.mkdir(parents=True, exist_ok=True)
        with open(error_log, 'w') as f:
            f.write(f"Exception: {e}\n")
        print(f"  训练异常: {e}")
        return False


def run_baseline_training(base_config: str, exp_dir: Path, frame_id: int,
                          n_iterations: int, dataset_path: str,
                          seed: int = 42) -> bool:
    """运行全7视角基线训练（无 held-out camera）"""
    config_path = Path(base_config).resolve()
    # Hydra config-path must be the configs/ root (not configs/apps/) so that
    # defaults like /render:3dgut can resolve render/3dgut.yaml etc.
    # config-name uses the relative path from configs/ root, e.g. apps/wildtrack_3dgut
    configs_root = config_path
    while configs_root.parent.name != "configs" and configs_root.parent != configs_root:
        configs_root = configs_root.parent
    if configs_root.parent.name == "configs":
        configs_root = configs_root.parent
    config_dir = str(configs_root)
    config_name = str(config_path.relative_to(configs_root).with_suffix(''))

    cmd = [
        sys.executable, "train.py",
        f"--config-path={config_dir}",
        f"--config-name={config_name}",
        f"path={dataset_path}",
        f"out_dir={exp_dir}",
        f"experiment_name=run",
        f"n_iterations={n_iterations}",
        f"++dataset.single_frame_id={frame_id}",
        f"++random_seed={seed}",
    ]

    print(f"  运行基线训练: {' '.join(cmd)}")
    try:
        env = os.environ.copy()
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        env["PYTHONHASHSEED"] = str(seed)

        result = subprocess.run(cmd, capture_output=True, text=True,
                                cwd=str(Path(__file__).parent.parent), env=env)
        hydra_dir = _find_hydra_dir(exp_dir)
        if result.returncode != 0:
            error_log = hydra_dir / "error.log"
            error_log.parent.mkdir(parents=True, exist_ok=True)
            with open(error_log, 'w') as f:
                f.write("=== COMMAND ===\n")
                f.write(' '.join(cmd) + '\n\n')
                f.write("=== STDOUT ===\n")
                f.write(result.stdout or '(empty)')
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr or '(empty)')
            print(f"  基线训练失败! 完整日志: {error_log}")
            return False

        _save_train_metrics(exp_dir, result.stdout)
        print(f"  基线训练完成")
        return True

    except Exception as e:
        hydra_dir = _find_hydra_dir(exp_dir)
        error_log = hydra_dir / "error.log"
        error_log.parent.mkdir(parents=True, exist_ok=True)
        with open(error_log, 'w') as f:
            f.write(f"Exception: {e}\n")
        print(f"  基线训练异常: {e}")
        return False


def _save_train_metrics(exp_dir: Path, stdout: str):
    """从训练日志解析最终 PSNR 并保存到 Hydra 输出目录"""
    metrics = {"final_val_psnr": None}
    for line in reversed(stdout.splitlines()):
        if "psnr/val" in line.lower() or "Val PSNR" in line:
            # 尝试提取数字
            import re
            m = re.search(r'(\d+\.\d+)', line.split("psnr")[-1])
            if m:
                metrics["final_val_psnr"] = float(m.group(1))
                break

    hydra_dir = _find_hydra_dir(exp_dir)
    metrics_path = hydra_dir / "train_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Leave-One-Out 跨视角泛化验证实验")
    parser.add_argument("--frames", nargs="+", type=int, required=True,
                        help="要测试的帧ID列表")
    parser.add_argument("--config", type=str, required=True,
                        help="基础配置文件路径")
    parser.add_argument("--output_root", type=str, default="experiments/leave_one_out",
                        help="实验输出根目录")
    parser.add_argument("--dataset_path", type=str,
                        default="/data02/zhangrunxiang/data/Wildtrack_small_sample",
                        help="WildTrack 数据集路径")
    parser.add_argument("--n_iterations", type=int, default=15000,
                        help="训练迭代数")
    parser.add_argument("--include_baseline", action="store_true",
                        help="对每个帧额外跑一次全7视角训练作为基线")
    parser.add_argument("--cameras", nargs="+", type=str, default=None,
                        help="要测试的 held-out 相机列表（默认全部7个）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（默认42）。所有 fold 使用相同种子，"
                             "确保 PSNR 差异仅来自视角缺失而非初始化差异")

    args = parser.parse_args()

    cameras = args.cameras or [f"C{i}" for i in range(1, 8)]
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # 在主进程中设置种子（用于数据集 shuffle 等确定性操作）
    _set_seed(args.seed)

    print("=" * 80)
    print("Leave-One-Out 跨视角泛化验证实验")
    print("=" * 80)
    print(f"帧列表: {args.frames}")
    print(f"相机列表: {cameras}")
    print(f"训练迭代数: {args.n_iterations}")
    print(f"随机种子: {args.seed}")
    print(f"数据集路径: {args.dataset_path}")
    print(f"输出根目录: {output_root}")
    print(f"包含基线: {args.include_baseline}")
    print("=" * 80)

    total = len(args.frames) * len(cameras)
    if args.include_baseline:
        total += len(args.frames)
    completed = 0
    failed = []

    for frame_id in args.frames:
        print(f"\n{'=' * 80}")
        print(f"处理帧 {frame_id}")
        print(f"{'=' * 80}")

        # 基线对照（全7视角训练）
        if args.include_baseline:
            completed += 1
            print(f"\n[{completed}/{total}] 帧{frame_id}, 基线(全7视角)")
            baseline_dir = output_root / f"frame_{frame_id}" / "baseline"
            baseline_dir.mkdir(parents=True, exist_ok=True)
            success = run_baseline_training(
                args.config, baseline_dir, frame_id,
                args.n_iterations, args.dataset_path, seed=args.seed
            )
            if not success:
                failed.append((frame_id, "baseline", "training"))

        # LOO 实验
        for held_cam in cameras:
            completed += 1
            print(f"\n[{completed}/{total}] 帧{frame_id}, held-out={held_cam}")

            exp_dir = output_root / f"frame_{frame_id}" / f"held_{held_cam}"
            exp_dir.mkdir(parents=True, exist_ok=True)

            success = run_training(
                args.config, exp_dir, frame_id, held_cam,
                args.n_iterations, args.dataset_path, seed=args.seed
            )
            if not success:
                failed.append((frame_id, held_cam, "training"))

    # 输出总结
    print(f"\n{'=' * 80}")
    print("实验完成总结")
    print("=" * 80)
    print(f"总实验数: {total}")
    print(f"成功: {total - len(failed)}")
    print(f"失败: {len(failed)}")
    if failed:
        print("\n失败列表:")
        for fid, cam, stage in failed:
            print(f"  - 帧{fid}, {cam}, 阶段: {stage}")

    print(f"\n输出目录: {output_root}")
    print(f"\n渲染结果位置: {output_root}/frame_{{fid}}/held_{{cam}}/run/Wildtrack-*/ours_{{N}}/renders/")
    print(f"下一步: 运行 eval_leave_one_out.py 进行评估")


if __name__ == "__main__":
    main()
