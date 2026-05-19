#!/usr/bin/env python3
"""
Phase 14: Collect smoke run metrics and generate comparison report.
"""

import re
import csv
import json
from pathlib import Path
from collections import defaultdict

def parse_train_log(log_path: Path):
    """Parse training log to extract metrics."""
    if not log_path.exists():
        return None
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Extract validation metrics
    # Pattern: Step XXXX -- Running validation.. followed by table
    step_pattern = r'Step (\d+) -- Running validation\.'
    psnr_pattern = r'│\s+(\d+\.\d+)\s*│\s+(\d+\.\d+)\s*│\s+(\d+\.\d+)\s*│'
    
    steps = re.findall(step_pattern, content)
    psnr_matches = re.findall(psnr_pattern, content)
    
    metrics = []
    for i, step in enumerate(steps):
        if i < len(psnr_matches):
            psnr, ssim, lpips = psnr_matches[i]
            metrics.append({
                'step': int(step),
                'psnr': float(psnr),
                'ssim': float(ssim),
                'lpips': float(lpips),
            })
    
    # Extract Gaussian counts
    gaussian_pattern = r'(?:Cloned|Splitted|Density-pruned)\s+(\d+)\s*/\s*(\d+)'
    gaussian_matches = re.findall(gaussian_pattern, content)
    gaussian_counts = [int(m[1]) for m in gaussian_matches]
    
    # Extract clone/split/prune counts
    clone_pattern = r'Cloned\s+(\d+)\s*/'
    split_pattern = r'Splitted\s+(\d+)\s*/'
    prune_pattern = r'Density-pruned\s+(\d+)\s*/'
    
    clone_counts = [int(x) for x in re.findall(clone_pattern, content)]
    split_counts = [int(x) for x in re.findall(split_pattern, content)]
    prune_counts = [int(x) for x in re.findall(prune_pattern, content)]
    
    # Extract density stats from logs (if available)
    # These may not be in the log, so we'll use defaults
    
    return {
        'metrics': metrics,
        'gaussian_counts': gaussian_counts,
        'clone_counts': clone_counts,
        'split_counts': split_counts,
        'prune_counts': prune_counts,
    }


def generate_comparison_report(base_dir: Path):
    """Generate comparison report from all smoke runs."""
    schemes = {
        'smoke_old_config': 'Old Config (freq=3000, max=0.01)',
        'smoke_no_reset': 'No Reset (freq=999999)',
        'smoke_soft_reset': 'Soft Reset (freq=10000, max=0.1)',
    }
    
    results = {}
    for scheme_dir, scheme_name in schemes.items():
        log_path = base_dir / scheme_dir / 'train.log'
        if not log_path.exists():
            log_path = base_dir / scheme_dir / 'train_stdout.log'
        
        parsed = parse_train_log(log_path)
        if parsed is None:
            print(f"  WARNING: No log found for {scheme_dir}")
            continue
        
        results[scheme_name] = parsed
    
    # Generate CSV
    csv_path = base_dir / 'smoke_compare.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'scheme', 'reset_frequency', 'new_max_density',
            'psnr_1k', 'psnr_3k', 'psnr_5k',
            'ssim_3k', 'ssim_5k',
            'gaussian_N_1k', 'gaussian_N_3k', 'gaussian_N_5k',
            'clone_total', 'split_total', 'prune_total',
            'collapse_detected', 'verdict'
        ])
        
        for scheme_name, data in results.items():
            metrics = data['metrics']
            
            # Extract values at specific steps
            def get_metric_at_step(step_target, metric_name, default=None):
                for m in metrics:
                    if m['step'] == step_target:
                        return m.get(metric_name, default)
                return default
            
            psnr_1k = get_metric_at_step(1000, 'psnr', 'N/A')
            psnr_3k = get_metric_at_step(3000, 'psnr', 'N/A')
            psnr_5k = get_metric_at_step(5000, 'psnr', 'N/A')
            ssim_3k = get_metric_at_step(3000, 'ssim', 'N/A')
            ssim_5k = get_metric_at_step(5000, 'ssim', 'N/A')
            
            # Gaussian counts (approximate by index)
            gc = data['gaussian_counts']
            gaussian_N_1k = gc[len(gc)//5] if len(gc) > 5 else 'N/A'
            gaussian_N_3k = gc[3*len(gc)//5] if len(gc) > 5 else 'N/A'
            gaussian_N_5k = gc[-1] if gc else 'N/A'
            
            clone_total = sum(data['clone_counts'])
            split_total = sum(data['split_counts'])
            prune_total = sum(data['prune_counts'])
            
            # Detect collapse
            collapse = False
            if isinstance(psnr_3k, (int, float)) and isinstance(psnr_1k, (int, float)):
                if psnr_3k < 7 and psnr_1k > 10:
                    collapse = True
            if isinstance(ssim_3k, (int, float)) and ssim_3k < 0.01:
                collapse = True
            
            verdict = 'COLLAPSE' if collapse else 'STABLE'
            
            # Extract reset config from command file
            cmd_path = base_dir / scheme_dir.replace('Old Config (freq=', 'smoke_').replace(')', '') / 'smoke_command.txt'
            # Simplified: use scheme name to infer config
            if 'freq=3000' in scheme_name:
                reset_freq = 3000
                new_max = 0.01
            elif 'freq=999999' in scheme_name:
                reset_freq = 999999
                new_max = 0.01
            elif 'freq=10000' in scheme_name:
                reset_freq = 10000
                new_max = 0.1
            else:
                reset_freq = 'N/A'
                new_max = 'N/A'
            
            writer.writerow([
                scheme_name, reset_freq, new_max,
                psnr_1k, psnr_3k, psnr_5k,
                ssim_3k, ssim_5k,
                gaussian_N_1k, gaussian_N_3k, gaussian_N_5k,
                clone_total, split_total, prune_total,
                collapse, verdict
            ])
    
    print(f"Comparison CSV saved to: {csv_path}")
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='outputs/phase14_clean_geometry/reset_density_ablation')
    args = parser.parse_args()
    
    results = generate_comparison_report(Path(args.base_dir))
    
    print("\nSmoke Run Results Summary:")
    for scheme, data in results.items():
        metrics = data['metrics']
        if metrics:
            last = metrics[-1]
            print(f"  {scheme}: PSNR={last['psnr']:.3f}, SSIM={last['ssim']:.3f} at step {last['step']}")
        else:
            print(f"  {scheme}: No metrics found")
