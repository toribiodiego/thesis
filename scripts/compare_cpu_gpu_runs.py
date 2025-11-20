#!/usr/bin/env python3
"""
Compare CPU vs GPU training runs for hardware validation.

Fetches GPU run data from W&B and compares against local CPU baseline.
"""

import argparse
import csv
import os
import statistics
import wandb
from pathlib import Path


def analyze_local_run(run_dir):
    """Analyze local training run from CSV files."""
    run_path = Path(run_dir)

    # Load training steps data
    training_csv = run_path / "csv" / "training_steps.csv"
    if not training_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {training_csv}")

    # Read CSV with csv module
    with open(training_csv, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Extract metrics
    fps_values = [float(row['fps']) for row in rows]
    loss_values = [float(row['loss']) for row in rows]
    env_steps = [int(row['env_step']) for row in rows]
    episodes = [int(row['episode']) for row in rows]

    # Calculate stats
    total_frames = max(env_steps)
    total_episodes = max(episodes)
    fps_mean = statistics.mean(fps_values)
    fps_median = statistics.median(fps_values)
    fps_std = statistics.stdev(fps_values) if len(fps_values) > 1 else 0
    loss_initial = loss_values[0]
    loss_final = loss_values[-1]
    loss_mean = statistics.mean(loss_values)

    # Evaluation data
    eval_csv = run_path / "eval" / "evaluations.csv"
    eval_data = None
    if eval_csv.exists():
        with open(eval_csv, 'r') as f:
            reader = csv.DictReader(f)
            eval_rows = list(reader)
        if eval_rows:
            eval_data = {
                'num_evals': len(eval_rows),
                'final_mean_return': float(eval_rows[-1]['mean_return']),
                'final_std_return': float(eval_rows[-1]['std_return']),
            }

    return {
        'run_dir': str(run_path),
        'total_frames': total_frames,
        'total_episodes': total_episodes,
        'fps_mean': fps_mean,
        'fps_median': fps_median,
        'fps_std': fps_std,
        'loss_initial': loss_initial,
        'loss_final': loss_final,
        'loss_mean': loss_mean,
        'eval_data': eval_data,
    }


def fetch_wandb_run(entity, project, run_id):
    """Fetch run data from W&B."""
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    # Get summary metrics
    summary = run.summary._json_dict

    # Get history (training steps data)
    history = run.history()

    return {
        'run_id': run_id,
        'name': run.name,
        'config': run.config,
        'summary': summary,
        'history': history,
        'url': run.url,
    }


def compare_runs(cpu_stats, gpu_data):
    """Compare CPU and GPU run metrics."""
    gpu_history = gpu_data['history']

    # Calculate GPU metrics from history dataframe
    gpu_fps_values = gpu_history['fps'].tolist() if 'fps' in gpu_history.columns else []
    gpu_loss_values = gpu_history['loss'].tolist() if 'loss' in gpu_history.columns else []

    gpu_fps_mean = statistics.mean(gpu_fps_values) if gpu_fps_values else None
    gpu_fps_median = statistics.median(gpu_fps_values) if gpu_fps_values else None
    gpu_loss_final = gpu_loss_values[-1] if gpu_loss_values else None
    gpu_total_frames = len(gpu_history) * 4000 if len(gpu_history) > 0 else None  # Assuming 4k step intervals

    comparison = {
        'cpu': {
            'fps_mean': cpu_stats['fps_mean'],
            'fps_median': cpu_stats['fps_median'],
            'loss_final': cpu_stats['loss_final'],
            'total_frames': cpu_stats['total_frames'],
        },
        'gpu': {
            'fps_mean': gpu_fps_mean,
            'fps_median': gpu_fps_median,
            'loss_final': gpu_loss_final,
            'total_frames': gpu_total_frames,
        },
        'speedup': {
            'fps_mean_ratio': gpu_fps_mean / cpu_stats['fps_mean'] if gpu_fps_mean else None,
            'fps_median_ratio': gpu_fps_median / cpu_stats['fps_median'] if gpu_fps_median else None,
        },
        'convergence_match': {
            'loss_diff': abs(gpu_loss_final - cpu_stats['loss_final']) if gpu_loss_final else None,
        }
    }

    return comparison


def print_comparison(comparison):
    """Print formatted comparison report."""
    print("\n" + "="*60)
    print("CPU vs GPU Hardware Comparison")
    print("="*60)

    print("\nCPU Performance (Mac M1):")
    print(f"  FPS Mean:    {comparison['cpu']['fps_mean']:.2f}")
    print(f"  FPS Median:  {comparison['cpu']['fps_median']:.2f}")
    print(f"  Final Loss:  {comparison['cpu']['loss_final']:.4f}")
    print(f"  Total Frames: {comparison['cpu']['total_frames']:,}")

    print("\nGPU Performance (Colab A100):")
    gpu_fps_mean = comparison['gpu']['fps_mean']
    gpu_fps_median = comparison['gpu']['fps_median']
    gpu_loss_final = comparison['gpu']['loss_final']
    gpu_total_frames = comparison['gpu']['total_frames']

    print(f"  FPS Mean:    {gpu_fps_mean:.2f if gpu_fps_mean else 'N/A'}")
    print(f"  FPS Median:  {gpu_fps_median:.2f if gpu_fps_median else 'N/A'}")
    print(f"  Final Loss:  {gpu_loss_final:.4f if gpu_loss_final else 'N/A'}")
    print(f"  Total Frames: {gpu_total_frames:,}" if gpu_total_frames else "  Total Frames: N/A")

    print("\nSpeedup:")
    if comparison['speedup']['fps_mean_ratio']:
        print(f"  FPS Mean:    {comparison['speedup']['fps_mean_ratio']:.2f}x")
    if comparison['speedup']['fps_median_ratio']:
        print(f"  FPS Median:  {comparison['speedup']['fps_median_ratio']:.2f}x")

    print("\nConvergence Match:")
    if comparison['convergence_match']['loss_diff'] is not None:
        print(f"  Loss Difference: {comparison['convergence_match']['loss_diff']:.4f}")

    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Compare CPU vs GPU training runs')
    parser.add_argument('--cpu-run', required=True, help='Path to local CPU run directory')
    parser.add_argument('--gpu-run-id', required=True, help='W&B run ID for GPU run')
    parser.add_argument('--entity', default='Cooper-Union', help='W&B entity')
    parser.add_argument('--project', default='dqn-atari', help='W&B project')
    parser.add_argument('--output', help='Output markdown file path')

    args = parser.parse_args()

    print("Analyzing CPU run...")
    cpu_stats = analyze_local_run(args.cpu_run)

    print("Fetching GPU run from W&B...")
    gpu_data = fetch_wandb_run(args.entity, args.project, args.gpu_run_id)

    print("Comparing runs...")
    comparison = compare_runs(cpu_stats, gpu_data)

    print_comparison(comparison)

    if args.output:
        # Write detailed comparison to markdown
        with open(args.output, 'w') as f:
            f.write("# GPU Validation: CPU vs GPU Hardware Comparison\n\n")
            f.write(f"**CPU Run:** `{cpu_stats['run_dir']}`\n")
            f.write(f"**GPU Run:** [{gpu_data['run_id']}]({gpu_data['url']})\n\n")
            f.write("## Performance Metrics\n\n")
            f.write("| Metric | CPU (Mac M1) | GPU (Colab A100) | Speedup |\n")
            f.write("|--------|--------------|------------------|----------|\n")

            gpu_fps_mean_str = f"{comparison['gpu']['fps_mean']:.2f}" if comparison['gpu']['fps_mean'] else 'N/A'
            gpu_fps_median_str = f"{comparison['gpu']['fps_median']:.2f}" if comparison['gpu']['fps_median'] else 'N/A'
            gpu_loss_final_str = f"{comparison['gpu']['loss_final']:.4f}" if comparison['gpu']['loss_final'] else 'N/A'
            gpu_total_frames_str = f"{comparison['gpu']['total_frames']:,}" if comparison['gpu']['total_frames'] else 'N/A'
            speedup_mean_str = f"{comparison['speedup']['fps_mean_ratio']:.2f}x" if comparison['speedup']['fps_mean_ratio'] else '-'
            speedup_median_str = f"{comparison['speedup']['fps_median_ratio']:.2f}x" if comparison['speedup']['fps_median_ratio'] else '-'

            f.write(f"| FPS Mean | {comparison['cpu']['fps_mean']:.2f} | {gpu_fps_mean_str} | {speedup_mean_str} |\n")
            f.write(f"| FPS Median | {comparison['cpu']['fps_median']:.2f} | {gpu_fps_median_str} | {speedup_median_str} |\n")
            f.write(f"| Final Loss | {comparison['cpu']['loss_final']:.4f} | {gpu_loss_final_str} | - |\n")
            f.write(f"| Total Frames | {comparison['cpu']['total_frames']:,} | {gpu_total_frames_str} | - |\n")
            f.write("\n## Convergence Check\n\n")
            f.write(f"Loss difference: {comparison['convergence_match']['loss_diff']:.4f}\n\n")
        print(f"Detailed comparison written to: {args.output}")


if __name__ == '__main__':
    main()
