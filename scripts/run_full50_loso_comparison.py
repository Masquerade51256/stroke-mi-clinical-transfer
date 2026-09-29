"""
Run the full 50-subject LOSO comparison on XWStroke.

Runs two Streaming-LOSO experiments sequentially:
  Task A (Left/Right labels):      configs/experiment/xwstroke_full_loso_lr.yaml
  Task B (Affected/unaffected):    configs/experiment/xwstroke_full_loso_affected.yaml

Notes:
  - 50 subjects x 100 epochs: roughly 4-6 hours per experiment on a modern GPU
  - Uses StreamingLOSOTrainer; memory footprint is about 2-4 GB

Usage:
    python scripts/run_full50_loso_comparison.py
"""

import sys, os, json, subprocess, time, pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def run_exp(config_path, exp_name):
    """Run experiment with fixed name to ensure deterministic output directory."""
    print(f"\n{'='*70}")
    print(f"Running: {config_path}")
    print(f"Experiment name: {exp_name}")
    print(f"{'='*70}")
    
    start_time = time.time()
    cmd = [sys.executable, "train.py", "--config", config_path, "--name", exp_name]
    result = subprocess.run(cmd, cwd=os.path.join(os.path.dirname(__file__), '..'))
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"[ERROR] Failed: {config_path} (elapsed: {timedelta(seconds=int(elapsed))})")
        return None
    
    res_file = Path("experiments") / exp_name / "results" / "results.json"
    if not res_file.exists():
        print(f"[ERROR] Results file not found: {res_file}")
        return None
    
    with open(res_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"[OK] Completed in {timedelta(seconds=int(elapsed))}")
    print(f"     Overall: {data['overall_mean']*100:.2f}% ± {data['overall_std']*100:.2f}%")
    return data


def compare(results_lr, results_aff):
    """Compare two LOSO results and output statistics."""
    print(f"\n{'='*70}")
    print("LOSO Comparison (Full 50 subjects)")
    print(f"{'='*70}")
    
    rows = []
    for r_lr, r_aff in zip(results_lr['subjects'], results_aff['subjects']):
        sid = r_lr['test_subject_id']
        rows.append({
            'Subject': f'sub-{sid:02d}',
            'TaskA_LR': r_lr['test_acc'],
            'TaskB_Aff': r_aff['test_acc'],
            'Diff': r_aff['test_acc'] - r_lr['test_acc'],
            'Diff_Pct': f"{(r_aff['test_acc'] - r_lr['test_acc'])*100:+.1f}%",
        })
    
    df = pd.DataFrame(rows)
    
    # Print summary
    print("\nTop 10 Improvers:")
    top_improved = df.nlargest(10, 'Diff')[['Subject', 'TaskA_LR', 'TaskB_Aff', 'Diff_Pct']]
    print(top_improved.to_string(index=False))
    
    print("\nTop 10 Decliners:")
    top_declined = df.nsmallest(10, 'Diff')[['Subject', 'TaskA_LR', 'TaskB_Aff', 'Diff_Pct']]
    print(top_declined.to_string(index=False))
    
    m_lr = results_lr['overall_mean']*100
    s_lr = results_lr['overall_std']*100
    m_aff = results_aff['overall_mean']*100
    s_aff = results_aff['overall_std']*100
    diff_mean = m_aff - m_lr
    improved = int((df['Diff'] > 0).sum())
    
    print(f"\n{'='*70}")
    print("Overall Summary:")
    print(f"  Task A (Left/Right):      {m_lr:.2f}% ± {s_lr:.2f}%")
    print(f"  Task B (Affected/Health): {m_aff:.2f}% ± {s_aff:.2f}%")
    print(f"  Difference (B - A):       {diff_mean:+.2f}%")
    print(f"  Relative Improvement:     {diff_mean/m_lr*100:+.1f}%")
    print(f"  Improved subjects:        {improved}/50 ({improved/50*100:.0f}%)")
    print(f"  Variance change:          {s_lr:.2f}% → {s_aff:.2f}%")
    print(f"{'='*70}")
    
    # Save outputs
    out_dir = Path("results/comparisons")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Detailed CSV
    df_display = df.copy()
    df_display['TaskA_LR'] = (df_display['TaskA_LR'] * 100).round(2)
    df_display['TaskB_Aff'] = (df_display['TaskB_Aff'] * 100).round(2)
    df_display['Diff'] = (df_display['Diff'] * 100).round(2)
    df_display.to_csv(out_dir / 'full50_loso_comparison.csv', index=False, encoding='utf-8-sig')
    
    # Summary JSON
    summary = {
        'task_a_lr': {'mean': m_lr, 'std': s_lr, 'n': 50},
        'task_b_affected': {'mean': m_aff, 'std': s_aff, 'n': 50},
        'difference': diff_mean,
        'relative_improvement_pct': diff_mean/m_lr*100,
        'improved_count': improved,
        'total_count': 50,
        'timestamp': datetime.now().isoformat(),
    }
    with open(out_dir / 'full50_loso_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved to:")
    print(f"  - {out_dir / 'full50_loso_comparison.csv'}")
    print(f"  - {out_dir / 'full50_loso_summary.json'}")
    
    # Run full analysis script automatically
    print(f"\n[Auto] Running full analysis script...")
    res_a_path = Path("experiments") / "XWStroke_Full50_LR" / "results" / "results.json"
    res_b_path = Path("experiments") / "XWStroke_Full50_Affected" / "results" / "results.json"
    if res_a_path.exists() and res_b_path.exists():
        analysis_cmd = [
            sys.executable, "scripts/run_full_analysis.py",
            "--results-a", str(res_a_path),
            "--results-b", str(res_b_path),
            "--label-a", "Left/Right",
            "--label-b", "Affected/Unaffected",
        ]
        subprocess.run(analysis_cmd, cwd=os.path.join(os.path.dirname(__file__), '..'))


def main():
    print("\n" + "="*70)
    print("XWStroke Full 50-Subject LOSO Validation")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("WARNING: This will take approximately 8-12 hours to complete.")
    print("Press Ctrl+C within 5 seconds to cancel...")
    
    try:
        time.sleep(5)
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        return 0
    
    total_start = time.time()
    
    # Task A: Left/Right
    res_lr = run_exp("configs/experiment/xwstroke_full_loso_lr.yaml", "XWStroke_Full50_LR")
    
    # Task B: Affected/Unaffected
    res_aff = run_exp("configs/experiment/xwstroke_full_loso_affected.yaml", "XWStroke_Full50_Affected")
    
    total_elapsed = time.time() - total_start
    print(f"\nTotal elapsed time: {timedelta(seconds=int(total_elapsed))}")
    
    if res_lr and res_aff:
        compare(res_lr, res_aff)
    else:
        print("\n[ERROR] Some experiment failed. Please check logs.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
