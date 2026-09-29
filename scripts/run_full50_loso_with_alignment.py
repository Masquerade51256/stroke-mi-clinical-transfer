"""
Run the full 50-subject LOSO hemisphere-alignment experiment on XWStroke.

Runs Task C (Affected/unaffected labels + hemisphere alignment):
  configs/experiment/xwstroke_full_loso_affected_aligned.yaml
After completion, results are automatically compared against the existing
Task A (Left/Right) results.

Usage:
    python scripts/run_full50_loso_with_alignment.py
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


def compare_with_baseline(results_aligned):
    """Compare Task C (Aligned) with existing Task A (LR) results."""
    res_a_path = Path("experiments") / "XWStroke_Full50_LR" / "results" / "results.json"
    res_c_path = Path("experiments") / "XWStroke_Full50_LOSO_Affected_Aligned" / "results" / "results.json"

    if not res_a_path.exists():
        print(f"\n[ERROR] Task A results not found: {res_a_path}")
        print("        Please ensure the Left/Right LOSO experiment has been run.")
        return 1

    if not res_c_path.exists():
        print(f"\n[ERROR] Task C results not found: {res_c_path}")
        return 1

    print(f"\n{'='*70}")
    print("LOSO Comparison: Task A (LR) vs Task C (Affected + Hemisphere Alignment)")
    print(f"{'='*70}")

    with open(res_a_path, 'r', encoding='utf-8') as f:
        res_a = json.load(f)

    rows = []
    for r_a, r_c in zip(res_a['subjects'], results_aligned['subjects']):
        sid = r_a['test_subject_id']
        rows.append({
            'Subject': f'sub-{sid:02d}',
            'TaskA_LR': r_a['test_acc'],
            'TaskC_Aligned': r_c['test_acc'],
            'Diff': r_c['test_acc'] - r_a['test_acc'],
            'Diff_Pct': f"{(r_c['test_acc'] - r_a['test_acc'])*100:+.1f}%",
        })

    df = pd.DataFrame(rows)

    # Print summary
    print("\nTop 10 Improvers:")
    top_improved = df.nlargest(10, 'Diff')[['Subject', 'TaskA_LR', 'TaskC_Aligned', 'Diff_Pct']]
    print(top_improved.to_string(index=False))

    print("\nTop 10 Decliners:")
    top_declined = df.nsmallest(10, 'Diff')[['Subject', 'TaskA_LR', 'TaskC_Aligned', 'Diff_Pct']]
    print(top_declined.to_string(index=False))

    m_lr = res_a['overall_mean']*100
    s_lr = res_a['overall_std']*100
    m_c = results_aligned['overall_mean']*100
    s_c = results_aligned['overall_std']*100
    diff_mean = m_c - m_lr
    improved = int((df['Diff'] > 0).sum())

    print(f"\n{'='*70}")
    print("Overall Summary:")
    print(f"  Task A (Left/Right):                 {m_lr:.2f}% ± {s_lr:.2f}%")
    print(f"  Task C (Affected + Hemisphere Align): {m_c:.2f}% ± {s_c:.2f}%")
    print(f"  Difference (C - A):                  {diff_mean:+.2f}%")
    print(f"  Relative Improvement:                {diff_mean/m_lr*100:+.1f}%")
    print(f"  Improved subjects:                   {improved}/50 ({improved/50*100:.0f}%)")
    print(f"  Variance change:                     {s_lr:.2f}% → {s_c:.2f}%")
    print(f"{'='*70}")

    # Save quick comparison CSV
    out_dir = Path("results/comparisons_aligned")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_display = df.copy()
    df_display['TaskA_LR'] = (df_display['TaskA_LR'] * 100).round(2)
    df_display['TaskC_Aligned'] = (df_display['TaskC_Aligned'] * 100).round(2)
    df_display['Diff'] = (df_display['Diff'] * 100).round(2)
    df_display.to_csv(out_dir / 'aligned_vs_lr_comparison.csv', index=False, encoding='utf-8-sig')

    summary = {
        'task_a_lr': {'mean': m_lr, 'std': s_lr, 'n': 50},
        'task_c_aligned': {'mean': m_c, 'std': s_c, 'n': 50},
        'difference': diff_mean,
        'relative_improvement_pct': diff_mean/m_lr*100,
        'improved_count': improved,
        'total_count': 50,
        'timestamp': datetime.now().isoformat(),
    }
    with open(out_dir / 'aligned_vs_lr_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved quick comparison to:")
    print(f"  - {out_dir / 'aligned_vs_lr_comparison.csv'}")
    print(f"  - {out_dir / 'aligned_vs_lr_summary.json'}")

    # Run full analysis script
    print(f"\n[Auto] Running full analysis script...")
    analysis_cmd = [
        sys.executable, "scripts/run_analysis_aligned.py",
        "--results-a", str(res_a_path),
        "--results-b", str(res_c_path),
        "--label-a", "Left/Right",
        "--label-b", "Affected/Unaffected + Hemisphere Alignment",
        "--output-dir", "results/comparisons_aligned",
    ]
    subprocess.run(analysis_cmd, cwd=os.path.join(os.path.dirname(__file__), '..'))

    return 0


def main():
    print("\n" + "="*70)
    print("XWStroke Full 50-Subject LOSO with Hemisphere Alignment")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("This experiment runs ONLY Task C (Affected + Hemisphere Alignment).")
    print("Task A (Left/Right) results will be loaded from existing experiment.")
    print("WARNING: This will take approximately 4-6 hours to complete.")
    print("Press Ctrl+C within 5 seconds to cancel...")

    try:
        time.sleep(5)
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        return 0

    total_start = time.time()

    # Task C: Affected/Unaffected + Hemisphere Alignment
    res_aligned = run_exp(
        "configs/experiment/xwstroke_full_loso_affected_aligned.yaml",
        "XWStroke_Full50_LOSO_Affected_Aligned"
    )

    total_elapsed = time.time() - total_start
    print(f"\nTotal elapsed time: {timedelta(seconds=int(total_elapsed))}")

    if res_aligned:
        return compare_with_baseline(res_aligned)
    else:
        print("\n[ERROR] Experiment failed. Please check logs.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
