"""
Stratified subgroup analysis for the XWStroke dataset.

Functionality:
1. Stratify subjects by lesion location (Cortical / Subcortical / Brainstem /
   Cerebellar / Mixed)
2. Stratify subjects by stroke duration (Acute <=3 months / Chronic >3 months)
3. Compute per-subject Lateralization Index (LI)
4. Correlate LI with LOSO decoding accuracy
5. Compare Left/Right vs. Affected/Aligned labelling within each subgroup

Also exports the per-subject clinical metadata table
(results/stratified/stratified_analysis_detailed.csv) used by the
validation-based pipeline selection script.

Usage:
    python scripts/run_stratified_analysis.py
"""

import sys, os, json, re
from pathlib import Path
from collections import Counter

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal, stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Generic fonts
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def load_results(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def classify_stroke_location(loc_str):
    """
    Classify stroke location into broad categories.
    Categories: Cortical, Subcortical, Brainstem, Cerebellar, Mixed
    """
    if pd.isna(loc_str):
        return 'Unknown'

    s = str(loc_str).lower()

    # Keywords for each category
    cortical_kw = ['cortex', 'frontal', 'parietal', 'temporal', 'occipital', 'insula', 'watershed']
    subcortical_kw = ['basal ganglia', 'thalamus', 'internal capsule', 'corona radiata',
                      'centrum semiovale', 'paraventricular', 'subcortical']
    brainstem_kw = ['pons', 'medulla oblongata']
    cerebellar_kw = ['cerebellum', 'cerebellar']

    has_cortical = any(kw in s for kw in cortical_kw)
    has_subcortical = any(kw in s for kw in subcortical_kw)
    has_brainstem = any(kw in s for kw in brainstem_kw)
    has_cerebellar = any(kw in s for kw in cerebellar_kw)

    flags = [has_cortical, has_subcortical, has_brainstem, has_cerebellar]
    n_categories = sum(flags)

    if n_categories >= 2:
        return 'Mixed'
    elif has_cortical:
        return 'Cortical'
    elif has_subcortical:
        return 'Subcortical'
    elif has_brainstem:
        return 'Brainstem'
    elif has_cerebellar:
        return 'Cerebellar'
    else:
        return 'Other'


def compute_mu_power(eeg_data, fs, ch_idx, f_band=(8, 13)):
    """
    Compute average mu-band power for a given channel across trials.

    Args:
        eeg_data: shape (n_trials, n_channels, n_times)
        fs: sampling rate
        ch_idx: channel index
        f_band: frequency band [low, high]

    Returns:
        power_per_trial: shape (n_trials,)
    """
    n_trials = eeg_data.shape[0]
    powers = []

    # Design bandpass filter
    sos = signal.butter(4, f_band, btype='band', fs=fs, output='sos')

    for trial_idx in range(n_trials):
        ch_data = eeg_data[trial_idx, ch_idx, :]
        filtered = signal.sosfiltfilt(sos, ch_data)
        # Instantaneous power via Hilbert
        analytic = signal.hilbert(filtered)
        inst_power = np.abs(analytic) ** 2
        # Average over time
        powers.append(np.mean(inst_power))

    return np.array(powers)


def compute_lateralization_index(subject_id, dataset_info):
    """
    Compute Lateralization Index (LI) for a subject based on mu-band power.

    LI is defined as:
        For left-paralysis:  LI = (P_C3_aff - P_C4_aff) / (P_C3_aff + P_C4_aff)
        For right-paralysis: LI = (P_C4_aff - P_C3_aff) / (P_C4_aff + P_C3_aff)

    Where P_C3_aff = average mu power at C3 during affected-side MI trials.
    Positive LI indicates lateralization (ipsilesional > contralesional).
    LI near 0 indicates bilateral/broken lateralization.

    Returns:
        li_affected: LI for affected-side trials
        li_unaffected: LI for unaffected-side trials
        p_c3_aff, p_c4_aff, p_c3_unaff, p_c4_unaff: raw powers
    """
    from data.datasets import _load_paralysis_side

    # Load raw 4s data
    data_dir = dataset_info.get('dataset', {}).get('data_dir', 'src/datasets/21679035/sourcedata')
    data_var_name = dataset_info.get('dataset', {}).get('data_var_name', 'eeg_data_4s')
    labels_var_name = dataset_info.get('dataset', {}).get('labels_var_name', 'labels')
    fs = dataset_info.get('dataset', {}).get('original_sr', 500)

    subject_file = os.path.join(data_dir, f'sub-{subject_id:02d}',
                                f'sub-{subject_id:02d}_task-motor-imagery_eeg_4s.mat')
    if not os.path.exists(subject_file):
        return None

    mat = sio.loadmat(subject_file)
    eeg_data = mat[data_var_name]  # (trials, ch, time)
    labels = mat[labels_var_name].flatten()  # 1-based: 1=left, 2=right

    # Channel indices for C3 and C4 (standard XWStroke order)
    C3_IDX = 13
    C4_IDX = 14

    # Determine affected side
    paralysis_side = _load_paralysis_side(subject_id, dataset_info)

    # Map labels to affected/unaffected (1-based labels)
    # label 1 = left hand, label 2 = right hand
    if paralysis_side == 'left':
        affected_label = 1    # left hand is affected
        unaffected_label = 2  # right hand is unaffected
    else:
        affected_label = 2    # right hand is affected
        unaffected_label = 1  # left hand is unaffected

    # Compute mu power
    p_c3_all = compute_mu_power(eeg_data, fs, C3_IDX)
    p_c4_all = compute_mu_power(eeg_data, fs, C4_IDX)

    p_c3_aff = np.mean(p_c3_all[labels == affected_label])
    p_c4_aff = np.mean(p_c4_all[labels == affected_label])
    p_c3_unaff = np.mean(p_c3_all[labels == unaffected_label])
    p_c4_unaff = np.mean(p_c4_all[labels == unaffected_label])

    # Compute LI with consistent direction
    if paralysis_side == 'left':
        li_affected = (p_c3_aff - p_c4_aff) / (p_c3_aff + p_c4_aff + 1e-10)
        li_unaffected = (p_c4_unaff - p_c3_unaff) / (p_c4_unaff + p_c3_unaff + 1e-10)
    else:
        li_affected = (p_c4_aff - p_c3_aff) / (p_c4_aff + p_c3_aff + 1e-10)
        li_unaffected = (p_c3_unaff - p_c4_unaff) / (p_c3_unaff + p_c4_unaff + 1e-10)

    return {
        'li_affected': li_affected,
        'li_unaffected': li_unaffected,
        'p_c3_aff': p_c3_aff,
        'p_c4_aff': p_c4_aff,
        'p_c3_unaff': p_c3_unaff,
        'p_c4_unaff': p_c4_unaff,
        'paralysis_side': paralysis_side,
    }


def plot_stratum_boxplot(df, stratify_col, title, out_path):
    """Boxplot comparing Task A vs Task C within each stratum."""
    groups = sorted(df[stratify_col].unique())
    fig, axes = plt.subplots(1, len(groups), figsize=(4 * len(groups), 5), sharey=True)
    if len(groups) == 1:
        axes = [axes]

    for ax, g in zip(axes, groups):
        sub = df[df[stratify_col] == g]
        data_to_plot = [sub['acc_a'].values * 100, sub['acc_c'].values * 100]
        bp = ax.boxplot(data_to_plot, labels=['Task A\n(LR)', 'Task C\n(Aligned)'],
                        patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor('#3498db')
        bp['boxes'][1].set_facecolor('#e67e22')

        # Add mean markers
        ax.scatter([1, 2], [sub['acc_a'].mean() * 100, sub['acc_c'].mean() * 100],
                   color='red', marker='D', zorder=3, label='Mean')

        # Add diff annotation
        diff = (sub['acc_c'].mean() - sub['acc_a'].mean()) * 100
        ax.annotate(f'{diff:+.1f}%', xy=(1.5, max(sub['acc_c'].max(), sub['acc_a'].max()) * 100 + 1),
                    ha='center', fontsize=10, fontweight='bold',
                    color='green' if diff > 0 else 'red')

        ax.set_title(f'{g}\n(N={len(sub)})', fontsize=11)
        ax.set_ylabel('LOSO Accuracy (%)')
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(30, 75)

    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_scatter_li_vs_accuracy(df, out_path):
    """Scatter: Lateralization Index vs LOSO Accuracy."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Task A
    for side, color in [('left', '#3498db'), ('right', '#e67e22')]:
        sub = df[df['ParalysisSide'] == side]
        if len(sub) < 3:
            continue
        axes[0].scatter(sub['li_affected'], sub['acc_a'] * 100,
                       c=color, label=side.capitalize(), s=80, alpha=0.7, edgecolors='black')
        # Regression line
        if len(sub) > 2:
            z = np.polyfit(sub['li_affected'], sub['acc_a'] * 100, 1)
            p = np.poly1d(z)
            x_line = np.linspace(sub['li_affected'].min(), sub['li_affected'].max(), 100)
            axes[0].plot(x_line, p(x_line), '--', color=color, alpha=0.5)

    axes[0].axvline(x=0, color='black', linestyle='-', alpha=0.3)
    axes[0].set_xlabel('Lateralization Index (Affected MI)', fontsize=11)
    axes[0].set_ylabel('Task A LOSO Accuracy (%)', fontsize=11)
    axes[0].set_title('LI vs Accuracy: Task A (Left/Right)', fontsize=12, fontweight='bold')
    axes[0].legend(title='Paralysis Side')
    axes[0].grid(True, alpha=0.3)

    # Task C
    for side, color in [('left', '#3498db'), ('right', '#e67e22')]:
        sub = df[df['ParalysisSide'] == side]
        if len(sub) < 3:
            continue
        axes[1].scatter(sub['li_affected'], sub['acc_c'] * 100,
                       c=color, label=side.capitalize(), s=80, alpha=0.7, edgecolors='black')
        if len(sub) > 2:
            z = np.polyfit(sub['li_affected'], sub['acc_c'] * 100, 1)
            p = np.poly1d(z)
            x_line = np.linspace(sub['li_affected'].min(), sub['li_affected'].max(), 100)
            axes[1].plot(x_line, p(x_line), '--', color=color, alpha=0.5)

    axes[1].axvline(x=0, color='black', linestyle='-', alpha=0.3)
    axes[1].set_xlabel('Lateralization Index (Affected MI)', fontsize=11)
    axes[1].set_ylabel('Task C LOSO Accuracy (%)', fontsize=11)
    axes[1].set_title('LI vs Accuracy: Task C (Aligned)', fontsize=12, fontweight='bold')
    axes[1].legend(title='Paralysis Side')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_li_distribution(df, out_path):
    """Histogram of LI by paralysis side."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for side, color in [('left', '#3498db'), ('right', '#e67e22')]:
        sub = df[df['ParalysisSide'] == side]
        ax.hist(sub['li_affected'], bins=12, alpha=0.6, label=f'{side.capitalize()} (N={len(sub)})',
                color=color, edgecolor='black')

    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='LI=0 (No lateralization)')
    ax.set_xlabel('Lateralization Index (Affected MI)', fontsize=12)
    ax.set_ylabel('Number of Subjects', fontsize=12)
    ax.set_title('Distribution of Lateralization Index by Paralysis Side', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def generate_report(df, corr_results, out_path):
    """Generate Chinese Markdown report."""

    lines = [
        "# XWStroke 分层深度分析报告",
        "",
        "## 1. 病灶位置分层",
        "",
    ]

    # Stroke location table
    loc_summary = df.groupby('StrokeLocation_Category').apply(
        lambda g: pd.Series({
            'N': len(g),
            'TaskA_Mean': f"{g['acc_a'].mean()*100:.1f}%",
            'TaskC_Mean': f"{g['acc_c'].mean()*100:.1f}%",
            'Diff': f"{(g['acc_c'].mean() - g['acc_a'].mean())*100:+.1f}%",
            'Improved': f"{(g['diff'] > 0).sum()}/{len(g)}",
        })
    ).reset_index()

    lines.append(loc_summary.to_markdown(index=False))
    lines.append("")

    # Duration stratification
    lines.append("## 2. 病程分层")
    lines.append("")
    dur_summary = df.groupby('Duration_Category').apply(
        lambda g: pd.Series({
            'N': len(g),
            'TaskA_Mean': f"{g['acc_a'].mean()*100:.1f}%",
            'TaskC_Mean': f"{g['acc_c'].mean()*100:.1f}%",
            'Diff': f"{(g['acc_c'].mean() - g['acc_a'].mean())*100:+.1f}%",
            'Improved': f"{(g['diff'] > 0).sum()}/{len(g)}",
        })
    ).reset_index()
    lines.append(dur_summary.to_markdown(index=False))
    lines.append("")

    # LI correlation
    lines.append("## 3. 侧化指数（LI）分析")
    lines.append("")
    lines.append(f"- **LI 计算方法**：基于患侧 MI 试次的 μ 频段（8-13Hz）瞬时功率，")
    lines.append(f"  统一方向定义为：(P_病灶同侧 - P_病灶对侧) / (P_病灶同侧 + P_病灶对侧)。")
    lines.append(f"  LI > 0 表示存在侧化，LI ≈ 0 表示双侧化/侧化受损。")
    lines.append("")
    lines.append(f"- **左偏瘫患者** (N={len(df[df['ParalysisSide']=='left'])})：")
    lines.append(f"  LI 均值 = {df[df['ParalysisSide']=='left']['li_affected'].mean():.3f} ± {df[df['ParalysisSide']=='left']['li_affected'].std():.3f}")
    lines.append(f"  LI 范围 = [{df[df['ParalysisSide']=='left']['li_affected'].min():.3f}, {df[df['ParalysisSide']=='left']['li_affected'].max():.3f}]")
    lines.append("")
    lines.append(f"- **右偏瘫患者** (N={len(df[df['ParalysisSide']=='right'])})：")
    lines.append(f"  LI 均值 = {df[df['ParalysisSide']=='right']['li_affected'].mean():.3f} ± {df[df['ParalysisSide']=='right']['li_affected'].std():.3f}")
    lines.append(f"  LI 范围 = [{df[df['ParalysisSide']=='right']['li_affected'].min():.3f}, {df[df['ParalysisSide']=='right']['li_affected'].max():.3f}]")
    lines.append("")

    # Correlations
    lines.append("### LI 与 LOSO 准确率的相关性")
    lines.append("")
    for task, label in [('acc_a', 'Task A (LR)'), ('acc_c', 'Task C (Aligned)')]:
        lines.append(f"**{label}**：")
        for side in ['left', 'right', 'all']:
            key = f'{task}_{side}'
            if key in corr_results:
                r, p = corr_results[key]
                sig = "*" if p < 0.05 else "†" if p < 0.10 else ""
                lines.append(f"  - {side.capitalize()}: r = {r:.3f}, p = {p:.4f} {sig}")
        lines.append("")

    # Interpretation
    lines.append("## 4. 关键发现与解读")
    lines.append("")

    # Find best stratum
    best_loc = loc_summary.loc[(loc_summary['Diff'].str.replace('%','').str.replace('+','').astype(float)).idxmax(), 'StrokeLocation_Category']
    best_diff = loc_summary.loc[(loc_summary['Diff'].str.replace('%','').str.replace('+','').astype(float)).idxmax(), 'Diff']

    lines.append(f"1. **病灶位置效应**：{best_loc} 病灶患者从半球对齐中受益最多（{best_diff}）。")
    lines.append(f"   混合病灶（Mixed）患者通常表现更差，可能因为双侧或多发损伤破坏了可迁移的模板。")
    lines.append("")

    lines.append(f"2. **病程效应**：急性期（≤3月）与慢性期（>3月）患者的改善模式可能不同，")
    lines.append(f"   需要结合具体临床意义解读。急性期脑可塑性更强，但空间模式也更不稳定。")
    lines.append("")

    # LI interpretation
    li_both_pos = ((df['li_affected'] > 0).sum())
    li_both_neg = ((df['li_affected'] < 0).sum())
    lines.append(f"3. **侧化指数分布**：50名患者中，{li_both_pos} 人 LI > 0（保留侧化），")
    lines.append(f"   {li_both_neg} 人 LI < 0（侧化反转/受损）。")
    lines.append("")

    if 'acc_c_all' in corr_results:
        r_all, p_all = corr_results['acc_c_all']
        if p_all < 0.10:
            lines.append(f"4. **LI 与准确率相关**：Task C 中 LI 与 LOSO 准确率存在趋势相关（r={r_all:.3f}, p={p_all:.4f}），")
            lines.append(f"   提示保留较好侧化的患者更受益于功能侧+空间对齐策略。")
        else:
            lines.append(f"4. **LI 与准确率无显著相关**：Task C 中 LI 与 LOSO 准确率的相关不显著（r={r_all:.3f}, p={p_all:.4f}）。")
            lines.append(f"   提示简单的 C3/C4 功率侧化可能不足以解释个体解码差异，")
            lines.append(f"   需要更精细的通道选择或多频段分析。")
    lines.append("")

    lines.append("---")
    lines.append("*Generated by scripts/run_stratified_analysis.py*")

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Saved report: {out_path}")


def main():
    print("=" * 70)
    print("XWStroke Stratified Deep Analysis")
    print("=" * 70)

    # Paths
    res_a_path = Path('experiments/XWStroke_Full50_LR/results/results.json')
    res_c_path = Path('experiments/XWStroke_Full50_LOSO_Affected_Aligned/results/results.json')
    participants_path = Path('src/datasets/21679035/participants.tsv')
    out_dir = Path('results/stratified')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    print("\n[1/6] Loading LOSO results...")
    res_a = load_results(res_a_path)
    res_c = load_results(res_c_path)
    subjects = [r['test_subject_id'] for r in res_a['subjects']]
    acc_a = np.array([r['test_acc'] for r in res_a['subjects']])
    acc_c = np.array([r['test_acc'] for r in res_c['subjects']])

    # Load participants
    print("[2/6] Loading clinical data...")
    participants = pd.read_csv(participants_path, sep='\t')
    participants['subject_id'] = participants['Participant_ID'].str.replace('sub-', '').astype(int)

    # Merge
    df = pd.DataFrame({
        'subject_id': subjects,
        'acc_a': acc_a,
        'acc_c': acc_c,
        'diff': acc_c - acc_a,
    })
    df = df.merge(participants, on='subject_id', how='left')

    # Stratification: Stroke Location
    print("[3/6] Classifying stroke locations...")
    df['StrokeLocation_Category'] = df['StrokeLocation'].apply(classify_stroke_location)
    print("  Distribution:", dict(Counter(df['StrokeLocation_Category'])))

    # Stratification: Duration
    df['Duration_Category'] = df['Duration'].apply(lambda x: 'Acute(≤3mo)' if x <= 3 else 'Chronic(>3mo)')
    print("  Duration:", dict(Counter(df['Duration_Category'])))

    # Compute Lateralization Index for all subjects
    print("[4/6] Computing Lateralization Index (this may take a minute)...")
    from core.config import Config
    dataset_info = Config.fromfile('configs/dataset/XWStroke_affected_aligned.yaml').to_dict()

    li_records = []
    for sid in subjects:
        li_data = compute_lateralization_index(sid, dataset_info)
        if li_data is not None:
            li_records.append({'subject_id': sid, **li_data})
        else:
            li_records.append({'subject_id': sid, 'li_affected': np.nan, 'li_unaffected': np.nan,
                               'p_c3_aff': np.nan, 'p_c4_aff': np.nan,
                               'p_c3_unaff': np.nan, 'p_c4_unaff': np.nan,
                               'paralysis_side': None})

    df_li = pd.DataFrame(li_records)
    df = df.merge(df_li, on='subject_id', how='left')

    # Drop subjects with failed LI computation
    df_valid = df.dropna(subset=['li_affected'])
    print(f"  LI computed for {len(df_valid)}/50 subjects")
    print(f"  Mean LI (affected): {df_valid['li_affected'].mean():.3f} ± {df_valid['li_affected'].std():.3f}")

    # Correlation analysis
    print("[5/6] Correlation analysis...")
    corr_results = {}
    for task_col, task_name in [('acc_a', 'Task A'), ('acc_c', 'Task C')]:
        for side in ['left', 'right', 'all']:
            if side == 'all':
                sub = df_valid
            else:
                sub = df_valid[df_valid['ParalysisSide'] == side]
            if len(sub) < 3:
                continue
            r, p = stats.pearsonr(sub['li_affected'], sub[task_col])
            corr_results[f'{task_col}_{side}'] = (r, p)
            print(f"  {task_name} {side}: r={r:.3f}, p={p:.4f}")

    # Visualizations
    print("[6/6] Generating visualizations...")

    plot_stratum_boxplot(df, 'StrokeLocation_Category',
                         'LOSO Accuracy by Stroke Location Category',
                         out_dir / 'fig_stratum_stroke_location.png')

    plot_stratum_boxplot(df, 'Duration_Category',
                         'LOSO Accuracy by Disease Duration',
                         out_dir / 'fig_stratum_duration.png')

    plot_li_distribution(df_valid, out_dir / 'fig_li_distribution.png')

    if len(df_valid) >= 10:
        plot_scatter_li_vs_accuracy(df_valid, out_dir / 'fig_li_vs_accuracy.png')

    # Generate report
    print("\n[Report] Generating stratified analysis report...")
    generate_report(df_valid, corr_results, out_dir / 'STRATIFIED_ANALYSIS_REPORT.md')

    # Save detailed CSV
    df[['subject_id', 'ParalysisSide', 'StrokeLocation', 'StrokeLocation_Category',
        'Duration', 'Duration_Category', 'NIHSS', 'li_affected', 'li_unaffected',
        'acc_a', 'acc_c', 'diff']].to_csv(
        out_dir / 'stratified_analysis_detailed.csv', index=False, encoding='utf-8-sig')

    print("\n" + "=" * 70)
    print("Stratified Analysis Complete!")
    print("=" * 70)
    print(f"Output files in: {out_dir}")
    print(f"  - fig_stratum_stroke_location.png")
    print(f"  - fig_stratum_duration.png")
    print(f"  - fig_li_distribution.png")
    print(f"  - fig_li_vs_accuracy.png")
    print(f"  - STRATIFIED_ANALYSIS_REPORT.md")
    print(f"  - stratified_analysis_detailed.csv")


if __name__ == '__main__':
    main()
