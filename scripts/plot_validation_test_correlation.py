"""
Plot validation-vs-test accuracy correlation for clinical-nearest validation
pipeline selection (Fig. 5 of the IEEE BHI paper).

Inputs:
    experiments/XWStroke_ValBased_ClinicalNearest/results/results.json
Outputs:
    results/figures/fig5_validation_test_correlation.png
    results/figures/fig5_validation_test_correlation.pdf
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats


def main():
    result_path = 'experiments/XWStroke_ValBased_ClinicalNearest/results/results.json'
    with open(result_path) as f:
        data = json.load(f)

    val_accs, test_accs, pipeline_labels = [], [], []
    for subj in data['subjects']:
        for det in subj['pipeline_validation_details']:
            val_accs.append(det['val_acc'])
            test_accs.append(det['test_acc'])
            pipeline_labels.append(det['pipeline'])

    val_accs = np.array(val_accs)
    test_accs = np.array(test_accs)
    pipeline_labels = np.array(pipeline_labels)

    r, p = stats.pearsonr(val_accs, test_accs)
    print(f"Overall validation-test correlation: r={r:.3f}, p={p:.4f}, N={len(val_accs)}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    palette = {
        'LR_noEA': '#1f77b4', 'LR_EA': '#ff7f0e',
        'Aff_noEA': '#2ca02c', 'Aff_EA': '#d62728'
    }

    # (a) Scatter by pipeline
    ax = axes[0]
    for pipe in ['LR_noEA', 'LR_EA', 'Aff_noEA', 'Aff_EA']:
        mask = pipeline_labels == pipe
        ax.scatter(val_accs[mask] * 100, test_accs[mask] * 100,
                   c=palette[pipe], label=pipe, alpha=0.6, s=30)
    ax.plot([30, 80], [30, 80], 'k--', alpha=0.3, label='y=x')
    ax.set_xlabel('Validation accuracy (%)')
    ax.set_ylabel('Test accuracy (%)')
    ax.set_title(f'Validation vs. test accuracy\n(r={r:.3f}, p={p:.4f})')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (b) Per-pipeline correlation
    ax = axes[1]
    pipes = ['LR_noEA', 'LR_EA', 'Aff_noEA', 'Aff_EA']
    pipe_r, pipe_p = [], []
    for pipe in pipes:
        mask = pipeline_labels == pipe
        r_p, p_p = stats.pearsonr(val_accs[mask], test_accs[mask])
        pipe_r.append(r_p)
        pipe_p.append(p_p)
        print(f"{pipe}: r={r_p:.3f}, p={p_p:.4f}")

    bars = ax.bar(pipes, pipe_r,
                  color=[palette[p] for p in pipes],
                  alpha=0.7, edgecolor='black')
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_ylabel('Pearson r')
    ax.set_title('Validation-test correlation by pipeline')
    ax.set_ylim(-0.5, 0.8)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, r_p in zip(bars, pipe_r):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03,
                f'{r_p:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    os.makedirs('results/figures', exist_ok=True)
    plt.savefig('results/figures/fig5_validation_test_correlation.png', dpi=300)
    plt.savefig('results/figures/fig5_validation_test_correlation.pdf', dpi=300)
    plt.close()
    print('Saved Fig. 5 to results/figures/fig5_validation_test_correlation.{png,pdf}')


if __name__ == '__main__':
    main()
