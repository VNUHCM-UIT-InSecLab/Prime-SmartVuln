#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive metrics visualization and analysis script
Generates 5 charts: 3 individual metrics + 1 weighted avg combined + 1 macro avg combined
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================================
# DATA DEFINITION
# ============================================================================

# Support values for weighted averaging (from Table 10)
support = {
    'Other': 6091,
    'Access Control': 476,
    'Arithmetic': 5755,
    'Denial Service': 1268,
    'Front Running': 1018,
    'Reentrancy': 3798,
    'Time Manipulation': 458,
    'Unchecked Calls': 4039
}
total_support = sum(support.values())

# Table 10 - MatiVuln fusion64
table10 = {
    'Other': {'Precision': 0.9365, 'Recall': 0.9343, 'F1': 0.9354},
    'Access Control': {'Precision': 0.7684, 'Recall': 0.5714, 'F1': 0.6554},
    'Arithmetic': {'Precision': 0.9515, 'Recall': 0.9654, 'F1': 0.9584},
    'Denial Service': {'Precision': 0.9125, 'Recall': 0.8880, 'F1': 0.9001},
    'Front Running': {'Precision': 0.7855, 'Recall': 0.7338, 'F1': 0.7588},
    'Reentrancy': {'Precision': 0.9421, 'Recall': 0.8647, 'F1': 0.9017},
    'Time Manipulation': {'Precision': 0.8476, 'Recall': 0.7773, 'F1': 0.8109},
    'Unchecked Calls': {'Precision': 0.9513, 'Recall': 0.9485, 'F1': 0.9499}
}

# Table 9 - MatiVuln fusion8
table9 = {
    'Other': {'Precision': 0.9294, 'Recall': 0.9381, 'F1': 0.9337},
    'Access Control': {'Precision': 0.7041, 'Recall': 0.5798, 'F1': 0.6359},
    'Arithmetic': {'Precision': 0.9435, 'Recall': 0.9685, 'F1': 0.9558},
    'Denial Service': {'Precision': 0.8788, 'Recall': 0.8920, 'F1': 0.8853},
    'Front Running': {'Precision': 0.7954, 'Recall': 0.6837, 'F1': 0.7353},
    'Reentrancy': {'Precision': 0.9196, 'Recall': 0.8791, 'F1': 0.8989},
    'Time Manipulation': {'Precision': 0.7977, 'Recall': 0.7664, 'F1': 0.7817},
    'Unchecked Calls': {'Precision': 0.9530, 'Recall': 0.9334, 'F1': 0.9431}
}

# Deng's method
dengs = {
    'Other': {'Precision': 0.8504, 'Recall': 0.7796, 'F1': 0.8135},
    'Access Control': {'Precision': 0.9167, 'Recall': 0.2088, 'F1': 0.3401},
    'Arithmetic': {'Precision': 0.9188, 'Recall': 0.9727, 'F1': 0.945},
    'Denial Service': {'Precision': 0.8629, 'Recall': 0.9311, 'F1': 0.8957},
    'Front Running': {'Precision': 0.7417, 'Recall': 0.6083, 'F1': 0.6684},
    'Reentrancy': {'Precision': 0.7900, 'Recall': 0.6392, 'F1': 0.7066},
    'Time Manipulation': {'Precision': 0.7527, 'Recall': 0.2961, 'F1': 0.4251},
    'Unchecked Calls': {'Precision': 0.7756, 'Recall': 0.737, 'F1': 0.7558}
}

# Sendner's method
sendners = {
    'Other': {'Precision': 0.02, 'Recall': 0.50, 'F1': 0.29},
    'Access Control': {'Precision': 0.46, 'Recall': 0.50, 'F1': 0.48},
    'Arithmetic': {'Precision': 0.01, 'Recall': 0.50, 'F1': 0.17},
    'Denial Service': {'Precision': 0.37, 'Recall': 0.50, 'F1': 0.43},
    'Front Running': {'Precision': 0.41, 'Recall': 0.50, 'F1': 0.45},
    'Reentrancy': {'Precision': 0.35, 'Recall': 0.50, 'F1': 0.41},
    'Time Manipulation': {'Precision': 0.46, 'Recall': 0.50, 'F1': 0.48},
    'Unchecked Calls': {'Precision': 0.34, 'Recall': 0.50, 'F1': 0.41}
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_weighted_avg(data, metric):
    """Calculate weighted average for a specific metric."""
    total = 0
    for label, values in data.items():
        total += values[metric] * support[label]
    return total / total_support

def calculate_macro_avg(data, metric):
    """Calculate macro average for a specific metric."""
    values = [v[metric] for v in data.values()]
    return np.mean(values)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Define methods and labels
    methods = {
        'MatiVuln fusion64': table10,
        'MatiVuln fusion8': table9,
        "Deng's Method": dengs,
        "Sendner's Method": sendners
    }

    labels = list(table10.keys())
    label_short = ['Other', 'Acc. Ctrl', 'Arith.', 'Denial', 'Front Run', 'Reentrancy', 'Time Manip.', 'Unchecked']
    methods_short = ['MatiVuln fusion_64', 'MatiVuln fusion_8', "Deng's", "Sendner's"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    metrics = ['Precision', 'Recall', 'F1']

    # ========================================================================
    # STEP 1: CALCULATE WEIGHTED AVG AND MACRO AVG
    # ========================================================================
    print("=" * 100)
    print("CALCULATING WEIGHTED AVG AND MACRO AVG FOR DENG'S AND SENDNER'S METHODS")
    print("=" * 100)

    results = {}
    for method_name, data in methods.items():
        results[method_name] = {}
        print(f"\n{method_name}:")
        for metric in metrics:
            weighted = calculate_weighted_avg(data, metric)
            macro = calculate_macro_avg(data, metric)
            results[method_name][metric] = {'weighted': weighted, 'macro': macro}
            print(f"  {metric:12s} - Weighted Avg: {weighted:.4f}, Macro Avg: {macro:.4f}")

    # ========================================================================
    # STEP 2: CREATE 3 CHARTS FOR INDIVIDUAL METRICS
    # ========================================================================
    print("\n" + "=" * 100)
    print("CREATING 3 CHARTS FOR INDIVIDUAL METRICS (PRECISION, RECALL, F1)")
    print("=" * 100)

    method_list = list(methods.values())
    method_names_list = list(methods.keys())

    for metric_idx, metric in enumerate(metrics):
        fig, ax = plt.subplots(figsize=(14, 6))

        x = np.arange(len(labels))
        width = 0.2

        for i, (method_name, method_data) in enumerate(zip(method_names_list, method_list)):
            values = [method_data[label][metric] for label in labels]
            ax.bar(x + (i - 1.5) * width, values, width, label=methods_short[i], color=colors[i], alpha=0.8)

        ax.set_xlabel('Vulnerability Labels', fontweight='bold', fontsize=11)
        ax.set_ylabel('Score', fontweight='bold', fontsize=11)
        ax.set_title(f'{metric} Score Comparison Across All Methods', fontweight='bold', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(label_short, rotation=45, ha='right', fontsize=10)
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.05])

        filename = f'01_metric_{metric.lower()}.png'
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {filename}")
        plt.close()

    # ========================================================================
    # STEP 3: CREATE COMBINED WEIGHTED AVERAGE CHART (3 SUBPLOTS)
    # ========================================================================
    print("\n" + "=" * 100)
    print("CREATING COMBINED WEIGHTED AVERAGE CHART (3 SUBPLOTS)")
    print("=" * 100)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        weighted_avgs = [results[method][metric]['weighted'] for method in method_names_list]
        x = np.arange(len(method_names_list))
        bars = ax.bar(x, weighted_avgs, color=colors, alpha=0.8, width=0.6)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_ylabel('Weighted Average Score', fontweight='bold', fontsize=10)
        ax.set_title(f'{metric}', fontweight='bold', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(['fusion_64', 'fusion_8', "Deng's", "Sendner's"], fontsize=9, rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.0])

    plt.suptitle('Weighted Average Metrics Comparison', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    filename = '02_weighted_avg_combined.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {filename}")
    plt.close()

    # ========================================================================
    # STEP 4: CREATE COMBINED MACRO AVERAGE CHART (3 SUBPLOTS)
    # ========================================================================
    print("\n" + "=" * 100)
    print("CREATING COMBINED MACRO AVERAGE CHART (3 SUBPLOTS)")
    print("=" * 100)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        macro_avgs = [results[method][metric]['macro'] for method in method_names_list]
        x = np.arange(len(method_names_list))
        bars = ax.bar(x, macro_avgs, color=colors, alpha=0.8, width=0.6)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_ylabel('Macro Average Score', fontweight='bold', fontsize=10)
        ax.set_title(f'{metric}', fontweight='bold', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(['fusion_64', 'fusion_8', "Deng's", "Sendner's"], fontsize=9, rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.0])

    plt.suptitle('Macro Average Metrics Comparison', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    filename = '03_macro_avg_combined.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {filename}")
    plt.close()

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 100)
    print("COMPLETED! All 5 charts have been saved successfully.")
    print("=" * 100)
    print("\nGenerated files:")
    print("  Individual Metrics (3 files, rectangular 14x6):")
    print("    - 01_metric_precision.png")
    print("    - 01_metric_recall.png")
    print("    - 01_metric_f1.png")
    print("  Combined Charts (2 files, 15x4.5):")
    print("    - 02_weighted_avg_combined.png")
    print("    - 03_macro_avg_combined.png")
    print("\n" + "=" * 100)
