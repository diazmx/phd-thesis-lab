#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ABAC Policy Statistical Analysis
---------------------------------
Reproducible statistical analysis pipeline for automatically generated ABAC policies.

Author: <Your Name>
Description:
    Computes structural, diversity, redundancy, and attribute-level metrics
    for ABAC rules generated via automatic mining.

Input format:
    Each line:
    [["id_com", "X"], [[attr1, val1], [attr2, val2], ...]]

Only the second tuple is analyzed.
"""

import ast
import os
import json
import argparse
import itertools
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from scipy.stats import entropy

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams["figure.dpi"] = 300

# ==========================================================
# 1. LOADING AND PREPROCESSING
# ==========================================================

def load_rules(filepath):
    rules = []

    with open(filepath, 'r') as f:
        for line in f:
            parsed = ast.literal_eval(line.strip())
            rule_part = parsed[1]
            rule = tuple(sorted((attr, val) for attr, val in rule_part))
            rules.append(rule)

    return rules


# ==========================================================
# 2. BASIC STRUCTURAL METRICS
# ==========================================================

def rule_length_stats(rules):
    lengths = np.array([len(r) for r in rules])

    return {
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "std": float(np.std(lengths)),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "p25": float(np.percentile(lengths, 25)),
        "p75": float(np.percentile(lengths, 75)),
        "p90": float(np.percentile(lengths, 90)),
    }, lengths


def total_conditions(rules):
    return sum(len(r) for r in rules)


def compute_wsc(rules, weight=1):
    return sum(weight * len(r) for r in rules)


# ==========================================================
# 3. ATTRIBUTE-LEVEL METRICS
# ==========================================================

def attribute_statistics(rules):

    attr_frequency = Counter()
    attr_coverage = Counter()
    attr_domains = defaultdict(set)
    attr_values = defaultdict(list)

    for rule in rules:
        attrs_in_rule = set()

        for attr, val in rule:
            attr_frequency[attr] += 1
            attr_domains[attr].add(val)
            attr_values[attr].append(val)
            attrs_in_rule.add(attr)

        for attr in attrs_in_rule:
            attr_coverage[attr] += 1

    attr_entropy = {}

    for attr, values in attr_values.items():
        counts = np.array(list(Counter(values).values()))
        probs = counts / counts.sum()
        attr_entropy[attr] = float(entropy(probs))

    return attr_frequency, attr_coverage, attr_domains, attr_entropy


# ==========================================================
# 4. DIVERSITY METRICS
# ==========================================================

def unique_rule_ratio(rules):
    return len(set(rules)) / len(rules)


def average_jaccard(rules):

    sims = []
    rule_sets = [set(r) for r in rules]

    for r1, r2 in itertools.combinations(rule_sets, 2):
        inter = len(r1 & r2)
        union = len(r1 | r2)
        sims.append(inter / union if union > 0 else 0)

    return float(np.mean(sims)) if sims else 0


def structural_signatures(rules):
    signatures = [
        tuple(sorted(attr for attr, _ in rule))
        for rule in rules
    ]
    return signatures


def heterogeneity_index(rules):
    signatures = structural_signatures(rules)
    return len(set(signatures)) / len(rules)


# ==========================================================
# 5. REDUNDANCY (SUBSUMPTION)
# ==========================================================

def count_subsumed_rules(rules):

    subsumed = 0
    rule_sets = [set(r) for r in rules]

    for i, r1 in enumerate(rule_sets):
        for j, r2 in enumerate(rule_sets):
            if i != j and r1.issubset(r2):
                subsumed += 1
                break

    return subsumed


# ==========================================================
# 6. CO-OCCURRENCE MATRIX
# ==========================================================

def attribute_cooccurrence(rules):

    co_matrix = Counter()

    for rule in rules:
        attrs = sorted(attr for attr, _ in rule)
        for a1, a2 in itertools.combinations(attrs, 2):
            co_matrix[(a1, a2)] += 1

    return co_matrix

def plot_rule_length(lengths, output_dir):

    plt.figure()
    sns.histplot(lengths, bins=15)
    plt.xlabel("Rule Length")
    plt.ylabel("Frequency")
    plt.title("Distribution of Rule Lengths")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rule_length_histogram.png"))
    plt.close()

    # Boxplot
    plt.figure()
    sns.boxplot(x=lengths)
    plt.xlabel("Rule Length")
    plt.title("Rule Length Boxplot")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rule_length_boxplot.png"))
    plt.close()

def plot_attribute_frequency(attr_freq, output_dir, top_k=15):

    sorted_items = sorted(attr_freq.items(), key=lambda x: x[1], reverse=True)[:top_k]
    attrs, values = zip(*sorted_items)

    plt.figure(figsize=(8,5))
    sns.barplot(x=list(values), y=list(attrs))
    plt.xlabel("Frequency")
    plt.ylabel("Attribute")
    plt.title(f"Top {top_k} Most Frequent Attributes")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "attribute_frequency.png"))
    plt.close()

def plot_attribute_coverage(attr_cov, num_rules, output_dir):

    coverage_ratio = {k: v/num_rules for k,v in attr_cov.items()}
    sorted_items = sorted(coverage_ratio.items(), key=lambda x: x[1], reverse=True)

    attrs, values = zip(*sorted_items)

    plt.figure(figsize=(8,5))
    sns.barplot(x=list(values), y=list(attrs))
    plt.xlabel("Coverage Ratio")
    plt.ylabel("Attribute")
    plt.title("Attribute Coverage Ratio")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "attribute_coverage.png"))
    plt.close()
    

def plot_attribute_entropy(attr_entropy, output_dir):

    sorted_items = sorted(attr_entropy.items(), key=lambda x: x[1], reverse=True)
    attrs, values = zip(*sorted_items)

    plt.figure(figsize=(8,5))
    sns.barplot(x=list(values), y=list(attrs))
    plt.xlabel("Entropy")
    plt.ylabel("Attribute")
    plt.title("Attribute Entropy")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "attribute_entropy.png"))
    plt.close()

def plot_cooccurrence(co_matrix, output_dir): 

    if not co_matrix:
        return

    attrs = sorted(set(a for pair in co_matrix.keys() for a in pair))
    matrix = pd.DataFrame(0, index=attrs, columns=attrs)

    for (a1, a2), count in co_matrix.items():
        matrix.loc[a1, a2] = count
        matrix.loc[a2, a1] = count

    plt.figure(figsize=(10,8))
    sns.heatmap(matrix, cmap="Blues")
    plt.title("Attribute Co-occurrence Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cooccurrence_heatmap.png"))
    plt.close()


# ==========================================================
# 7. MAIN PIPELINE
# ==========================================================

def run_analysis(input_path, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    rules = load_rules(input_path)

    # ----- Structural Metrics -----
    length_stats, lengths = rule_length_stats(rules)

    summary_metrics = {
        "num_rules": len(rules),
        "total_conditions": total_conditions(rules),
        "wsc": compute_wsc(rules),
        "unique_rule_ratio": unique_rule_ratio(rules),
        "avg_jaccard_similarity": average_jaccard(rules),
        "heterogeneity_index": heterogeneity_index(rules),
        "subsumed_rules": count_subsumed_rules(rules),
        "rule_length_stats": length_stats
    }

    # Save summary metrics
    with open(os.path.join(output_dir, "summary_metrics.json"), "w") as f:
        json.dump(summary_metrics, f, indent=4)

    # Save rule length distribution
    pd.DataFrame({"rule_length": lengths}).to_csv(
        os.path.join(output_dir, "rule_length_distribution.csv"),
        index=False
    )

    # ----- Attribute Metrics -----
    attr_freq, attr_cov, attr_domains, attr_entropy = attribute_statistics(rules)

    attr_df = pd.DataFrame({
        "frequency": attr_freq,
        "coverage": attr_cov,
        "domain_size": {k: len(v) for k, v in attr_domains.items()},
        "entropy": attr_entropy
    }).fillna(0)

    attr_df.to_csv(os.path.join(output_dir, "attribute_stats.csv"))

    # ----- Co-occurrence -----
    co_matrix = attribute_cooccurrence(rules)

    co_df = pd.DataFrame([
        {"attr1": a1, "attr2": a2, "count": count}
        for (a1, a2), count in co_matrix.items()
    ])

    co_df.to_csv(os.path.join(output_dir, "cooccurrence_matrix.csv"), index=False)

    print("Analysis completed successfully.")
    print(f"Results saved in: {output_dir}")

        # ===== Generate Plots =====
    plot_rule_length(lengths, output_dir)
    plot_attribute_frequency(attr_freq, output_dir)
    plot_attribute_coverage(attr_cov, len(rules), output_dir)
    plot_attribute_entropy(attr_entropy, output_dir)
    plot_cooccurrence(co_matrix, output_dir)


# ==========================================================
# 8. CLI ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Statistical Analysis for ABAC Policies"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to ABAC rules file"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Directory to store results"
    )

    args = parser.parse_args()

    run_analysis(args.input, args.output)
    
