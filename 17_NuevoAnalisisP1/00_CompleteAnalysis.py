#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ast
import argparse
import itertools
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams["figure.dpi"] = 300

DATASET_NAME = "K_IoT"


# ==========================================================
# 1. LOAD DATA
# ==========================================================

def load_rules(filepath):
    rules = []
    with open(filepath, "r") as f:
        for line in f:
            parsed = ast.literal_eval(line.strip())
            rule = tuple(sorted((attr, val) for attr, val in parsed[1]))
            rules.append(rule)
    return rules


def load_dataset(filepath):
    df = pd.read_csv(filepath)
    df = df[df["ACTION"] == 1]  # solo accesos permitidos
    return df.to_dict(orient="records")


def load_graph_access(filepath):

    df = pd.read_csv(filepath)

    # crear vector alineado por rule_id
    access = {}

    for _, row in df.iterrows():
        access[int(row["rule_id"])] = row["granted_accesses"]

    return access

def build_access_array(rules, access_dict):

    access_array = []

    for i in range(len(rules)):
        access_array.append(access_dict.get(i, 0))

    return np.array(access_array)


# ==========================================================
# 2. SIZE & COMPLEXITY
# ==========================================================

def rule_size_stats(rules):
    sizes = np.array([len(r) for r in rules])
    return sizes


def rule_density(rules):
    attributes = set(a for r in rules for a, _ in r)
    total_conditions = sum(len(r) for r in rules)
    return total_conditions / (len(rules) * len(attributes))

def compute_wsc(rules):
    return sum([len(r) for r in rules])


# ==========================================================
# 3. ATTRIBUTE STATS
# ==========================================================

def attribute_stats(rules):
    attr_freq = Counter()
    attr_val_freq = Counter()

    total_rules = len(rules)

    for r in rules:
        attrs_seen = set()

        for attr, val in r:
            attr_val_freq[(attr, val)] += 1
            attrs_seen.add(attr)

        for attr in attrs_seen:
            attr_freq[attr] += 1

    # convertir a frecuencia relativa
    attr_freq_rel = {k: v / total_rules for k, v in attr_freq.items()}
    attr_val_freq_rel = {k: v / total_rules for k, v in attr_val_freq.items()}

    return attr_freq_rel, attr_val_freq_rel

def attribute_value_dataset_freq(dataset):
    counter = Counter()

    for row in dataset:
        for attr, val in row.items():
            if attr == "ACTION":
                continue
            counter[(attr, val)] += 1

    total = len(dataset)

    return {k: v / total for k, v in counter.items()}

def export_attribute_value_comparison(attr_val_rules, dataset, output_file="attribute_value_comparison.csv"):

    dataset_freq = attribute_value_dataset_freq(dataset)

    rows = []

    for key, rule_freq in attr_val_rules.items():

        if rule_freq == 0:
            continue

        data_freq = dataset_freq.get(key, 0)

        rows.append({
            "attribute": key[0],
            "value": key[1],
            "rule_frequency": rule_freq,
            "dataset_frequency": data_freq
        })

    df = pd.DataFrame(rows)

    df["abs_diff"] = abs(df["rule_frequency"] - df["dataset_frequency"])

    # ordenar de mayor a menor por frecuencia en reglas
    df = df.sort_values(by="rule_frequency", ascending=False)

    df.to_csv(output_file, index=False)

    plot_rule_vs_dataset(df)
    plot_abs_diff(df)

    print(f"Saved: {output_file}")


def generate_rule_importance_csv(rules, access_counts, attr_val_df, output_file="rule_importance.csv"):

    # crear lookup rápido (attr, val) -> frecuencia
    freq_lookup = {
        (row["attribute"], row["value"]): row["rule_frequency"]
        for _, row in attr_val_df.iterrows()
    }

    results = []

    for i, rule in enumerate(rules):

        num_tuples = len(rule)
        access = access_counts[i] if access_counts is not None else 0

        conditions_importance = []
        total_importance = 0

        for attr, val in rule:

            freq = freq_lookup.get((attr, val), 0)

            conditions_importance.append([[attr, val], round(freq, 4)])
            total_importance += freq

        # promedio (tu definición)
        relative_total_importance = total_importance / num_tuples if num_tuples > 0 else 0

        results.append({
            "rule_id": i,
            "number_of_access": access,
            "num_tuples": num_tuples,
            "total_importance": round(total_importance, 4),
            "relative_total_importance": round(relative_total_importance, 4),
            "conditions_importance": str(conditions_importance)
        })

    df = pd.DataFrame(results)
    df["corr"] = df["number_of_access"].corr(df["relative_total_importance"])

    df.to_csv(output_file, index=False)

    plot_importance_vs_access(df)

    print(f"Saved: {output_file}")

def generate_rule_importance_graph_csv(
    rules,
    graph_access_file,
    attr_val_df,
    output_file="rule_importance_graph.csv"
):

    # ==========================================================
    # 1. Cargar graph access
    # ==========================================================
    df_graph = pd.read_csv(graph_access_file)

    # crear lookup rule_id -> granted_accesses
    graph_access_lookup = {
        int(row["rule_id"]): row["granted_accesses"]
        for _, row in df_graph.iterrows()
    }

    # ==========================================================
    # 2. Lookup de frecuencias (atributo-valor)
    # ==========================================================
    freq_lookup = {
        (row["attribute"], row["value"]): row["rule_frequency"]
        for _, row in attr_val_df.iterrows()
    }

    results = []

    # ==========================================================
    # 3. Construcción de métricas por regla
    # ==========================================================
    for i, rule in enumerate(rules):

        num_tuples = len(rule)

        access = graph_access_lookup.get(i, 0)

        conditions_importance = []
        total_importance = 0

        for attr, val in rule:

            freq = freq_lookup.get((attr, val), 0)

            conditions_importance.append([[attr, val], round(freq, 4)])
            total_importance += freq

        relative_total_importance = (
            total_importance / num_tuples if num_tuples > 0 else 0
        )

        results.append({
            "rule_id": i,
            "graph_access": access,
            "num_tuples": num_tuples,
            "total_importance": round(total_importance, 4),
            "relative_total_importance": round(relative_total_importance, 4),
            "conditions_importance": str(conditions_importance)
        })

    df = pd.DataFrame(results)

    # ==========================================================
    # 4. Correlación (correcta)
    # ==========================================================
    corr = df["graph_access"].corr(df["relative_total_importance"])

    print("\n===== GRAPH IMPORTANCE CORRELATION =====")
    print("Correlation (importance vs graph access):", corr)

    # ==========================================================
    # 5. Guardar CSV
    # ==========================================================
    df.to_csv(output_file, index=False)

    # ==========================================================
    # 6. Gráfica adaptada
    # ==========================================================
    plot_importance_vs_graph_access(df)

    print(f"Saved: {output_file}")

# ==========================================================
# 4. JACCARD
# ==========================================================

def jaccard_stats(rules):
    sims = []
    sets = [set(r) for r in rules]

    for r1, r2 in itertools.combinations(sets, 2):
        inter = len(r1 & r2)
        union = len(r1 | r2)
        sims.append(inter / union if union else 0)

    return np.array(sims)


# ==========================================================
# 5. ACCESS GRANTED PER RULE
# ==========================================================

def access_per_rule(rules, dataset):

    results = []

    for r in rules:
        count = 0
        for row in dataset:
            match = True
            for attr, val in r:
                if attr not in row or row[attr] != val:
                    match = False
                    break
            if match:
                count += 1
        results.append(count)

    return np.array(results)


# ==========================================================
# 6. SUBSUMPTION
# ==========================================================

def subsumption(rules):
    rule_sets = [set(r) for r in rules]
    subsumed_flags = []

    for i, r1 in enumerate(rule_sets):
        is_subsumed = False
        for j, r2 in enumerate(rule_sets):
            if i != j and r1.issubset(r2):
                is_subsumed = True
                break
        subsumed_flags.append(is_subsumed)

    return np.array(subsumed_flags)

def export_subsumed_rules(rules, subsumed_flags, access_counts, type_access, output_file="subsumed_rules_analysis.csv"):

    rows = []

    for i, rule in enumerate(rules):

        rows.append({
            "rule_id": i,
            "is_subsumed": subsumed_flags[i],
            "num_tuples": len(rule),
            "number_of_access": access_counts[i] if access_counts is not None else 0,
            "rule": str(rule)
        })

    df = pd.DataFrame(rows)

    # ordenar por accesos descendente
    df = df.sort_values(by="number_of_access", ascending=False)

    if type_access == 0:
        df.to_csv(output_file+"_flat", index=False)
    else:
        df.to_csv(output_file+"_graph", index=False)

    print(f"Saved: {output_file}")

# ==========================================================
# 7. PLOTS
# ==========================================================

def plot_rule_size(sizes):
    plt.figure()
    bins_to_plot = [2,3,4,5,6,7,8,9,10]
    sns.histplot(sizes, bins=bins_to_plot, discrete=True, shrink=0.5)
    plt.xlabel("Number of attributes per rule")
    plt.ylabel("Number of rules")
    plt.title("Rule Size Distribution - " + DATASET_NAME)
    plt.savefig(DATASET_NAME + "_rule_size_hist.png")
    plt.close()


def plot_attribute_frequency(attr_freq_rel):
    items = sorted(attr_freq_rel.items(), key=lambda x: x[1], reverse=True)
    attrs, vals = zip(*items)

    plt.figure(figsize=(8,5))
    sns.barplot(x=list(vals), y=list(attrs))
    plt.xlabel("Relative Frequency")
    plt.ylabel("Attribute")
    plt.title("Attribute Relative Frequency - "+ DATASET_NAME)
    plt.xlim(0,1)
    plt.tight_layout()
    plt.savefig(DATASET_NAME + "_attribute_frequency.png")
    plt.close()

def plot_rule_vs_dataset(df):

    plt.figure(figsize=(6,6))

    plt.scatter(df["dataset_frequency"], df["rule_frequency"], alpha=0.6)

    # línea ideal
    #max_val = max(df["dataset_frequency"].max(), df["rule_frequency"].max())
    #plt.plot([0, max_val], [0, max_val], linestyle="--")

    plt.xlabel("Dataset Frequency")
    plt.ylabel("Rule Frequency")
    plt.title("Rule vs Dataset Frequency - " + DATASET_NAME)

    plt.tight_layout()
    plt.savefig(DATASET_NAME + "_rule_vs_dataset_scatter.png")
    plt.close()

def plot_abs_diff(df):

    plt.figure()

    sns.histplot(df["abs_diff"], bins=20, stat="probability")

    plt.xlabel("Absolute Difference")
    plt.title("Distribution of Absolute Difference in Rule-Dataset Frequency - " + DATASET_NAME)

    plt.tight_layout()
    plt.savefig(DATASET_NAME + "_abs_diff_distribution.png")
    plt.close()

def plot_jaccard(sims):
    plt.figure()
    sns.histplot(sims, bins=20, stat="probability")
    plt.xlabel("Jaccard Similarity Value")
    plt.title("Jaccard Similarity Distribution - " + DATASET_NAME)
    plt.savefig(DATASET_NAME + "_jaccard_distribution.png")
    plt.close()


def plot_access(access_counts):
    plt.figure()
    sns.histplot(access_counts, bins=20, stat="probability")
    plt.xlabel("Number of access requests permited")
    plt.ylabel("Number of rules")
    plt.title("Access Granted per Rule - " + DATASET_NAME)
    plt.savefig(DATASET_NAME + "_access_per_rule.png")
    plt.close()

def plot_access_dos(access_counts):
    plt.figure()
    sns.histplot(access_counts, bins=20, stat="probability")
    plt.xlabel("Number of access requests permited")
    plt.ylabel("Number of rules")
    plt.title("Access Granted per Rule in Graph - " + DATASET_NAME)
    plt.savefig(DATASET_NAME + "_access_per_rule_graph.png")
    plt.close()


def plot_subsumption(subsumed_flags):
    counts = Counter(subsumed_flags)

    plt.figure()
    plt.bar(["Not Subset", "Subset"], [counts[False], counts[True]])
    plt.title("Rule Subset - " + DATASET_NAME)
    plt.savefig(DATASET_NAME + "_subsumption.png")
    plt.close()

def plot_importance_vs_access(df):

    plt.figure()

    plt.scatter(
        df["relative_total_importance"],
        df["number_of_access"],
        alpha=0.6
    )

    plt.xlabel("Relative Importance")
    plt.ylabel("Access Granted")
    plt.title("Rule Importance vs Coverage in Dataset - " + DATASET_NAME)

    plt.tight_layout()
    plt.savefig(DATASET_NAME + "_importance_vs_access.png")
    plt.close()

def plot_subsumed_vs_access(subsumed_flags, access_counts, type_access):

    df = pd.DataFrame({
        "is_subsumed": subsumed_flags,
        "access": access_counts
    })

    plt.figure()

    sns.boxplot(x="is_subsumed", y="access", data=df)

    if type_access == 0:
        plt.xlabel("Is Subset (False=Specific, True=General)")
        plt.ylabel("Access Granted Flat")
        plt.title("Access Distribution: Subset vs Non-Subset Rules")

        plt.tight_layout()
        plt.savefig(DATASET_NAME + "_subsumed_vs_access_boxplot_flat.png")
    else:
        plt.xlabel("Is Subset (False=Specific, True=General)")
        plt.ylabel("Access Granted Graph")
        plt.title("Access Distribution: Subset vs Non-Subset Rules")

        plt.tight_layout()
        plt.savefig(DATASET_NAME + "_subsumed_vs_access_boxplot_graph.png")
    
    plt.close()

def plot_size_vs_access_subsumed(rules, subsumed_flags, access_counts, type_access):

    sizes = [len(r) for r in rules]

    df = pd.DataFrame({
        "size": sizes,
        "access": access_counts,
        "is_subset": subsumed_flags
    })

    plt.figure()

    sns.scatterplot(
        data=df,
        x="size",
        y="access",
        hue="is_subset"
    )

    plt.xlabel("Rule Size")
    plt.ylabel("Access Granted")

    if type_access == 0:
        plt.title("Rule Size vs Access Flat (Subset Highlighted)")

        plt.tight_layout()
        plt.savefig(DATASET_NAME + "_size_vs_access_subsumed_flat.png")
    else:
        plt.title("Rule Size vs Access Graph (Subset Highlighted)")

        plt.tight_layout()
        plt.savefig(DATASET_NAME + "_size_vs_access_subsumed_graph.png")
    
    plt.close()

def plot_importance_vs_graph_access(df):

    plt.figure()

    plt.scatter(
        df["relative_total_importance"],
        df["graph_access"],
        alpha=0.6
    )

    plt.xlabel("Rule Importance")
    plt.ylabel("Graph-Based Access")
    plt.title("Rule Importance vs Graph-Based Coverage")

    plt.tight_layout()
    plt.savefig(DATASET_NAME + "_importance_vs_graph_access.png")
    plt.close()

# ==========================================================
# 8. MAIN
# ==========================================================

def main(rules_path, dataset_path):

    rules = load_rules(rules_path)
    dataset = load_dataset(dataset_path)

    #graph_access_dict = load_graph_access("rule_usage.csv")
    #graph_access = build_access_array(rules, graph_access_dict)

    # Metrics
    sizes = rule_size_stats(rules)
    wsc = compute_wsc(rules)
    density = rule_density(rules)
    attr_freq_rel, attr_val_freq_rel = attribute_stats(rules)
    export_attribute_value_comparison(attr_val_freq_rel, dataset)
    jaccard = jaccard_stats(rules)
    access = access_per_rule(rules, dataset)
    attr_val_df = pd.read_csv("attribute_value_comparison.csv")
    generate_rule_importance_csv(rules, access, attr_val_df)
    #generate_rule_importance_graph_csv(rules,
    #    "rule_usage.csv",
    #    attr_val_df,
    #    output_file="rule_importance_graph.csv")
    subsumed = subsumption(rules)
    export_subsumed_rules(rules, subsumed, access, 0)
    plot_subsumed_vs_access(subsumed, access, 0)
    plot_size_vs_access_subsumed(rules, subsumed, access, 0)

    #export_subsumed_rules(rules, subsumed, graph_access, 1)
    #plot_subsumed_vs_access(subsumed, graph_access, 1)
    #plot_size_vs_access_subsumed(rules, subsumed, graph_access, 1)
    

    # Print summary
    print("\n===== SIZE =====")
    print("N° Rules:", len(rules))
    print("Mean:", np.mean(sizes))
    print("Std:", np.std(sizes))
    print("Min:", np.min(sizes))
    print("Max:", np.max(sizes))
    print("Density:", density)
    print("WSC:", wsc)

    print("\n===== JACCARD =====")
    print("Mean:", np.mean(jaccard))
    print("Std:", np.std(jaccard))

    print("\n===== ACCESS =====")
    print("Mean:", np.mean(access))
    print("Std:", np.std(access))
    print("Min:", np.min(access))
    print("Max:", np.max(access))

    print("\n===== SUBSUMPTION =====")
    print("Total:", np.sum(subsumed))
    print("Ratio:", np.mean(subsumed))

    # Save CSV
    pd.DataFrame({"rule_size": sizes}).to_csv("rule_sizes.csv", index=False)
    pd.DataFrame({"access": access}).to_csv("access_per_rule.csv", index=False)

    # Plots
    plot_rule_size(sizes)
    plot_attribute_frequency(attr_freq_rel)
    plot_jaccard(jaccard)
    plot_access(access)
    #plot_access_dos(graph_access)
    plot_subsumption(subsumed)

    print("\nPlots and CSV files generated.")


# ==========================================================
# ENTRY
# ==========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", required=True)
    parser.add_argument("--dataset", required=True)

    args = parser.parse_args()

    main(args.rules, args.dataset)