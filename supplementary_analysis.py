"""
Supplementary Analysis Script
=============================
This script performs:
1. Post-hoc Mann-Whitney U tests with all pairwise p-values (with Bonferroni correction)
2. Rejection rate analysis for RQ4
3. PR Types distribution visualization

Uses existing analyzed data from the most recent run.
"""

import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import os

def compute_ci_95(data, confidence=0.95):
    """Compute 95% confidence interval for the mean using bootstrap."""
    data = np.array(data)
    data = data[~np.isnan(data)]
    n = len(data)
    if n < 2:
        return np.nan, np.nan
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean - h, mean + h

def compute_ci_proportion(successes, total, confidence=0.95):
    """Compute 95% CI for a proportion using Wilson score interval."""
    if total == 0:
        return np.nan, np.nan
    p = successes / total
    z = stats.norm.ppf((1 + confidence) / 2)
    denominator = 1 + z**2 / total
    centre = p + z**2 / (2 * total)
    interval = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)
    lower = (centre - interval) / denominator
    upper = (centre + interval) / denominator
    return lower, upper

# Configure
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 150

# Find most recent results
results_base = "results"
runs = sorted([d for d in os.listdir(results_base) if d.startswith("run_")])
latest_run = runs[-1] if runs else None

if not latest_run:
    print("ERROR: No results found. Run main.py first.")
    exit(1)

print(f"Using results from: {latest_run}")

# Load data
data_path = os.path.join(results_base, latest_run, "data", "analyzed_combined.csv")
df = pd.read_csv(data_path)

print(f"Loaded {len(df)} records")
print(f"Agents: {df['agent'].unique()}")
print(f"States: {df['state'].unique()}")

# Output directory for new plots
output_dir = os.path.join(results_base, latest_run, "supplementary_analysis")
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# ANALYSIS 1: Post-hoc Mann-Whitney U Tests with ALL pairwise p-values
# =============================================================================
print("\n" + "="*70)
print("ANALYSIS 1: POST-HOC MANN-WHITNEY U TESTS (PAIRWISE COMPARISONS)")
print("="*70)

agents = df['agent'].unique()
n_agents = len(agents)
n_comparisons = n_agents * (n_agents - 1) // 2
bonferroni_alpha = 0.05 / n_comparisons

print(f"\nNumber of agents: {n_agents}")
print(f"Number of pairwise comparisons: {n_comparisons}")
print(f"Bonferroni-corrected alpha: {bonferroni_alpha:.6f}")

# Get friction scores by agent
agent_friction = {agent: df[df['agent'] == agent]['friction_score'].dropna().values
                  for agent in agents}

# Print sample sizes with 95% CI
print("\nSample sizes per agent (with 95% CI for mean):")
agent_stats_with_ci = []
for agent, scores in agent_friction.items():
    ci_lower, ci_upper = compute_ci_95(scores)
    agent_stats_with_ci.append({
        'Agent': agent,
        'n': len(scores),
        'Mean': np.mean(scores),
        'CI_95_Lower': ci_lower,
        'CI_95_Upper': ci_upper,
        'Median': np.median(scores),
        'Std': np.std(scores)
    })
    print(f"  {agent}: n={len(scores)}, mean={np.mean(scores):.4f} [95% CI: {ci_lower:.4f}-{ci_upper:.4f}], median={np.median(scores):.4f}")

# Save agent stats with CI
agent_stats_df = pd.DataFrame(agent_stats_with_ci)
agent_stats_df.to_csv(os.path.join(output_dir, "agent_friction_stats_with_ci.csv"), index=False)
print(f"\nAgent statistics with CI saved to: {os.path.join(output_dir, 'agent_friction_stats_with_ci.csv')}")

# Perform all pairwise comparisons
results_table = []
print("\n" + "-"*90)
print(f"{'Comparison':<45} {'U-stat':>12} {'p-value':>12} {'Corrected':>12} {'Significant':>12}")
print("-"*90)

for agent1, agent2 in combinations(agents, 2):
    g1 = agent_friction[agent1]
    g2 = agent_friction[agent2]

    if len(g1) >= 2 and len(g2) >= 2:
        # Two-sided Mann-Whitney U test
        u_stat, p_value = stats.mannwhitneyu(g1, g2, alternative='two-sided')

        # Bonferroni-corrected p-value
        p_corrected = min(p_value * n_comparisons, 1.0)

        # Significance
        sig_raw = "*" if p_value < 0.05 else ""
        sig_corrected = "**" if p_corrected < 0.05 else ""

        # Effect size (rank-biserial correlation)
        r = 1 - (2 * u_stat) / (len(g1) * len(g2))

        results_table.append({
            'Agent 1': agent1,
            'Agent 2': agent2,
            'n1': len(g1),
            'n2': len(g2),
            'U_statistic': u_stat,
            'p_value': p_value,
            'p_corrected': p_corrected,
            'effect_size_r': r,
            'significant_raw': p_value < 0.05,
            'significant_bonferroni': p_corrected < 0.05
        })

        comparison_str = f"{agent1} vs {agent2}"
        print(f"{comparison_str:<45} {u_stat:>12.0f} {p_value:>12.6f} {p_corrected:>12.6f} {sig_raw+sig_corrected:>12}")

print("-"*90)
print("* = significant at p<0.05 (uncorrected)")
print("** = significant at p<0.05 (Bonferroni-corrected)")

# Save to CSV
results_df = pd.DataFrame(results_table)
results_df.to_csv(os.path.join(output_dir, "pairwise_mannwhitney_results.csv"), index=False)
print(f"\nResults saved to: {os.path.join(output_dir, 'pairwise_mannwhitney_results.csv')}")

# Summary of significant comparisons
print("\n--- SUMMARY: Significant Pairwise Differences ---")
sig_uncorrected = results_df[results_df['significant_raw'] == True]
sig_corrected = results_df[results_df['significant_bonferroni'] == True]

print(f"Significant (uncorrected p<0.05): {len(sig_uncorrected)} comparisons")
for _, row in sig_uncorrected.iterrows():
    print(f"  - {row['Agent 1']} vs {row['Agent 2']}: p={row['p_value']:.6f}, effect size r={row['effect_size_r']:.3f}")

print(f"\nSignificant (Bonferroni-corrected p<0.05): {len(sig_corrected)} comparisons")
for _, row in sig_corrected.iterrows():
    print(f"  - {row['Agent 1']} vs {row['Agent 2']}: p_corrected={row['p_corrected']:.6f}")

# =============================================================================
# ANALYSIS 2: REJECTION RATE ANALYSIS (RQ4)
# =============================================================================
print("\n" + "="*70)
print("ANALYSIS 2: REJECTION RATE ANALYSIS (RQ4)")
print("="*70)

# Check state distribution
print("\nPR State distribution:")
print(df['state'].value_counts())

# Define rejection: state == 'closed' typically means rejected (not merged)
# Merged PRs usually have state == 'merged' or they have merged_at timestamp
df['is_rejected'] = df['state'].apply(lambda x: 1 if str(x).lower() == 'closed' else 0)
df['is_merged_explicit'] = df['state'].apply(lambda x: 1 if 'merge' in str(x).lower() else 0)

# Calculate per-PR stats (aggregate by PR)
# Since each PR can have multiple comments, aggregate at PR level
if 'id_pr' in df.columns:
    pr_level = df.groupby('id_pr').agg({
        'friction_score': 'mean',
        'agent': 'first',
        'state': 'first',
        'is_rejected': 'first',
        'is_merged_explicit': 'first',
        'pr_type': 'first'
    }).reset_index()
else:
    pr_level = df

print(f"\nTotal unique PRs: {len(pr_level)}")

# Rejection rate by agent with 95% CI
print("\n--- Rejection Rate by Agent (with 95% CI) ---")
rejection_by_agent = pr_level.groupby('agent').agg({
    'is_rejected': ['sum', 'count', 'mean'],
    'is_merged_explicit': ['sum', 'mean']
}).round(4)
rejection_by_agent.columns = ['rejected_count', 'total_prs', 'rejection_rate', 'merged_count', 'merge_rate']

# Add 95% CI for rejection rate (Wilson score interval)
ci_lower_list = []
ci_upper_list = []
for idx, row in rejection_by_agent.iterrows():
    ci_lower, ci_upper = compute_ci_proportion(int(row['rejected_count']), int(row['total_prs']))
    ci_lower_list.append(ci_lower)
    ci_upper_list.append(ci_upper)

rejection_by_agent['CI_95_Lower'] = ci_lower_list
rejection_by_agent['CI_95_Upper'] = ci_upper_list
rejection_by_agent = rejection_by_agent.sort_values('rejection_rate', ascending=False)

print(rejection_by_agent.to_string())
print("\n   Formatted with CI:")
for idx, row in rejection_by_agent.iterrows():
    print(f"   {idx}: {row['rejection_rate']*100:.1f}% [95% CI: {row['CI_95_Lower']*100:.1f}%-{row['CI_95_Upper']*100:.1f}%] (n={int(row['total_prs'])})")

# Statistical test: Chi-square for rejection rate independence across agents
print("\n--- Chi-Square Test: Rejection Independence Across Agents ---")
contingency = pd.crosstab(pr_level['agent'], pr_level['is_rejected'])
print(f"\nContingency table:")
print(contingency)

if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    print(f"\nChi-square statistic: {chi2:.4f}")
    print(f"Degrees of freedom: {dof}")
    print(f"P-value: {p_val:.6f}")
    print(f"Significant (p<0.05): {'Yes' if p_val < 0.05 else 'No'}")

# Correlation: Friction vs Rejection
print("\n--- Correlation: Mean Friction Score vs Rejection ---")
corr, p_corr = stats.pointbiserialr(pr_level['friction_score'], pr_level['is_rejected'])
print(f"Point-biserial correlation: r = {corr:.4f}")
print(f"P-value: {p_corr:.6f}")
print(f"Interpretation: {'Higher friction associated with rejection' if corr > 0 else 'Higher friction associated with merge'}")

# Compare friction scores: rejected vs merged PRs
print("\n--- Friction Score: Rejected vs Merged PRs ---")
rejected_friction = pr_level[pr_level['is_rejected'] == 1]['friction_score'].dropna()
merged_friction = pr_level[pr_level['is_rejected'] == 0]['friction_score'].dropna()

print(f"Rejected PRs: n={len(rejected_friction)}, mean friction={rejected_friction.mean():.4f}, median={rejected_friction.median():.4f}")
print(f"Merged PRs: n={len(merged_friction)}, mean friction={merged_friction.mean():.4f}, median={merged_friction.median():.4f}")

if len(rejected_friction) >= 2 and len(merged_friction) >= 2:
    u_stat, p_val = stats.mannwhitneyu(rejected_friction, merged_friction, alternative='two-sided')
    print(f"Mann-Whitney U test: U={u_stat:.0f}, p={p_val:.6f}")
    print(f"Significant difference (p<0.05): {'Yes' if p_val < 0.05 else 'No'}")

# Save rejection analysis
rejection_by_agent.to_csv(os.path.join(output_dir, "rejection_rate_by_agent.csv"))
print(f"\nRejection analysis saved to: {os.path.join(output_dir, 'rejection_rate_by_agent.csv')}")

# Visualization 1: Rejection rate by agent (Bar plot)
plt.figure(figsize=(10, 7))
rejection_sorted = rejection_by_agent.sort_values('rejection_rate', ascending=True)
colors = plt.cm.RdYlGn(1 - rejection_sorted['rejection_rate'].values)
bars = plt.barh(rejection_sorted.index, rejection_sorted['rejection_rate'] * 100, color=colors, edgecolor='black')
plt.xlabel('Rejection Rate (%)', fontsize=12)
plt.ylabel('AI Agent', fontsize=12)
plt.title('PR Rejection Rate by AI Agent', fontsize=14, fontweight='bold')
plt.xlim(0, 100)
for bar, val in zip(bars, rejection_sorted['rejection_rate']):
    plt.text(val*100 + 1, bar.get_y() + bar.get_height()/2, f'{val*100:.1f}%', va='center')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "rejection_rate_by_agent.png"), dpi=300, bbox_inches='tight')
print(f"Saved: {os.path.join(output_dir, 'rejection_rate_by_agent.png')}")
plt.close()

# Visualization 2: Friction by outcome (Box plot)
plt.figure(figsize=(8, 7))
pr_level['outcome'] = pr_level['is_rejected'].map({0: 'Merged', 1: 'Rejected'})
sns.boxplot(x='outcome', y='friction_score', data=pr_level, hue='outcome', palette=['green', 'red'], legend=False)
plt.xlabel('PR Outcome', fontsize=12)
plt.ylabel('Mean Friction Score', fontsize=12)
plt.title('Friction Score by PR Outcome', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "friction_by_outcome.png"), dpi=300, bbox_inches='tight')
print(f"Saved: {os.path.join(output_dir, 'friction_by_outcome.png')}")
plt.close()

# =============================================================================
# ANALYSIS 3: PR TYPES DISTRIBUTION VISUALIZATION
# =============================================================================
print("\n" + "="*70)
print("ANALYSIS 3: PR TYPES DISTRIBUTION VISUALIZATION")
print("="*70)

# Filter out unknown types
df_typed = df[df['pr_type'] != 'unknown'].copy()
print(f"\nPRs with known type: {len(df_typed)}")

# PR Type distribution
type_counts = df_typed['pr_type'].value_counts()
print("\nPR Type Distribution:")
print(type_counts)

# Calculate percentages
total = type_counts.sum()
print("\nPercentages:")
for pr_type, count in type_counts.items():
    print(f"  {pr_type}: {count} ({100*count/total:.1f}%)")

# Visualization 3: Pie Chart - PR Type Distribution
plt.figure(figsize=(10, 10))
colors = plt.cm.Set3(np.linspace(0, 1, len(type_counts)))
wedges, texts, autotexts = plt.pie(type_counts, labels=type_counts.index,
                                     autopct='%1.1f%%', colors=colors,
                                     startangle=90, pctdistance=0.8)
plt.title('Distribution of PR Types', fontsize=14, fontweight='bold')
for autotext in autotexts:
    autotext.set_fontsize(10)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pr_types_pie_chart.png"), dpi=300, bbox_inches='tight')
print(f"Saved: {os.path.join(output_dir, 'pr_types_pie_chart.png')}")
plt.close()

# Visualization 4: Bar Chart - PR Type Counts
plt.figure(figsize=(10, 8))
type_counts_sorted = type_counts.sort_values(ascending=True)
bars = plt.barh(type_counts_sorted.index, type_counts_sorted.values,
                color=plt.cm.viridis(np.linspace(0.2, 0.8, len(type_counts_sorted))),
                edgecolor='black')
plt.xlabel('Number of Review Comments', fontsize=12)
plt.ylabel('PR Type', fontsize=12)
plt.title('PR Type Frequency', fontsize=14, fontweight='bold')
for bar, val in zip(bars, type_counts_sorted.values):
    plt.text(val + 50, bar.get_y() + bar.get_height()/2, f'{val:,}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pr_types_bar_chart.png"), dpi=300, bbox_inches='tight')
print(f"Saved: {os.path.join(output_dir, 'pr_types_bar_chart.png')}")
plt.close()

# Visualization 5: Stacked Bar - PR Type by Agent
plt.figure(figsize=(12, 8))
type_agent = df_typed.groupby(['pr_type', 'agent']).size().unstack(fill_value=0)
type_agent_pct = type_agent.div(type_agent.sum(axis=1), axis=0) * 100
type_agent_pct.plot(kind='barh', stacked=True, colormap='tab10', edgecolor='white', figsize=(12, 8))
plt.xlabel('Percentage (%)', fontsize=12)
plt.ylabel('PR Type', fontsize=12)
plt.title('PR Type Distribution by AI Agent', fontsize=14, fontweight='bold')
plt.legend(title='AI Agent', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pr_types_by_agent_stacked.png"), dpi=300, bbox_inches='tight')
print(f"Saved: {os.path.join(output_dir, 'pr_types_by_agent_stacked.png')}")
plt.close()

# Visualization 6: Grouped Bar - PR Types per Agent
plt.figure(figsize=(14, 8))
agent_type = df_typed.groupby(['agent', 'pr_type']).size().unstack(fill_value=0)
agent_type.plot(kind='bar', colormap='Set3', edgecolor='black', width=0.8, figsize=(14, 8))
plt.xlabel('AI Agent', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('PR Type Counts by AI Agent', fontsize=14, fontweight='bold')
plt.legend(title='PR Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pr_types_per_agent_grouped.png"), dpi=300, bbox_inches='tight')
print(f"Saved: {os.path.join(output_dir, 'pr_types_per_agent_grouped.png')}")
plt.close()

# Additional: PR Type statistics table with 95% CI
pr_type_stats = df_typed.groupby('pr_type').agg({
    'friction_score': ['mean', 'std', 'median', 'count'],
    'is_negative': 'mean'
}).round(4)
pr_type_stats.columns = ['mean_friction', 'std_friction', 'median_friction', 'comment_count', 'negative_rate']

# Add 95% CI for mean friction
ci_lower_list = []
ci_upper_list = []
for pr_type in pr_type_stats.index:
    scores = df_typed[df_typed['pr_type'] == pr_type]['friction_score'].dropna().values
    ci_lower, ci_upper = compute_ci_95(scores)
    ci_lower_list.append(ci_lower)
    ci_upper_list.append(ci_upper)

pr_type_stats['CI_95_Lower'] = ci_lower_list
pr_type_stats['CI_95_Upper'] = ci_upper_list
pr_type_stats = pr_type_stats.sort_values('mean_friction', ascending=False)

print("\n--- Friction Statistics by PR Type (with 95% CI) ---")
print(pr_type_stats.to_string())

print("\n   Formatted with CI:")
for pr_type, row in pr_type_stats.iterrows():
    print(f"   {pr_type}: mean={row['mean_friction']:.4f} [95% CI: {row['CI_95_Lower']:.4f}-{row['CI_95_Upper']:.4f}] (n={int(row['comment_count'])})")

pr_type_stats.to_csv(os.path.join(output_dir, "pr_type_statistics_with_ci.csv"))
print(f"\nPR Type statistics saved to: {os.path.join(output_dir, 'pr_type_statistics_with_ci.csv')}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*70)
print("ANALYSIS COMPLETE - SUMMARY")
print("="*70)

print(f"""
OUTPUT FILES CREATED:
CSV Files (with 95% Confidence Intervals):
1. {os.path.join(output_dir, 'pairwise_mannwhitney_results.csv')}
   - All pairwise Mann-Whitney U test results with p-values and Bonferroni correction

2. {os.path.join(output_dir, 'agent_friction_stats_with_ci.csv')}
   - Friction statistics per agent WITH 95% CI

3. {os.path.join(output_dir, 'rejection_rate_by_agent.csv')}
   - Rejection rate analysis by AI agent WITH 95% CI (Wilson score interval)

4. {os.path.join(output_dir, 'pr_type_statistics_with_ci.csv')}
   - Friction statistics by PR type WITH 95% CI

PNG Images:
5. {os.path.join(output_dir, 'rejection_rate_by_agent.png')}
   - Bar chart of rejection rates by AI agent

6. {os.path.join(output_dir, 'friction_by_outcome.png')}
   - Box plot of friction scores by PR outcome (Merged vs Rejected)

7. {os.path.join(output_dir, 'pr_types_pie_chart.png')}
   - Pie chart of PR types distribution

8. {os.path.join(output_dir, 'pr_types_bar_chart.png')}
   - Horizontal bar chart of PR type frequencies

9. {os.path.join(output_dir, 'pr_types_by_agent_stacked.png')}
   - Stacked bar chart showing PR type distribution by AI agent

10. {os.path.join(output_dir, 'pr_types_per_agent_grouped.png')}
   - Grouped bar chart showing PR type counts per AI agent

KEY FINDINGS:
""")

# Mann-Whitney findings
print("1. MANN-WHITNEY U PAIRWISE COMPARISONS:")
print(f"   - Total comparisons: {n_comparisons}")
print(f"   - Significant (raw p<0.05): {len(sig_uncorrected)}")
print(f"   - Significant (Bonferroni-corrected): {len(sig_corrected)}")

# Rejection rate findings
print("\n2. REJECTION RATE ANALYSIS:")
overall_rejection = pr_level['is_rejected'].mean()
print(f"   - Overall rejection rate: {overall_rejection*100:.1f}%")
print(f"   - Highest rejection: {rejection_by_agent['rejection_rate'].idxmax()} ({rejection_by_agent['rejection_rate'].max()*100:.1f}%)")
print(f"   - Lowest rejection: {rejection_by_agent['rejection_rate'].idxmin()} ({rejection_by_agent['rejection_rate'].min()*100:.1f}%)")

# PR Type findings
print("\n3. PR TYPES DISTRIBUTION:")
print(f"   - Most common type: {type_counts.idxmax()} ({type_counts.max()} comments, {100*type_counts.max()/total:.1f}%)")
print(f"   - Least common type: {type_counts.idxmin()} ({type_counts.min()} comments, {100*type_counts.min()/total:.1f}%)")
print(f"   - Highest friction type: {pr_type_stats['mean_friction'].idxmax()} (mean={pr_type_stats['mean_friction'].max():.4f})")
