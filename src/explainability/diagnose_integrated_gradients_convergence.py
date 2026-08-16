from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INPUT_FILE = Path('results/resnet50v2_finetuned/integrated_gradients_convergence/integrated_gradients_convergence_per_case.csv')
OUTPUT_DIR = Path('results/resnet50v2_finetuned/integrated_gradients_convergence/diagnostics')
PER_CASE_OUTPUT = OUTPUT_DIR / 'ig_convergence_diagnostic_200_vs_400.csv'
SUMMARY_OUTPUT = OUTPUT_DIR / 'ig_convergence_diagnostic_summary.csv'
JSON_OUTPUT = OUTPUT_DIR / 'ig_convergence_diagnostic_summary.json'
FIGURE_DENOMINATOR = OUTPUT_DIR / 'relative_error_vs_prediction_difference.png'
FIGURE_ABSOLUTE = OUTPUT_DIR / 'absolute_completeness_error_200_vs_400.png'
FIGURE_RELATIVE = OUTPUT_DIR / 'relative_completeness_error_200_vs_400.png'
FIGURE_MAP = OUTPUT_DIR / 'map_similarity_200_vs_400.png'

STEP_COUNTS_TO_COMPARE = [200, 400]
SMALL_DENOMINATOR_THRESHOLD = 0.10
VERY_SMALL_DENOMINATOR_THRESHOLD = 0.05

if not INPUT_FILE.exists():
    raise FileNotFoundError(f'Convergence results not found:\n{INPUT_FILE.resolve()}')

df = pd.read_csv(INPUT_FILE)
required = {
    'full_image_id','patient_id','outcome','saved_probability','generated_probability',
    'baseline_probability','prediction_minus_baseline','integration_steps',
    'attribution_sum_signed','completeness_delta_signed',
    'completeness_absolute_error','completeness_relative_error',
    'map_correlation_vs_400','map_mae_vs_400'
}
missing = required - set(df.columns)
if missing:
    raise ValueError('Missing expected columns: ' + ', '.join(sorted(missing)))

diagnostic_df = df[df['integration_steps'].isin(STEP_COUNTS_TO_COMPARE)].copy()
if diagnostic_df.empty:
    raise ValueError('No 200-step or 400-step rows found.')

diagnostic_df['abs_prediction_minus_baseline'] = np.abs(diagnostic_df['prediction_minus_baseline'])
diagnostic_df['small_denominator'] = diagnostic_df['abs_prediction_minus_baseline'] < SMALL_DENOMINATOR_THRESHOLD
diagnostic_df['very_small_denominator'] = diagnostic_df['abs_prediction_minus_baseline'] < VERY_SMALL_DENOMINATOR_THRESHOLD
diagnostic_df['completeness_ratio_signed'] = (
    diagnostic_df['attribution_sum_signed'] /
    diagnostic_df['prediction_minus_baseline'].replace(0, np.nan)
)

base_columns = [
    'full_image_id','patient_id','outcome','saved_probability','baseline_probability',
    'prediction_minus_baseline','abs_prediction_minus_baseline',
    'small_denominator','very_small_denominator'
]
metadata = diagnostic_df[base_columns].drop_duplicates('full_image_id').set_index('full_image_id')
metrics = [
    'attribution_sum_signed','completeness_delta_signed','completeness_absolute_error',
    'completeness_relative_error','completeness_ratio_signed',
    'map_correlation_vs_400','map_mae_vs_400'
]
pivot = diagnostic_df.pivot(index='full_image_id', columns='integration_steps', values=metrics)
pivot.columns = [f'{metric}_{int(step)}' for metric, step in pivot.columns]
case_df = metadata.join(pivot, how='inner').reset_index()

case_df['absolute_error_improvement_200_to_400'] = (
    case_df['completeness_absolute_error_200'] - case_df['completeness_absolute_error_400']
)
case_df['relative_error_improvement_200_to_400'] = (
    case_df['completeness_relative_error_200'] - case_df['completeness_relative_error_400']
)
case_df = case_df.sort_values('completeness_relative_error_400', ascending=False).reset_index(drop=True)

summary_rows = []
for steps in STEP_COUNTS_TO_COMPARE:
    subset = diagnostic_df[diagnostic_df['integration_steps'] == steps]
    summary_rows.append({
        'integration_steps': steps,
        'n': len(subset),
        'mean_abs_prediction_minus_baseline': subset['abs_prediction_minus_baseline'].mean(),
        'median_abs_prediction_minus_baseline': subset['abs_prediction_minus_baseline'].median(),
        'n_small_denominator_lt_0_10': int(subset['small_denominator'].sum()),
        'n_very_small_denominator_lt_0_05': int(subset['very_small_denominator'].sum()),
        'mean_absolute_completeness_error': subset['completeness_absolute_error'].mean(),
        'median_absolute_completeness_error': subset['completeness_absolute_error'].median(),
        'mean_relative_completeness_error': subset['completeness_relative_error'].mean(),
        'median_relative_completeness_error': subset['completeness_relative_error'].median(),
        'mean_map_correlation_vs_400': subset['map_correlation_vs_400'].mean(),
        'median_map_correlation_vs_400': subset['map_correlation_vs_400'].median(),
        'mean_map_mae_vs_400': subset['map_mae_vs_400'].mean(),
        'median_map_mae_vs_400': subset['map_mae_vs_400'].median(),
    })
summary_df = pd.DataFrame(summary_rows)

corr_200 = float(np.corrcoef(case_df['abs_prediction_minus_baseline'], case_df['completeness_relative_error_200'])[0,1])
corr_400 = float(np.corrcoef(case_df['abs_prediction_minus_baseline'], case_df['completeness_relative_error_400'])[0,1])
median_abs_reduction = float(case_df['absolute_error_improvement_200_to_400'].median())
median_rel_reduction = float(case_df['relative_error_improvement_200_to_400'].median())
median_map_corr_200 = float(case_df['map_correlation_vs_400_200'].median())
median_map_mae_200 = float(case_df['map_mae_vs_400_200'].median())

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
case_df.to_csv(PER_CASE_OUTPUT, index=False)
summary_df.to_csv(SUMMARY_OUTPUT, index=False)

fig = plt.figure(figsize=(8,6))
for steps in STEP_COUNTS_TO_COMPARE:
    subset = diagnostic_df[diagnostic_df['integration_steps'] == steps]
    plt.scatter(subset['abs_prediction_minus_baseline'], subset['completeness_relative_error'], label=f'{steps} steps', alpha=0.8)
plt.axvline(SMALL_DENOMINATOR_THRESHOLD, linestyle='--', label=f"|F(x)-F(x')| = {SMALL_DENOMINATOR_THRESHOLD:.2f}")
plt.xlabel(r"$|F(x)-F(x')|$")
plt.ylabel('Relative completeness error')
plt.title('Integrated Gradients Relative Error vs Prediction Difference')
plt.legend(); plt.grid(alpha=0.25); plt.tight_layout()
fig.savefig(FIGURE_DENOMINATOR, dpi=300, bbox_inches='tight'); plt.close(fig)

fig = plt.figure(figsize=(7,6))
plt.scatter(case_df['completeness_absolute_error_200'], case_df['completeness_absolute_error_400'])
mx = float(max(case_df['completeness_absolute_error_200'].max(), case_df['completeness_absolute_error_400'].max()))
plt.plot([0,mx],[0,mx], linestyle='--')
plt.xlabel('Absolute completeness error at 200 steps'); plt.ylabel('Absolute completeness error at 400 steps')
plt.title('Integrated Gradients Completeness: 200 vs 400 Steps'); plt.grid(alpha=0.25); plt.tight_layout()
fig.savefig(FIGURE_ABSOLUTE, dpi=300, bbox_inches='tight'); plt.close(fig)

fig = plt.figure(figsize=(7,6))
plt.scatter(case_df['completeness_relative_error_200'], case_df['completeness_relative_error_400'])
mx = float(max(case_df['completeness_relative_error_200'].max(), case_df['completeness_relative_error_400'].max()))
plt.plot([0,mx],[0,mx], linestyle='--')
plt.xlabel('Relative completeness error at 200 steps'); plt.ylabel('Relative completeness error at 400 steps')
plt.title('Integrated Gradients Relative Completeness: 200 vs 400 Steps'); plt.grid(alpha=0.25); plt.tight_layout()
fig.savefig(FIGURE_RELATIVE, dpi=300, bbox_inches='tight'); plt.close(fig)

ordered = case_df.sort_values('map_correlation_vs_400_200')
fig = plt.figure(figsize=(8,6))
plt.plot(np.arange(len(ordered)), ordered['map_correlation_vs_400_200'], marker='o')
plt.xlabel('Convergence test case'); plt.ylabel('Pearson correlation'); plt.ylim(0.995, 1.0005)
plt.title('200-Step Integrated Gradients Spatial Stability'); plt.grid(alpha=0.25); plt.tight_layout()
fig.savefig(FIGURE_MAP, dpi=300, bbox_inches='tight'); plt.close(fig)

n_small = int(case_df['small_denominator'].sum())
n_very_small = int(case_df['very_small_denominator'].sum())
cases_improved_abs = int((case_df['absolute_error_improvement_200_to_400'] > 0).sum())
cases_improved_rel = int((case_df['relative_error_improvement_200_to_400'] > 0).sum())

summary_json = {
    'sample_cases': int(len(case_df)),
    'small_denominator_threshold': SMALL_DENOMINATOR_THRESHOLD,
    'very_small_denominator_threshold': VERY_SMALL_DENOMINATOR_THRESHOLD,
    'cases_with_abs_prediction_minus_baseline_lt_0_10': n_small,
    'cases_with_abs_prediction_minus_baseline_lt_0_05': n_very_small,
    'correlation_abs_prediction_difference_vs_relative_error_200': corr_200,
    'correlation_abs_prediction_difference_vs_relative_error_400': corr_400,
    'cases_with_lower_absolute_error_at_400': cases_improved_abs,
    'cases_with_lower_relative_error_at_400': cases_improved_rel,
    'median_absolute_error_reduction_200_to_400': median_abs_reduction,
    'median_relative_error_reduction_200_to_400': median_rel_reduction,
    'median_map_correlation_200_vs_400': median_map_corr_200,
    'median_map_mae_200_vs_400': median_map_mae_200,
}
with JSON_OUTPUT.open('w', encoding='utf-8') as handle:
    json.dump(summary_json, handle, indent=2)

print('='*72)
print('INTEGRATED GRADIENTS CONVERGENCE DIAGNOSTIC')
print('='*72)
print(f'\nCases analysed: {len(case_df)}')
print('\nPrediction-minus-baseline magnitude:')
print(case_df['abs_prediction_minus_baseline'].describe())
print(f'\nCases with |F(x)-F(x\')| < {SMALL_DENOMINATOR_THRESHOLD:.2f}: {n_small}')
print(f'Cases with |F(x)-F(x\')| < {VERY_SMALL_DENOMINATOR_THRESHOLD:.2f}: {n_very_small}')
print('\nCorrelation between denominator magnitude and relative completeness error:')
print(f'200 steps: {corr_200:.4f}')
print(f'400 steps: {corr_400:.4f}')
print('\n200 -> 400 improvement:')
print(f'Cases with lower absolute error at 400: {cases_improved_abs}/{len(case_df)}')
print(f'Cases with lower relative error at 400: {cases_improved_rel}/{len(case_df)}')
print(f'Median absolute error reduction: {median_abs_reduction:.6f}')
print(f'Median relative error reduction: {median_rel_reduction:.6f}')
print('\nSpatial stability of 200-step maps:')
print(f'Median Pearson correlation vs 400: {median_map_corr_200:.6f}')
print(f'Median map MAE vs 400: {median_map_mae_200:.6f}')
print('\nWorst 400-step relative completeness cases:')
cols = ['patient_id','outcome','saved_probability','baseline_probability','prediction_minus_baseline','completeness_absolute_error_400','completeness_relative_error_400','map_correlation_vs_400_200']
print(case_df[cols].head(10).to_string(index=False))
print('\nSummary by step:')
print(summary_df.to_string(index=False))
print('\nSaved diagnostics:')
print(OUTPUT_DIR.resolve())