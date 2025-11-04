
1. Rosetta Score Metrics and Normalization:
Metric	Direction	Normalization	Weight within Rosetta_score
Interface ΔG (dG_separated)	more negative = better	inverted min–max	0.5
Interface area (int_area)	higher = better	min–max	0.4
Number of H-bonds (n_hbonds)	higher = better	min–max	0.2

2. Rosetta_score = 0.5 × ΔG_norm + 0.3 × Area_norm + 0.2 × H-bonds_norm

PISA Score Metrics and Normalization:
Metric	Direction	Normalization	Weight within PISA_score
Interface area (int_area)	higher = better	min–max	0.4
Interface stability energy (stab_en)	more negative = better	inverted min–max	0.3
Number of H-bonds (n_hbonds)	higher = better	min–max	0.2
p-value (pvalue)	lower = better	inverted min–max	0.1

PISA_score = 0.4 × Area_norm + 0.3 × Stability_norm + 0.2 × H-bonds_norm + 0.1 × p-value_norm

3. Final Composite Score

After normalizing Rosetta_score and PISA_score to [0,1] across all models:

\text{composite\_score} = 0.7 \times \text{Rosetta_score} + 0.3 \times \text{PISA_score}

Higher composite_score → stronger predicted binding and better interface properties.