RUN:

0. Go to my Google Colab to find out how to run AlphaFold, RFDiffusion and protein MPNN 
https://github.com/olesok87/Miniproject2_CDR_hallucination_Google_Colab.git

1. In the Script folder you will find script scoring data from Rosetta and PISA for protein-protein docking models. The scoring scheme combines metrics from both tools to evaluate the quality of the docking interfaces.

2. Place score.sc (RosettaInterfaceAnalyzer) in Results/RosettaAnalyzer (or change the path in Scripts/Subscripts/Rosetta_interface_Analysis.py). The scripts exports data from .sc format to csv which is necessary for the composite script (next)

3. Place pisa_interfaces_summary.csv (PISA output) in Results/PISA (or change the path in Scripts/Subscripts/Normalise and Score Rosetta and PISA.py). This script will score/weight variables from Rosetta and PISA to produce normalised score.


WEIGHTS:
1. Rosetta Score Metrics and Normalization:
Metric	Direction	Normalization	Weight within Rosetta_score
Interface ΔG (dG_separated)	more negative = better	inverted min–max	0.5
Interface area (int_area)	higher = better	min–max	0.4
Number of H-bonds (n_hbonds)	higher = better	min–max	0.2
 Rosetta_score = 0.5 × ΔG_norm + 0.3 × Area_norm + 0.2 × H-bonds_norm

2. PISA Score Metrics and Normalization:
Metric	Direction	Normalization	Weight within PISA_score
Interface area (int_area)	higher = better	min–max	0.4
Interface stability energy (stab_en)	more negative = better	inverted min–max	0.3
Number of H-bonds (n_hbonds)	higher = better	min–max	0.2
p-value (pvalue)	lower = better	inverted min–max	0.1

PISA_score = 0.4 × Area_norm + 0.3 × Stability_norm + 0.2 × H-bonds_norm + 0.1 × p-value_norm


WORK IN PROGRESS (This will be important for large datasets in the future):
- Final Composite Score:
After normalizing Rosetta_score and PISA_score to [0,1] across all models:

composite_score = 0.7 x Rosetta Score + 0.3 x PISA_score /1


- Run the best 2 candidates through :
--Analyze biophysical propensities: Developability Analyzer (https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabpred)
--Solubility Analyzer (https://www-cohsoftware.ch.cam.ac.uk/index.php/camsolstrucorr)
--Aggregation Propensity (https://biocomp.chem.uw.edu.pl/A3D2)

Run my Simple Helicase Mutant Selector on the identified weak sequences (eg, loops) and score mutants by Rosetta_ddG_monomer
