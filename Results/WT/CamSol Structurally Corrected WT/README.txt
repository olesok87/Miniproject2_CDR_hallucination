This folder contains the output of the CamSol structurally corrected solubility calculation
Which consists in the followin files:
- a .pdb file with the intrinsic CamSol score in place of the occupancy column and the structurally corrected CamSol score in place of the bfactor column.
   This pdb can be used to color code either score on the surface of the molecule, by coloring the residues according to either the occupancy or the bfactor column
  (check the documentation of your molecule structure visualization software to learn how to do this)
- a .txt file (tab separated, can be opened with MS excel or any spreadsheet or text editors). With the actual numerical output and additional informations (such as residue solvent exposure).
- a _chimera.py file that is the figure of the input protein color coded with the structurally corrected CamSol score.
   You should open this directly with UCSF chimera (https://www.cgl.ucsf.edu/chimera/).
   Blue means highly soluble, red poorly soluble. You should also find an image named Solubility_profile_color_code.png
   with the conversion of the color-code to the stucturally corrected solublity profile.
- one .png figure per chain of the input structure. This figure contains both the intrinsic and the structurally corrected CamSol score profiles.
  If the input pdb file contains more than one model error bars on the structurally corrected profile are standard deviations among the different model, the plotted value is the average.
