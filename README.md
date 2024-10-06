# II-COS
This repository contains the codes to reproduce the experiments results in *Real-Time Selection Under General Constraints via Predictive Inference (NeurIPS 2024)* by Yuyang Huo, Lin Lu, Haojie Ren and Changliang Zou.

## Introduction of II-COS
 We consider the problem of sample selection in the online setting, where one encounters a possibly infinite sequence of individuals collected over time with covariate information available. The goal is to select samples of interest that are characterized by their unobserved responses until the user-specified stopping time. We derive a new decision rule that enables us to find more preferable samples that meet practical requirements by simultaneously controlling two types of general constraints: individual and interactive constraints.
## Folder contents

- **synthetic-data**: The codes to reproduce the synthetic experiments results in the paper.

- **real-data**: Two real data sets in application.


## Folders
- `synthetic-data/`: contains the codes for synthetic data simulation experiments in the paper.
- `real-data/`: contains the codes for real data experiments in the paper.


## Guide for the codes in `synthetic-data/` folder

### Helper functions for all the experiments.
- `functions_OnSel.R/`
- `algoclass_OnSel.R/`

### Codes for reproducing results of the synthetic data.
- Classification example `(Figure 1, Figure 2 in Section 4.1)`: `simulation_cla_final.R/`
- Regression example `(Figure 7, Figure 8 in Appendix E.4)`: `simulation_reg_final.R/`


## Guide for the codes in `real-data/` folder
### Codes for reproducing results in Section 4.2
- Candidate data `(Table 1 (a), Figure 3 left in Section 4.2)`: `Candidate-Main results and the table.R/`, `Candidate-plot.R/`
- Income data `(Table 1 (b), Figure 3 right in Section 4.2)`: `Census-Main results and the table.R/`, `Census-plot.R/`

