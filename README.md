# Pb(II) MOF: Multi-objective Assessment and Feature-guided Design

Datasets, trained models, and analysis scripts for multi-objective assessment and feature-guided design of MOF-based Pb(II) adsorbents.

## Layout

```text
PbII-MOF-ML-CKR/
  data/                   Literature datasets, training matrices, and screening records
  model/C/                Six reported capacity regressors and their validation records
  model/K/                Six reported uptake-at-time regressors and validation records
  model/R/                Six reported retention regressors and validation records
  model/best/             Three selected trained models
  scripts/                Portable training and interpretation entry points
  scripts/historical/     Earlier training workflows
  outputs/                Analysis tables and figures
  README.md
  requirements.txt
  manifest.json           SHA-256 inventory
```

## Tasks and Selected Models

| Task | Response | Selected algorithm | Archived training rows | OOF R2 |
| --- | --- | --- | ---: | ---: |
| C | Adsorption capacity, csv_qe (mg/g) | CatBoost | 3060 | 0.894178 |
| K | Uptake at contact time, csv_qt (mg/g) | XGBoost | 2579 | 0.888994 |
| R | Retention, csv_retention_percent (%) | CatBoost | 2013 | 0.956889 |

C, K, and R represent adsorption capacity, uptake at a specified contact time, and adsorption-capacity retention, respectively. An evidence-based water-stability score (W) complements these measures in the performance assessment index (PAI).

## Installation and Commands

Fitted models are stored with Git LFS. Install Git LFS before cloning, or run `git lfs install` and `git lfs pull` after cloning to download the model binaries.

```bash
git lfs install
git clone https://github.com/MMD-XTT/PbII-MOF-ML-CKR.git
cd PbII-MOF-ML-CKR
```

Install the dependencies listed in `requirements.txt` and run the following commands from the repository root.

```bash
pip install -r requirements.txt
python scripts/verify_package.py --all-models
python scripts/train_models.py --task C --algorithm best --cv
python scripts/train_models.py --task K --algorithm best --cv
python scripts/train_models.py --task R --algorithm all --cv
python scripts/train_models.py --task K --mode bayesian --trials 30 --cv
python scripts/shap_analysis.py --task all
python scripts/create_ckr_pdp.py
python scripts/create_c_2d_pdp.py
python scripts/create_c_categorical_shap_bars.py --group all --chart all
python scripts/create_kr_categorical_shap.py --chart all
```

`train_models.py` trains models on the current workbooks using the saved model configurations. The `--cv` option enables five-fold cross-validation, and `--mode bayesian` enables Optuna TPE hyperparameter optimization. Use `--input-mode archived` to select the model-specific training matrices. Training outputs are saved to `outputs/training/`.

`shap_analysis.py` calculates SHAP contributions for the selected models. The categorical SHAP and PDP scripts generate feature-importance, response-distribution, and partial-dependence plots.

## Data and Provenance

- data/master.xlsx: complete literature database.
- data/C_model.xlsx: C dataset.
- data/K模型_qt统一.xlsx: the sole K source workbook for new training and interpretation.
- data/R_model.xlsx: R dataset, including direct cycle records and auxiliary retention labels.
- data/training/: model-specific feature matrices, identifiers, targets, and original row indices.
- data/screening/: component scores, candidate rankings, enrichment tables and water-stability registers.
- model/*/*/: trained models, hyperparameters, performance metrics, fold results, out-of-fold predictions, and optimization records.
- model_registry.csv: all 18 archived runs and their training-source populations.

The selected K model uses a 2579-row training matrix. The unified K workbook contains 2645 labeled rows and is the default input for new training and interpretation. Model-specific training matrices are available in `data/training/`.

## Screening

PAI screening uses reference conditions of pH 5, 298 K, adsorbent dose 0.2 g/L, equilibrium concentration 300 mg/L, and the third reuse cycle. Component scores, candidate rankings, feature proportions, enrichment results, and water-stability records are provided in `data/screening/`.

## Additional Scripts

- train_six_regressors.py: six-algorithm training workflow with configurable input, output, target, and condition variables.
- historical/: earlier workflows, including alternative descriptor and target configurations.
- build_figure5_pai_final_reference_panels.py, build_figure5_formal_screening_v2.py: component score calculation and screening plots.
- build_figure6_design_guidance.py, build_figure6e_zn_f1_m1_pai.py: enrichment/design guidance and parent-framework comparison.

Only load serialized model files from trusted sources.
