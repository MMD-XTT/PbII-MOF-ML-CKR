# Pb(II) MOF: Multi-objective Assessment and Feature-guided Design

Literature-derived data, fitted regression models, training recipes, SHAP/PDP interpretation, and material screening records for Pb(II) adsorption.

## Layout

```text
PbII-MOF-ML-CKR/
  data/                   Original workbooks, frozen training matrices, screening records
  model/C/                Six reported capacity regressors and their validation records
  model/K/                Six reported uptake-at-time regressors and validation records
  model/R/                Six reported retention regressors and validation records
  model/best/             Exact copies of the three selected fitted models
  scripts/                Portable training and interpretation entry points
  scripts/historical/     Earlier training scripts, retained for provenance
  outputs/                Generated analysis results; archived model files stay unchanged
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

K denotes the prediction task, and XGBoost denotes its selected algorithm. The screening K score is calculated separately from measured kinetic curves. W is an evidence-based stability score; it is not a fourth trained regression model.

## Installation and Commands

Fitted models are stored with Git LFS. Install Git LFS before cloning, or run `git lfs install` and `git lfs pull` after cloning to download the model binaries.

```bash
git lfs install
git clone https://github.com/MMD-XTT/PbII-MOF-ML-CKR.git
cd PbII-MOF-ML-CKR
```

The package is checked with Python 3.12.14 and the versions in requirements.txt. Run commands from this folder; file locations are resolved from script locations, not from a personal drive letter.

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

`train_models.py` refits clones of the saved estimator recipes on the current workbooks by default. K reads only data/K模型_qt统一.xlsx. `--input-mode archived` uses the frozen model-specific matrices for historical reproduction. `--mode bayesian` performs a NEW Optuna TPE search with five-fold CV; its results can differ from archived fits. All new training files go into outputs/training; archived models are never overwritten. Full optimization is computationally expensive and is not automatically run during packaging.

`shap_analysis.py` uses native CatBoost/XGBoost tree contributions on the current workbooks by default; `--input-mode archived` selects frozen training matrices. Existing publication-format SHAP and categorical/PDP plotting scripts are included separately. Their tables and original analysis files are retained.

## Data and Provenance

- data/master.xlsx: complete literature database.
- data/C_model.xlsx: C dataset.
- data/K模型_qt统一.xlsx: the sole K source workbook for new training and interpretation.
- data/R_model.xlsx: R dataset, including direct cycle records and auxiliary retention labels.
- data/training/: estimator-specific feature matrices with identifiers, target, and original row index. Target sequences and material identifiers are checked against each archived OOF file.
- data/screening/: component scores, candidate rankings, enrichment tables and water-stability registers.
- model/*/*/: exact fitted binaries, metrics, parameters, fold metrics, OOF predictions and search histories where available.
- model_registry.csv: all 18 archived runs and their training-source populations.

The existing selected K XGBoost fit records 2579 training rows; the current unified workbook has 2645 labeled rows. Its fitted binary, OOF records and frozen input matrix retain the original training history. The older K workbook is not included. Rerunning training on the unified workbook produces a new fit and does not change the archived model or its reported metrics.

The C archived no-T fits and the remaining five K fits document random-search recipes, while selected K XGBoost and the six R no-T fits document Optuna TPE optimization. A history filename containing "bayesopt" alone does not establish the search method. Stored parameter/metric records are retained as the authoritative historical evidence.

## Packaging Checks

All 18 fitted models load and return finite predictions. Each frozen target sequence and material-row order agrees with its archived OOF file. Training is smoke-tested by refitting C/Ridge. SHAP is smoke-tested on 64 rows per selected model, and 1D/2D PDP on 96-row samples with reduced grids. These generated smoke-test outputs are not manuscript results; the original full analysis tables and images remain separately archived. Full hyperparameter optimization is not rerun during collection.

PAI retains the reference settings of pH 5, 298 K, adsorbent dose 0.2 g/L, Ce 300 mg/L and the third reuse cycle. Experimental validation is planned for five cycles. Archived ranks are unchanged. Experimental measurements not yet completed are not created or added to this repository.

## Additional Scripts

- train_six_regressors.py: general-purpose six-algorithm no-T Bayesian trainer for a NEW input dataset, accepting explicit input/output/target/condition arguments. It creates new models; it does not reproduce historical C preprocessing automatically.
- historical/: earlier training scripts with repository-relative paths. Some describe prior feature/target variants (including T or qr); they are provenance records, not the recommended manuscript reproduction entry points.
- build_figure5_pai_final_reference_panels.py, build_figure5_formal_screening_v2.py: component score calculation and screening plots.
- build_figure6_design_guidance.py, build_figure6e_zn_f1_m1_pai.py: enrichment/design guidance and parent-framework comparison.

Only load joblib files from trusted sources. Pickled scikit-learn models can require their original library versions. Dataset copies and fitted binaries are preserved byte-for-byte; changes are confined to copied code paths and new packaging helpers.

Several historical C/K scikit-learn binaries record version 1.9.0. The smoke tests here use 1.9.1, which emits compatibility warnings; all 18 models load and return finite test predictions. For exact historical numerical reproduction, retain the original training environment as well as these archived binaries. Dependency versions describe the verified packaging environment, not a claim that every historical model was fitted in that environment.
