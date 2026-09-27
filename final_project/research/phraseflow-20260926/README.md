# Sonja PhraseFlow improvement study

Author: Sonja Sahebzad | Sonja Projects | 26 September 2026

[Read the published study](https://sonja242.github.io/data-science-capstone-swiftkey/phraseflow-controlled-study.html). Open `PhraseFlow_Study.html` for results and `PhraseFlow_Study.Rmd` for its source. Open the existing `final_project/Sonja Next Word.Rproj` in RStudio. The local product preview has its own project in `final_project/phraseflow_preview`.

## Decision and scope

Keep the existing production model. The target-only head averages 162/600 top-three matches and conditional-teacher training 161.3/600, versus the CPU baseline's 164/600. All six seeds/runs are retained. These are exploratory comparisons on reused development cases. The reserved 900 cases were not opened. Quiz demonstrations were performed only after freezing the decision and do not support model selection.

`protocol.json` was saved before training. `training_manifest.json`, `teacher_training_fingerprint.json` and `code_fingerprints.json` identify inputs and code. `final_audit.json` reconciles saved case-level results, confirms unchanged model/protocol checksums and counts five pitch slides.

## Reproduction

Use a fresh copy of the full corpus project, preserving the existing private input datasets, CPU model, original neural student and frozen teacher. Scripts use the corpus project root as their working directory. They refuse to overwrite completed study evidence; do not delete the original results. In a separate fresh copy, use a new output location consistently or omit only the generated study outputs from that copy before execution.

Run with R 4.6.1 and the project's `.venv-neural/Scripts/python.exe` environment:

1. `Rscript --vanilla final_project/research/phraseflow-20260926/prepare_training.R`
2. `.venv-neural/Scripts/python.exe final_project/research/phraseflow-20260926/score_training_teacher.py`
3. `.venv-neural/Scripts/python.exe final_project/research/phraseflow-20260926/train_reranker.py`
4. `Rscript --vanilla final_project/research/phraseflow-20260926/evaluate_r.R`
5. Build the separate historical preview using `build_preview.py`. The optional quiz-verification script remains private because it contains an assessment answer; only its aggregate outputs are published. Those quiz checks are not needed to reproduce the model comparison.
6. Run `finalize_evidence.py` to reconcile the recorded case results and generate source/context summaries.
7. Knit `PhraseFlow_Study.Rmd` using `rmarkdown::render()`.

Teacher scoring and training require the existing CUDA environment. Exported prediction and verification run in native R on CPU. Exact R packages are listed in `R-session-info.txt`; device, costs and settings are in the JSON records. No command needs the reserved test examples.

## Evidence files

- `selected_comparison.csv`: six run summaries and paired bootstrap intervals.
- `development_cases.csv`: 3,600 case/run rows, including baseline and candidate outcomes, without raw text.
- `subgroup_results.csv`: exploratory counts by source and context length for every seed.
- `learning_curves.csv`: all epochs and objectives, including losses and teacher divergence.
- `cpu_checks.json`, `cpu_timing.csv`: complete inference timing and R/Python parity.
- `quiz_demonstration_summary.csv`: aggregate comparisons with saved reviewed/submitted choices; archived neural rankings are not new teacher runs.
- `preview_checks.json`: Shiny server checks and unchanged first three outputs on 600 cases.

Private texts, candidate word details, checkpoints and exports remain under `data/phraseflow_20260926`. Do not publish that folder or quiz answers. Keep documentation, app behavior and model experiments distinct: a longer visible list does not establish higher top-three accuracy.

## Documentation and release

The local preview includes `documentation/PhraseFlow_User_Guide.Rmd`, knitted HTML and a four-page PDF. `PhraseFlow_Pitch.Rpres` is the five-slide native RStudio Presenter source. Product name, styling and documentation are for review; existing Shiny, RPubs and GitHub pages have not been replaced by this study. No final course submission was made.

## Publication on 27 September 2026

The study is now published separately from the later multilingual release. The experimental counts, protocol and original decision are unchanged. Publication changes add current product links and clarify that the course requires only the app and five-slide deck URLs; the portfolio is not submitted for peer grading. The historical release records and code fingerprints describe the original experiment. Current publication hashes are recorded separately.

The tables and figures can be knitted from the committed CSV/JSON measurements without private data or model training. Full retraining requires the previously saved input checkpoints and private corpus splits documented above. Raw text, held-out answers, checkpoints and the private quiz-verification script are excluded from GitHub.

The same report is available on [RPubs](https://rpubs.com/Sonja_Janssen/phraseflow-controlled-study).
