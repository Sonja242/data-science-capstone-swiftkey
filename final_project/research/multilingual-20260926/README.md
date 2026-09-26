# Multilingual extension study

Author: Sonja Sahebzad. Sonja Projects. 26 September 2026.

Run from the corpus project root. Preserve the completed run; preparation and training refuse overwrites. For a new experiment use a separate copy and a new output location.

Order: init_multilingual.py, prepare_multilingual.py, build_multilingual.R, install_multilingual_ui.py, portable_r_sources.py, test_multilingual.R, finish_multilingual.py. Use the project Python environment for Python scripts and R 4.6.1 for R scripts. App documents are rendered with rmarkdown and the five-slide pitch with native RStudio Presenter.

`protocol.json` fixes settings before held-out evaluation. `source_manifest.json` records the local official archive checksum and members. `metrics.json` contains separate language metrics, Wilson intervals, unigram references, CPU timings and size. `*_case_metrics.csv` preserves per-case outcomes without phrase or answer text. `checks.json` records unchanged English predictions and Unicode/server checks.

The three new fixed language models use 75,000 training lines each and 600 held-out examples each. English remains unchanged. The reserved English 900 cases are never read. These are small experimental language models; no all-language or calibrated-confidence claim is supported. The three language-specific hold-outs must not be reused as independent final tests after tuning based on these results.
