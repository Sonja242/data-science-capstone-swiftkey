# Compact context and distillation pilot

Author: Sonja Sahebzad. Sonja Projects.

The existing RStudio project is `final_project/Sonja Next Word.Rproj`.
Open `Student_Context_Report.Rmd` and select **Knit** to regenerate the report
from the saved results. This does not retrain a model or use the reserved test.
The report uses the same `milestone-report.css` as earlier project reports.

## Reproduction

Run the following from the **parent corpus project root**, in a fresh preserved
copy without this study's prior generated results. Scripts refuse to overwrite
the registration and completed selection. Reproduction requires the existing
corpus partitions, deployed model, local neural weights and adapter, and the
recorded R/Python packages. Raw text and model weights stay in ignored `data/`.

```r
source("final_project/research/student-context-20260925/prepare_study.R")
```

Using the project's existing Python environment:

```text
python final_project/research/student-context-20260925/student_study.py teacher
python final_project/research/student-context-20260925/student_study.py train
python final_project/research/student-context-20260925/check_precision.py
```

Then in R, from the same parent root:

```r
source("final_project/research/student-context-20260925/evaluate_students.R")
source("final_project/research/student-context-20260925/context_diagnostic.R")
```

The separate `check_export.py` and `check_export.R` perform a neutral cross-runtime
check before outcome analysis. `audit_study.py` recomputes all summaries, confirms
Python/R case-rank agreement, validates exported weight checksums and verifies
that training hashes overlap neither development nor reserved-test hashes.

The initial GPU-reference check found small numerical differences at suggestion
boundaries. `check_precision.py` preserves those results and recomputes the same
unchanged checkpoints with double-precision CPU inference, matching native R's
arithmetic. No retraining or candidate tuning is involved. The precision record
documents the original CUDA settings, errors and changed lists. Final selection
uses the verified native R results.

The repeated development sample is not an independent test. The GPU teacher is
not a deployment candidate. `decision.json` records the selection and whether
the reserved final comparison or any production change occurred. A candidate
must meet every registered gate before the reserved test can be accessed.

No quiz answers, raw corpus lines or user-entered app text are published here.
