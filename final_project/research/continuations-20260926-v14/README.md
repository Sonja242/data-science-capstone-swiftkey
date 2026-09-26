# Sonja PhraseFlow 1.4 — visible choices and short continuations

Author: Sonja Sahebzad · Utrecht, the Netherlands · Sonja Projects

The local app now offers 5, 10 or 20 visible word suggestions, plus up to three supplementary endings from existing training text. A click inserts the selected text and updates prediction. Version 1.3 is preserved in `final_project/phraseflow_multilingual_v13_20260926`. No public deployment or course submission was performed.

## Method and scope

`build_continuations.py` reads only the twelve existing training CSVs listed with SHA-256 hashes in `training_manifest.json`. It collects the final one to three normalized units of each training line, with the preceding two to four units as the context. Up to eight endings per context survive frequency-based pruning, with a lexical tie-break. `compact_indexes.R` removes contexts with fewer than two retained occurrences and saves keyed R tables. This avoids serving hundreds of thousands of one-off contexts per language.

`continuations.R` matches the longest available suffix of the input. With a two-unit match it returns the first next unit only; stronger three- or four-unit matches can return the saved full ending. Repeated first units are grouped by frequency for two-unit matches. Already-visible word suggestions are omitted from the extra buttons. Chinese output removes segmentation spaces. If no supported context matches, no extra ending is offered.

These are local phrase matches, not a semantic model of the whole sentence. No rule hardcodes the user's Dutch examples. The hand-inspected examples in `demonstration_examples.json` are demonstrations, not accuracy evidence. The word model's ranking and scores are unchanged. Wider choice and supplementary endings do not demonstrate a higher top-1 accuracy or calibrated confidence.

## Reproduction and files

Use the Data Science Capstone root as the working directory. A clean reproduction starts from the preserved v1.3 app and the original training CSVs. Preserve completed output: build scripts refuse existing experiment and backup directories. In an isolated project copy, run the Python build, the R compilation and compaction, then apply the UI migration. The authoritative final implementation is also retained directly as `app.R`, `predictor.R`, `continuations.R`, `input.js` and `styles.css`; these can be inspected without replaying one-time migration scripts.

The one-time UI migration order is `update_app.py`, `fix_language_typing.py`, `fix_programmatic_input.py`, `fix_append_refresh.py`, then `stable_buttons.py`. Stable completion IDs avoid selecting a different ending if the number of visible word choices changes. The browser sends both text and its language so stale results cannot be used across language changes. Server-originated insertions use a native input event to trigger the same update path as typing.

`test_features.R` checks all twelve languages, manual and automatic predictions, composition pause, the cache bound and identical English words and scores over all 600 development examples. `test_continuations.R` checks suffix matching, no-match behaviour, exclusion of duplicate word options, Chinese spacing and unchanged top ten results when twenty candidates are requested. It also records CPU time on the same development prefixes; these reused prefixes are not an independent quality test. The reserved English 900-case set was not opened.

Browser review exercised 5/10/20 visible choices, Dutch single-word and two-word insertion, immediate typing after changing language and Chinese insertion without spaces. The full native IME interaction remains outside this browser review. Native-speaker review and independent quality evaluation of the ending lookup remain outstanding.

`Multilingual_Guide.Rmd` was knitted to self-contained HTML and copied to the app's `www` directory. The existing graphs still describe the word model. The five-slide `PhraseFlow_Pitch.Rpres` was exported using RStudio Presenter, including the refreshed interface illustration. Both current HTML outputs are kept beside their editable sources in the app project. Before public RPubs publication, replace relative guide and credit links with verified public URLs.
