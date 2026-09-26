# Sonja PhraseFlow 1.4

Author: Sonja Sahebzad · Utrecht, the Netherlands · Sonja Projects

Twelve language models, 5/10/20 visible next-word suggestions and experimental short continuations. The English model is unchanged. Additional languages and ending suggestions are experimental; individual confidence is not calibrated.

- [Live Shiny app](https://sonjasahebzad.shinyapps.io/sonja-next-word/)
- [Five-slide RStudio Presenter pitch on RPubs](https://rpubs.com/Sonja_Janssen/sonja-next-word)
- [User guide, figures and results](https://sonja242.github.io/data-science-capstone-swiftkey/phraseflow-guide.html)
- [Data credits and training attribution](https://sonja242.github.io/data-science-capstone-swiftkey/phraseflow-data-credits.html)

## Run in RStudio

Open `Sonja PhraseFlow.Rproj`, install `shiny`, `data.table`, `jsonlite` and `stringi` if needed, open `app.R`, and select **Run App**. The supplied compact models need no retraining, Python, GPU or external prediction API. The cache retains English and at most three other models. Short continuations match training-text suffixes; they do not establish understanding of the full sentence.

## Editable documents and evidence

`Multilingual_Guide.Rmd` and its knitted HTML are stored together. Rebuild the guide from the repository with `rmarkdown::render("final_project/phraseflow_multilingual/Multilingual_Guide.Rmd")`; its tables use stored aggregate metrics, without reopening reserved English test examples.

`PhraseFlow_Pitch.Rpres` is the five-slide native RStudio Presenter source. In RStudio open the source using `.rs.showPresentation("PhraseFlow_Pitch.Rpres")`, then **More > Save as Web Page**. `PhraseFlow_Pitch.html` is the native export with public guide and credit links. It embeds the app illustration and two static scientific charts; those charts have captions and intervals, not hover tooltips.

Research scripts and evidence are in `../research/multilingual-20260926`, `../research/languages-20260926-v13` and `../research/continuations-20260926-v14`. R and Python code, fixed protocols, anonymized per-case outcomes and timing measurements are retained. One-time build scripts are documented as migrations, not repeatable installers. Private corpus text, held-out answers and course quiz answers are excluded from publication.

Earlier local previews and original English documentation are preserved on the author's laptop. `../publication-20260927` records this release and its checks. Coursera submission and peer-review messages require the author's review and have not been sent.
