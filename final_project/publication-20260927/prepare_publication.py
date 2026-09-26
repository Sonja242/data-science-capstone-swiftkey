from pathlib import Path
import json, shutil, hashlib, re
from datetime import datetime

root = Path('C:/Users/csj50/OneDrive/Documents/Sonja report/Data Science Capstone')
app = root/'final_project/phraseflow_multilingual'
portfolio = root.parent/'sonja-data-science-portfolio'
pub = root/'final_project/publication-20260927'
backup = root/'backups'/('phraseflow-publication-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
pub.mkdir(exist_ok=True)
backup.mkdir(parents=True, exist_ok=False)
site = 'https://sonja242.github.io/data-science-capstone-swiftkey/'
guide = site+'phraseflow-guide.html'
credits = site+'phraseflow-data-credits.html'
live = 'https://sonjasahebzad.shinyapps.io/sonja-next-word/'
pitch = 'https://rpubs.com/Sonja_Janssen/sonja-next-word'

def save(p, value):
    if p.exists():
        rel = p.relative_to(root) if p.is_relative_to(root) else Path('portfolio')/p.relative_to(portfolio)
        b = backup/rel
        b.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p,b)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(value,encoding='utf-8')

for name in ['PhraseFlow_Pitch.Rpres','PhraseFlow_Pitch.md','PhraseFlow_Pitch.html']:
    p=app/name
    s=p.read_text(encoding='utf-8')
    s=s.replace('September 26, 2026','September 27, 2026')
    s=s.replace('Local preview 1.4','Version 1.4').replace('Current public English app','Open the live app')
    s=s.replace('href="Multilingual_Guide.html"',f'href="{guide}"')
    s=s.replace('href="Third_Party_Notices.html"',f'href="{credits}"')
    save(p,s)

p=app/'app.R'
s=p.read_text(encoding='utf-8').replace('Multilingual preview 1.4. 26 September 2026.','Version 1.4. 27 September 2026.').replace('Preview 1.4','Version 1.4')
save(p,s)
p=app/'Multilingual_Guide.Rmd'
s=p.read_text(encoding='utf-8').replace('26 September 2026 | Sonja Projects | Preview 1.4','27 September 2026 | Sonja Projects | Version 1.4')
s=s.replace('This preview covers Mandarin','This version covers Mandarin')
s=s.replace('(Third_Party_Notices.html)',f'({credits})')
s=s.replace('The existing public app and RPubs presentation have not been replaced by this local preview.',f'The [live app]({live}) and [five-slide presentation]({pitch}) accompany this guide. The [source repository](https://github.com/Sonja242/data-science-capstone-swiftkey/tree/main/final_project/phraseflow_multilingual) includes the R app, editable guide and native Presenter source. Python and R research scripts are retained separately from the CPU app.')
save(p,s)

readme=f'''# Sonja PhraseFlow 1.4

Author: Sonja Sahebzad · Utrecht, the Netherlands · Sonja Projects

Twelve language models, 5/10/20 visible next-word suggestions and experimental short continuations. The English model is unchanged. Additional languages and ending suggestions are experimental; individual confidence is not calibrated.

- [Live Shiny app]({live})
- [Five-slide RStudio Presenter pitch on RPubs]({pitch})
- [User guide, figures and results]({guide})
- [Data credits and training attribution]({credits})

## Run in RStudio

Open `Sonja PhraseFlow.Rproj`, install `shiny`, `data.table`, `jsonlite` and `stringi` if needed, open `app.R`, and select **Run App**. The supplied compact models need no retraining, Python, GPU or external prediction API. The cache retains English and at most three other models. Short continuations match training-text suffixes; they do not establish understanding of the full sentence.

## Editable documents and evidence

`Multilingual_Guide.Rmd` and its knitted HTML are stored together. Rebuild the guide from the repository with `rmarkdown::render("final_project/phraseflow_multilingual/Multilingual_Guide.Rmd")`; its tables use stored aggregate metrics, without reopening reserved English test examples.

`PhraseFlow_Pitch.Rpres` is the five-slide native RStudio Presenter source. In RStudio open the source using `.rs.showPresentation("PhraseFlow_Pitch.Rpres")`, then **More > Save as Web Page**. `PhraseFlow_Pitch.html` is the native export with public guide and credit links. It embeds the app illustration and two static scientific charts; those charts have captions and intervals, not hover tooltips.

Research scripts and evidence are in `../research/multilingual-20260926`, `../research/languages-20260926-v13` and `../research/continuations-20260926-v14`. R and Python code, fixed protocols, anonymized per-case outcomes and timing measurements are retained. One-time build scripts are documented as migrations, not repeatable installers. Private corpus text, held-out answers and course quiz answers are excluded from publication.

Earlier local previews and original English documentation are preserved on the author's laptop. `../publication-20260927` records this release and its checks. Coursera submission and peer-review messages require the author's review and have not been sent.
'''
save(app/'README.md',readme)

manifest=json.loads((app/'preview_manifest.json').read_text(encoding='utf-8'))
manifest.update(version='1.4',date='2026-09-27',status='Prepared for public release; deployment evidence is stored separately',previous_preview='../phraseflow_multilingual_v13_20260926',publication_note='Deck and guide use public links. See ../publication-20260927/publication.json for verified deployment status.')
save(app/'preview_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')

p=root/'docs/index.html';s=p.read_text(encoding='utf-8')
s=s.replace('<h1>English Corpus Analysis</h1>','<h1>Sonja PhraseFlow &amp; NLP Research</h1>')
s=s.replace('<h2>Sonja Next Word: Final Product</h2>','<h2>Sonja PhraseFlow: Final Product</h2>')
s=s.replace('A live Shiny app with clickable next-word suggestions, a five-slide product pitch, and an independent evaluation of the compact CPU model.','Twelve selectable languages, 5/10/20 word choices and experimental short continuations, with an illustrated guide and measured results. Native R inference on a CPU.')
s=s.replace('<a href="final-product-report.html">Methods, results and reproducibility</a>',f'<a href="phraseflow-guide.html">User guide, figures and current results</a><br>\n        <a href="phraseflow-pitch.html">Five-slide HTML copy</a><br>\n        <a href="final-product-report.html">Archived English product evaluation</a>')
save(p,s)

p=root/'README.md';s=p.read_text(encoding='utf-8')
s=s.replace('## Final product: Sonja Next Word','## Final product: Sonja PhraseFlow 1.4')
s=s.replace('- [Final product report](https://sonja242.github.io/data-science-capstone-swiftkey/final-product-report.html)',f'- [Illustrated user guide, figures and results]({guide})\n- [Archived English product report](https://sonja242.github.io/data-science-capstone-swiftkey/final-product-report.html)')
s=s.replace('- [Run and reproduce the product](final_project/README.md)','- [Run the current app and open the editable RStudio documents](final_project/phraseflow_multilingual/README.md)\n- [Review pack and submission links](final_project/Submission_Details.md)')
s=s.replace('The deployed compact model is evaluated separately','The app now offers twelve language choices, 5/10/20 visible words and experimental short endings. Additional languages use separately evaluated models; their results do not demonstrate improved English accuracy or reliable full-sentence meaning.\n\nThe unchanged English compact model is evaluated separately')
save(p,s)
p=root/'final_project/README.md';s=p.read_text(encoding='utf-8')
s='> **Latest release:** [Sonja PhraseFlow 1.4](phraseflow_multilingual/README.md) contains the current multilingual app, guide and five-slide deck. The material below documents the original English product and remains an archived reference.\n\n'+s
save(p,s)

p=portfolio/'app.R';s=p.read_text(encoding='utf-8')
s=s.replace('title = "Sonja Next Word"','title = "Sonja PhraseFlow"',1)
s=s.replace('A compact English word-completion app with clickable suggestions, a five-slide pitch, and transparent evaluation on 900 previously unused examples.','A compact app with twelve prediction languages, 5/10/20 word choices and experimental short continuations. An illustrated guide explains the measured results and limitations.',1)
s=s.replace('c("Live next-word suggestions", "Independent accuracy and timing evidence", "Reproducible RStudio product and pitch")','c("Twelve languages and clickable suggestions", "User guide with figures and uncertainty intervals", "Five-slide pitch and reproducible R/Python sources")',1)
s=s.replace('report = "https://sonja242.github.io/data-science-capstone-swiftkey/final-product-report.html",',f'report = "{guide}",\n    report_label = "User guide & results",\n    pitch = "{pitch}",',1)
s=s.replace('rel = "noopener", "Report")','rel = "noopener", if (is.null(project$report_label)) "Report" else project$report_label),\n        if (!is.null(project$pitch)) tags$a(class = "text-link", href = project$pitch, target = "_blank", rel = "noopener", "Five-slide pitch")',1)
save(p,s)

ignore=root/'.gitignore'
s=ignore.read_text(encoding='utf-8')+'\n# Preserved previews and generated training intermediates (not public releases)\nfinal_project/phraseflow_multilingual_v*/\nfinal_project/phraseflow_preview/\nfinal_project/research/continuations-20260926-v14/*_index.tsv\nfinal_project/phraseflow_multilingual/PhraseFlow_Pitch-rpubs.html\nfinal_project/phraseflow_multilingual/documentation/\nfinal_project/phraseflow_multilingual/www/PhraseFlow_User_Guide.pdf\nfinal_project/phraseflow_multilingual/www/multilingual-preview.png\n'
save(ignore,s)

review=f'''# Final project: author review before submission

Author: Sonja Sahebzad · Utrecht, the Netherlands · Sonja Projects

**Status: publication verification in progress. Not submitted to Coursera. No peer-review message sent.**

**Project title:** Sonja PhraseFlow: Multilingual Next-Word Prediction

**Shiny app:** {live}

**Five-slide RPubs deck:** {pitch}

**User guide and measured results:** {guide}

**GitHub source:** https://github.com/Sonja242/data-science-capstone-swiftkey

**Portfolio:** https://sonjasahebzad.shinyapps.io/data-science-portfolio/

## Author review

1. Open the app, keep English selected and enter a phrase. Check that the large first suggestion is one word.
2. Click a suggestion and continue writing. Try the 5/10/20 choices, another language and the User guide / Handleiding button.
3. Open RPubs without signing in. Review all five slides, graphs and the guide link.
4. Confirm the project title and the two submission URLs above. Submit only after author approval.

The course rubric requires an operating Shiny app and a deck of at most five RStudio Presenter slides. The app predicts from a multiword input; the optional ending buttons supplement the prominent single-word prediction. Slide 2 explains the algorithm, slide 3 demonstrates use, and slides 4–5 provide quantitative results and limitations. Evidence of live checks is retained in `publication-20260927`.

The English score is 17.0% first-choice and 28.6% top-three on the archived 900-case evaluation. Those percentages are not per-suggestion confidence. Extra endings have no independent accuracy evaluation. No new reserved English test cases were opened for this release.

## Peer-review message — draft only

Hello everyone,

My final capstone project, **Sonja PhraseFlow**, is available to explore. It predicts the next word from an English phrase, with additional experimental language options and an illustrated user guide.

- App: {live}
- Five-slide presentation: {pitch}
- User guide and results: {guide}

After submission, the Coursera review link will be added here. Feedback on usability, clarity of the explanation and the reported evaluation would be appreciated. Please assess the project using the course rubric.

Thank you,\nSonja Sahebzad

The actual Coursera peer-review link can only be confirmed after submission. The app and presentation links are not substitutes for that review link. This draft has not been posted or sent.
'''
save(root/'final_project/Submission_Details.md',review)
save(pub/'Peer_Review_Draft.md',review.split('## Peer-review message — draft only\n\n')[1])
save(pub/'README.md','''# PhraseFlow 1.4 publication record

Author: Sonja Sahebzad · Sonja Projects · 27 September 2026

This directory records release preparation, exact file allowlists and publication verification. Deployment credentials and RPubs update receipts remain in ignored local rsconnect directories. Private data and course quiz answers are excluded. Previous app and document files are preserved under the ignored project backups directory.

The deck was originally exported with native RStudio Presenter. Publication changes update its date, version label and absolute documentation links alongside the matching .Rpres source; its five slides and embedded figures are retained.

Coursera submission and peer-review posting are withheld for author review.
''')
shutil.copy2(Path(__file__),pub/'prepare_publication.py')
print(json.dumps({'prepared':True,'backup':str(backup),'public_guide':guide,'course_submitted':False,'peer_message_sent':False},indent=2))
