from pathlib import Path
import html,json,shutil
root=Path.cwd();app=root/'final_project/phraseflow_multilingual';out=root/'final_project/research/languages-20260926-v13'
for name in ['Multilingual_Guide.Rmd','PhraseFlow_Pitch.Rpres']:
 shutil.copy2(out/name,app/name)
p=app/'PhraseFlow_Pitch.Rpres';s=p.read_text(encoding='utf-8')
s=s.replace('Sonja Projects · Preview 1.3<br>English is unchanged. Additional languages are experimental.', 'Sonja Projects · Local preview 1.3<br>English is unchanged. Additional languages are experimental.<br><a href="https://sonjasahebzad.shinyapps.io/sonja-next-word/" target="_blank">Current public English app</a>')
p.write_text(s,encoding='utf-8')
css=app/'pitch.css';s=css.read_text(encoding='utf-8')
s+='\n.reveal .chart-layout{display:grid;grid-template-columns:64% 32%;gap:4%;align-items:center}.reveal .language-chart-layout{display:grid;grid-template-columns:64% 32%;gap:4%;align-items:start}.reveal .evidence-chart{width:100%;max-height:660px!important;object-fit:contain}.reveal .chart-layout p,.reveal .language-chart-layout p{font-size:25px}.reveal .language-chart-layout h3{font-size:29px!important}.reveal .language-chart-layout+.references{margin-top:13px}.reveal .demo-layout .small-note{font-size:22px}\n'
css.write_text(s,encoding='utf-8')
langs=json.loads((app/'extension-languages.json').read_text(encoding='utf-8'))
links=''.join(f'<li>{html.escape(d["name"])}: <a href="attribution/{code}_training_credits.tsv.gz">training contributors and sentence URLs (TSV, gzip)</a></li>' for code,d in langs.items())
credits='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sonja PhraseFlow — Data credits</title><style>body{font:18px/1.65 "Segoe UI",Calibri,sans-serif;color:#15344e;max-width:950px;padding:30px;margin:auto}h1,h2{color:#123f68}a{color:#1f73a8}li{margin:8px 0}</style><h1>Training-data credits</h1><p>Sonja PhraseFlow · Sonja Projects<br>Author: Sonja Sahebzad · Utrecht, the Netherlands</p><h2>Tatoeba</h2><p>The eight additional models use sentences by Tatoeba contributors, downloaded from the <a href="https://tatoeba.org/en/downloads">official exports</a> dated 26 September 2026, under <a href="https://creativecommons.org/licenses/by/2.0/fr/">Creative Commons Attribution 2.0 France</a>. Text was normalized, segmented, sampled and aggregated into statistical models. Tatoeba does not endorse this app.</p><p>The files below preserve the credited contributor, sentence ID and original sentence URL for each training sentence. A missing owner in the original export is preserved as supplied; the sentence page provides its history. Held-out evaluation answers are not included.</p><ul>'''+links+'''</ul><h2>Earlier models</h2><p>English, German, Finnish and Russian use the <a href="https://d396qusza40orc.cloudfront.net/dsscapstone/dataset/Coursera-SwiftKey.zip">official Coursera SwiftKey corpus</a>. The project retains its source archive and checksum. No course quiz text or answers are included in these attribution lists.</p><h2>Software and method</h2><p>The app uses R, Shiny, data.table, jsonlite and stringi. Chinese segmentation uses <a href="https://unicode-org.github.io/icu/userguide/boundaryanalysis/">ICU dictionary word boundaries</a>. The compact language models use interpolated Kneser-Ney smoothing. Download checksums, runtime versions and transformations are recorded with the experiment scripts.</p><p><a href="Multilingual_Guide.html">Return to the user guide and results</a></p></html>'''
(app/'www/Third_Party_Notices.html').write_text(credits,encoding='utf-8')
(app/'Third_Party_Notices.html').write_text(credits.replace('href="attribution/','href="www/attribution/'),encoding='utf-8')
(app/'README.md').write_text('''# Sonja PhraseFlow — multilingual preview 1.3

Sonja Sahebzad · Utrecht, the Netherlands · Sonja Projects

Open `Sonja PhraseFlow.Rproj`, open `app.R`, and use Run App. Twelve choices have actual local models. English is unchanged. Other languages are experimental. The interface is English; selecting a language changes prediction, not translation. The cache retains English and at most three other models.

`Multilingual_Guide.Rmd` and its knitted HTML contain instructions, graphs, fixed protocols, independent sentence evaluation, intervals and limitations. The visible User guide / Handleiding button links to it. Data credits are in `Third_Party_Notices.html`. Original English documentation is retained as a version 1.1 artifact.

The five-slide source is `PhraseFlow_Pitch.Rpres`. In RStudio use `.rs.showPresentation("PhraseFlow_Pitch.Rpres")` and More > Save as Web Page for native Presenter HTML. Relative guide and credit links work when served with these sibling HTML files. Before RPubs publication, give them verified public URLs; a standalone RPubs deck cannot resolve local sibling files.

The previous four-language preview is preserved in `../phraseflow_multilingual_v12_20260926`. New scripts, protocol, manifests, metrics and per-case outcomes are in `../research/languages-20260926-v13`. Private text and held-out answers stay in the project data folder. Do not publish those private files.

No production deployment or course submission was made for this local preview. Measured accuracy is not calibrated confidence and is not 100%.
''',encoding='utf-8')
print('Guide, five-slide source, data credits and author location saved.')
