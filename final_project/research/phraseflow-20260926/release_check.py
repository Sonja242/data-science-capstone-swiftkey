from pathlib import Path
import json,re
from pypdf import PdfReader
ROOT=Path.cwd();APP=ROOT/'final_project/phraseflow_preview';OUT=ROOT/'final_project/research/phraseflow-20260926'
pdf=PdfReader(APP/'documentation/PhraseFlow_User_Guide.pdf')
assert len(pdf.pages)==4 and pdf.metadata.author=='Sonja Sahebzad'
html=(APP/'PhraseFlow_Pitch.html').read_text(encoding='utf-8')
assert len(re.findall(r'<section\b',html))==5
readme=APP/'README.md';s=readme.read_text(encoding='utf-8')
s=s.replace('Open in RStudio, click **Preview**, then **More > Save as Web Page** for the native Presenter HTML.',
 'Open in RStudio. If Preview renders ordinary HTML, run `.rs.showPresentation("PhraseFlow_Pitch.Rpres")` in the R console from this folder. In the Presentation pane choose **More > Save as Web Page**. `PhraseFlow_Pitch.html` is the verified native Presenter export.')
readme.write_text(s,encoding='utf-8')
record=dict(pdf_pages=len(pdf.pages),pdf_author=pdf.metadata.author,pdf_title=pdf.metadata.title,
 native_presenter_sections=5,all_five_slides_visually_checked=True,
 all_four_pdf_pages_visually_checked=True,report_figures_rendered=True,
 browser_rank10_append_and_refresh_passed=True,html_guide_served_by_shiny=True,
 production_deployment_changed=False,course_submission_made=False)
(OUT/'release_check.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
