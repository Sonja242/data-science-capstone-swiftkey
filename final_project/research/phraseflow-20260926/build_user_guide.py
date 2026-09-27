"""Build the PhraseFlow PDF and matching editable R Markdown user guide.
Author: Sonja Sahebzad | Sonja Projects.
"""
from pathlib import Path
import json, shutil
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,Table,TableStyle,PageBreak,KeepTogether
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader
ROOT=Path.cwd();OUT=ROOT/'final_project/phraseflow_preview';DOC=OUT/'documentation'
DOC.mkdir(exist_ok=True)
NAVY=colors.HexColor('#123f68');BLUE=colors.HexColor('#1f73a8');PALE=colors.HexColor('#edf5fa');INK=colors.HexColor('#17212b')
pdfmetrics.registerFont(TTFont('Segoe',r'C:/Windows/Fonts/segoeui.ttf'))
pdfmetrics.registerFont(TTFont('SegoeBold',r'C:/Windows/Fonts/segoeuib.ttf'))
pdfmetrics.registerFontFamily('Segoe',normal='Segoe',bold='SegoeBold',italic='Segoe',boldItalic='SegoeBold')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='GuideBody',fontName='Segoe',fontSize=10.3,leading=15,textColor=INK,spaceAfter=9))
styles.add(ParagraphStyle(name='GuideSmall',parent=styles['GuideBody'],fontSize=8.5,leading=12,textColor=colors.HexColor('#526d81')))
styles.add(ParagraphStyle(name='GuideTitle',fontName='SegoeBold',fontSize=28,leading=34,textColor=NAVY,spaceAfter=14))
styles.add(ParagraphStyle(name='GuideHeading',fontName='SegoeBold',fontSize=18,leading=23,textColor=NAVY,spaceBefore=12,spaceAfter=12))
styles.add(ParagraphStyle(name='GuideSub',fontName='SegoeBold',fontSize=11.5,leading=16,textColor=NAVY,spaceBefore=10,spaceAfter=7))
def P(t,style='GuideBody'):return Paragraph(t,styles[style])
def table(data,widths):
 t=Table([[P(str(c),'GuideSmall') for c in row] for row in data],colWidths=widths,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),PALE),('VALIGN',(0,0),(-1,-1),'TOP'),
  ('LINEBELOW',(0,0),(-1,0),.6,BLUE),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d8e5ee')),
  ('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8)]))
 return t
def page(canv,doc):
 canv.setTitle('Sonja PhraseFlow: User Guide');canv.setAuthor('Sonja Sahebzad')
 canv.setSubject('English next-word prediction: use, controls, evidence and limitations')
 canv.setKeywords('Sonja Projects, PhraseFlow, Shiny, next-word prediction, user guide, reproducibility')
 canv.setCreator('Sonja Projects - Python and ReportLab documentation build')
 canv.setStrokeColor(BLUE);canv.setLineWidth(2);canv.line(21*mm,281*mm,189*mm,281*mm)
 canv.setFont('SegoeBold',8);canv.setFillColor(NAVY);canv.drawString(21*mm,285*mm,'SONJA PROJECTS  /  PHRASEFLOW')
 canv.setFont('Segoe',8);canv.setFillColor(colors.HexColor('#526d81'))
 canv.drawString(21*mm,14*mm,'Sonja Sahebzad  |  Version 1.1 preview  |  26 September 2026')
 canv.drawRightString(189*mm,14*mm,f'{doc.page} / 4')
pdf=DOC/'PhraseFlow_User_Guide.pdf'
if pdf.exists():raise FileExistsError('Preserve existing guide.')
story=[P('Sonja PhraseFlow','GuideTitle'),P('A practical guide to your next-word assistant','GuideHeading'),
 P('<b>Author:</b> Sonja Sahebzad<br/><b>Version:</b> 1.1 preview &nbsp; | &nbsp; <b>Date:</b> 26 September 2026','GuideSmall'),
 P('Enter an English phrase, choose a suggested next word and continue writing. PhraseFlow offers a prominent first suggestion, two alternatives and an optional list of further suggestions.')]
shot=OUT/'www/app-preview.png'
if shot.exists():
 from PIL import Image as PILImage
 w,h=PILImage.open(shot).size
 width=min(168*mm,105*mm*w/h)
 story += [Image(str(shot),width=width,height=width*h/w),Spacer(1,4*mm)]
story += [P('Start in three steps','GuideSub'),
 P('<b>1. Write.</b> Type an English phrase or select an example. Finish the last word.'),
 P('<b>2. Predict.</b> Add a space to update suggestions automatically, or select <b>Predict next word</b>. Ctrl + Enter also works; use Command + Enter on Mac.'),
 P('<b>3. Continue.</b> Select a suggestion to append it. Continue typing whenever a different word fits better.'),
 P('This preview keeps the validated CPU prediction model. The longer-context research candidate is evaluated separately and is not silently substituted into the app.','GuideSmall'),PageBreak(),
 P('Controls and everyday use','GuideTitle'),
 table([['Control','What it does'],['Your English phrase','Accepts up to 500 characters. Enter complete words; partial-word autocompletion is not provided.'],
 ['Automatic updates','After a space or punctuation, suggestions refresh following a 150 ms typing pause. Turn this off for manual prediction.'],
 ['Predict next word','Requests a prediction immediately. The displayed computation time excludes the network and screen update.'],
 ['First suggestion / alternatives','Select any suggested word to add it to the phrase. The next suggestions then refresh.'],
 ['More suggestions','Expands ranks 4 to 10. These are additional choices from the same model, not a separate accuracy claim.'],
 ['Clear','Empties the phrase and removes the previous suggestions.'],
 ['Results / How to use','Shows measured model performance and practical instructions, including links to this guide.']],[45*mm,123*mm]),
 P('If the result looks unexpected','GuideSub'),
 P('Check the spelling and complete the last word, then select Predict next word. A longer sentence can help a reader, but the current app uses at most its last four words. New names and unusual expressions may fall back to common continuations.'),
 P('If the session disconnects, reload the page. A first visit may take longer while the hosting service starts the app. Do not interpret startup time as model computation time.'),
 P('Privacy','GuideSub'),P('The Shiny server processes the phrase for the current session. The app does not save phrases or send them to an external prediction service. Avoid entering confidential information into a public demonstration.'),PageBreak(),
 P('Understanding the results','GuideTitle'),
 P('A useful suggestion and an exact match are different outcomes. Several words can sensibly continue a sentence; this evaluation counts a prediction as correct only when it matches the recorded next word.'),
 table([['Measure','Recorded result','Meaning'],['First suggestion correct','153 / 900 = 17.0%<br/>95% interval: 14.7% to 19.6%','The first word matched the held-out next word.'],
 ['Correct within three','257 / 900 = 28.6%<br/>95% interval: 25.7% to 31.6%','At least one of the first three matched.'],
 ['Prediction returned','900 / 900 = 100%','Every example received a suggestion. This does not mean every answer was correct.'],
 ['Local computation','Median 0.58 ms<br/>95th percentile 0.83 ms','3,000 warm calls. Loading, typing delay, network and display time excluded.']],[48*mm,55*mm,65*mm]),
 P('What the intervals tell you','GuideSub'),P('The 95% intervals describe uncertainty in the measured success rates under the sampling assumptions. They are not the probability that a particular displayed word is correct.'),
 P('Why there is no confidence percentage beside a word','GuideSub'),P('The current model has not been calibrated to make reliable confidence claims. A high model score can still produce an unsuitable suggestion. Suggestions are therefore shown in rank order, without a misleading certainty percentage.'),
 P('Scope of the evidence','GuideSub'),P('The figures above belong to the unchanged deployed model and its previously recorded test. This preview adds interface and documentation features. New reranking experiments use separate development results and have not replaced these test figures.'),
 P('Quiz scores describe small, previously exposed multiple-choice exercises. They do not establish accuracy on unrestricted new writing.','GuideSmall'),PageBreak(),
 P('Method and document record','GuideTitle'),
 P('The model behind this preview','GuideSub'),P('The CPU model learns recurring word sequences from 3.21 million English training lines from blogs, news and Twitter. It combines patterns of up to five words with interpolated Kneser-Ney smoothing. Shorter patterns provide support when a longer pattern is uncommon. A retained vocabulary of 50,000 words and pruning keep the model compact.'),
 P('Research toward longer context','GuideSub'),P('A separate experiment tests a small neural reranker that reads up to 48 words and reorders 128 candidates. A frozen Qwen3 teacher supplies training guidance. This is a hypothesis being measured, not proof that the model understands a whole sentence or achieves perfect prediction.'),
 table([['Document property','Value'],['Title','Sonja PhraseFlow: User Guide'],['Author / publisher','Sonja Sahebzad / Sonja Projects'],['Version / issue date','1.1 preview / 26 September 2026'],['Application','Python and ReportLab documentation build'],['Companion source','PhraseFlow_User_Guide.Rmd and reproducible build_user_guide.py'],['Release status','Local review copy. Existing public product: Sonja Next Word.']],[48*mm,120*mm]),
 P('References and further reading','GuideSub'),
 P('1. Chen &amp; Goodman (1996). <link href="https://aclanthology.org/P96-1041/" color="#1f73a8">An Empirical Study of Smoothing Techniques for Language Modeling</link>. ACL. Basis for smoothing comparisons.','GuideSmall'),
 P('2. Qwen Team (2025). <link href="https://arxiv.org/abs/2505.09388" color="#1f73a8">Qwen3 Technical Report</link>. Technical report for the teacher model family; its benchmarks are not this app\'s accuracy.','GuideSmall'),
 P('3. <link href="https://shiny.posit.co/" color="#1f73a8">Posit Shiny documentation</link>. The web application framework.','GuideSmall'),
 P('4. <link href="https://github.com/Sonja242/data-science-capstone-swiftkey/tree/main/final_project" color="#1f73a8">Sonja Projects: project source and published evidence</link>. The local PhraseFlow research supplement contains the new experiment and current literature review.','GuideSmall')]
doc=SimpleDocTemplate(str(pdf),pagesize=(210*mm,297*mm),leftMargin=21*mm,rightMargin=21*mm,topMargin=24*mm,bottomMargin=23*mm)
doc.build(story,onFirstPage=page,onLaterPages=page)
reader=PdfReader(str(pdf));assert len(reader.pages)==4,len(reader.pages)
assert reader.metadata.author=='Sonja Sahebzad'
shutil.copy2(pdf,OUT/'www'/pdf.name)
print(pdf)
