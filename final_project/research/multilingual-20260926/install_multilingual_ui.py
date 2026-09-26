from pathlib import Path
import shutil
ROOT=Path.cwd();OUT=ROOT/'final_project/research/multilingual-20260926';APP=ROOT/'final_project/phraseflow_multilingual'
shutil.copy2(OUT/'multilingual_app.R',APP/'app.R')
p=APP/'www/input.js';js=p.read_text(encoding='utf-8')
js=js.replace("    const stale = card.getAttribute('data-prediction-phrase') !== box.value;", "    const language = document.getElementById('language');\n    const stale = card.getAttribute('data-prediction-phrase') !== box.value || !language || card.getAttribute('data-prediction-language') !== language.value;")
js=js.replace("card.getAttribute('data-prediction-phrase') !== phraseBox().value)","card.getAttribute('data-prediction-phrase') !== phraseBox().value || card.getAttribute('data-prediction-language') !== document.getElementById('language').value)")
js=js.replace("  document.addEventListener('click',", "  document.addEventListener('change', function(event) {\n    if (event.target.id !== 'language') return;\n    if (phraseBox()) phraseBox().value = '';\n    guardSuggestions(); sendCurrentText();\n  }, true);\n  document.addEventListener('click',")
p.write_text(js,encoding='utf-8')
pitch=(ROOT/'final_project/phraseflow_preview/PhraseFlow_Pitch.Rpres').read_text(encoding='utf-8')
pitch=pitch.replace('Let your next<br>word flow.','Your next word,<br>in your language.')
pitch=pitch.replace('English word suggestions in a clear, responsive interface.<br>Built in R. Measured on unseen text. Ready to explore.',
 'English, Deutsch, Suomi and Русский.<br>A local multilingual preview with separate language models.')
pitch=pitch.replace('Five slides. PhraseFlow is a local product preview using the unchanged deployed model.<br>The public app retains the name Sonja Next Word pending review.',
 'Five slides. English retains the deployed model. Three new languages are experimental.<br>The public English app remains Sonja Next Word pending review.')
pitch=pitch.replace('3.21 million training lines from English blogs, news and Twitter in the course corpus.',
 'English retains its 3.21-million-line model. Each new language uses up to 75,000 separate course-corpus lines.')
pitch=pitch.replace('CPU predictions. A fallback handles unfamiliar contexts.',
 'CPU predictions. A language selector loads the corresponding model.')
pitch=pitch.replace('<span>compressed model</span>','<span>English model on disk</span>')
pitch=pitch.replace('<span>possible next words</span>','<span>English vocabulary</span>')
pitch=pitch.replace('<ol><li>Enter an English phrase.</li>', '<ol><li>Choose a language and enter a phrase.</li>')
pitch=pitch.replace('The preview offers three prominent suggestions and seven optional alternatives.',
 'Three main suggestions and seven optional alternatives. Changing language resets the phrase.')
pitch=pitch.replace('www/app-preview.png','www/multilingual-preview.png')
pitch=pitch.replace('Local PhraseFlow preview with English phrase input, three main suggestions and an expandable list of further words.',
 'PhraseFlow multilingual preview showing a language selector, phrase input and ranked suggestions.')
pitch=pitch.replace('Measured performance, clearly defined','English model performance')
pitch=pitch.replace('Archived evaluation of the unchanged deployed model: 900 unseen examples,',
 'Archived English evaluation only: 900 unseen examples,')
pitch=pitch.replace('A usable product and a disciplined next step','Language coverage and next steps')
start=pitch.index('<div class="closing-layout">');end=pitch.index('<p class="references">',start)
pitch=pitch[:start]+'''<div class="closing-layout">
<div><h3>Four working languages</h3><p>English, German, Finnish and Russian. Separate models retain accents and Cyrillic letters. The app shows evidence for the selected language.</p><h3>Initial new-language results</h3><p id="language-score-copy">LANGUAGE_SCORES</p></div>
<div><h3>A foundation for expansion</h3><p>Add a corpus and tokenizer, train a compact model, then evaluate before adding the language to the menu. This release does not cover every language.</p><h3>Quality remains the priority</h3><p>Improve sentence context and representation, validate independently and calibrate confidence. More languages alone do not improve prediction accuracy.</p></div>
</div>
'''+pitch[end:]
pitch=pitch.replace('New preview and study are local review artifacts.', 'Language prototypes are local and experimental.')
(APP/'PhraseFlow_Pitch.Rpres').write_text(pitch,encoding='utf-8')
css=APP/'pitch.css';s=css.read_text(encoding='utf-8')
s+='\n/* Product identity at the very top of every slide. */\n.reveal .slides>section{padding-top:70px!important}\n.reveal .slides>section::before{content:"SONJA PHRASEFLOW   /   SONJA PROJECTS   /   SONJA SAHEBZAD";position:absolute;top:17px;left:25px;font-family:Calibri,"Segoe UI",sans-serif;font-size:19px;letter-spacing:1.4px;color:#1f73a8;font-weight:600}\n.reveal .cover-line{font-size:72px!important;margin:18px 0!important}\n.reveal .cover-kicker{display:none}\n'
css.write_text(s,encoding='utf-8')
print('Multilingual app and five-slide source installed. Top identity bar added.')
