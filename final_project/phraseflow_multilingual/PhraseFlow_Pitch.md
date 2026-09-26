Sonja PhraseFlow
========================================================
author: Author: Sonja Sahebzad
date: Utrecht, the Netherlands | September 27, 2026
css: pitch.css
width: 1440
height: 900
transition: none

<div class="cover-line">Your next word,<br>in your language.</div>
<p class="lead">Twelve selectable languages.<br>A compact R app with measured results and a clear user guide.</p>
<p class="launch"><a href="https://sonja242.github.io/data-science-capstone-swiftkey/phraseflow-guide.html" target="_blank">Open the illustrated user guide</a></p>
<p class="small-note">Sonja Projects · Version 1.4<br>English is unchanged. Additional languages are experimental.<br><a href="https://sonjasahebzad.shinyapps.io/sonja-next-word/" target="_blank">Open the live app</a></p>

The prediction model
========================================================

<div class="algorithm-layout"><div>
<h3>A separate model for each language</h3>
<p>English, German, Finnish and Russian use course-corpus text. Eight added languages use Tatoeba example sentences.</p>
<h3>Recent words guide the next word</h3>
<p>A five-gram model uses up to four recent units. Kneser-Ney smoothing blends longer and shorter patterns when evidence is sparse.</p>
<h3>Text processing follows the script</h3>
<p>Chinese uses ICU word segmentation. Korean retains space-delimited units. Accents, Cyrillic and Hindi vowel signs are preserved.</p>
</div><div class="side-stat"><strong>12</strong><span>selectable languages</span><strong>CPU</strong><span>native R inference</span><p>Lazy loading and a bounded cache.<br>No external prediction API.</p></div></div>
<p class="references">Methods: <a href="https://aclanthology.org/P96-1041/">Chen &amp; Goodman</a>; <a href="https://unicode-org.github.io/icu/userguide/boundaryanalysis/">ICU</a>. Additional data: <a href="https://tatoeba.org/en/downloads">Tatoeba contributors, CC BY 2.0 FR</a>.</p>

Using the app and the guide
========================================================

<div class="demo-layout"><div>
<ol><li>Choose a language and enter a phrase.</li><li>Finish a word, then pause or select <b>Predict next word</b>.</li><li>Click a suggestion to keep writing.</li></ol>
<p>Chinese does not require spaces. Choose <b>5, 10 or 20 words</b>. Extra buttons offer short endings from matching training phrases.</p>
<p><a href="https://sonja242.github.io/data-science-capstone-swiftkey/phraseflow-guide.html" target="_blank"><b>Open the illustrated user guide</b></a></p>
<p class="small-note">The <b>User guide / Handleiding</b> button explains both features. Short endings remain experimental; the accuracy charts measure the word model.</p>
</div><div><img class="app-shot" src="www/multilingual-demo.png" alt="Sonja PhraseFlow language selector, phrase input and next-word suggestions."></div></div>

English performance
========================================================

<div class="chart-layout"><div><img class="evidence-chart" src="www/figures/english-performance.png" alt="English first suggestion accuracy 17.0%, top three 28.6%, with 95% intervals."></div>
<div><h3>900 unseen examples</h3><p>First: <b>153/900</b><br>Within three: <b>257/900</b></p><p>Whiskers show 95% sampling intervals. They do not express certainty for one suggestion.</p><h3>Fast local lookup</h3><p>Archived median: <b>0.58 ms</b> for three suggestions. Network and display add time.</p></div></div>
<p class="small-note">Archived English evaluation; test examples were not reopened. All 900 received a suggestion. This is availability, not 100% accuracy. Individual confidence is not calibrated.</p>

Additional language results
========================================================

<div class="language-chart-layout"><div><img class="evidence-chart" src="www/figures/additional-languages.png" alt="Initial accuracy in eight additional languages, with first-suggestion and top-three scores and 95% intervals."></div>
<div><h3>600 cases per language</h3><p>Held-out Tatoeba sentences. Settings were fixed before evaluation.</p><p>Different data, training sizes and word units: these are separate results, not a language ranking.</p><h3>Next improvement</h3><p>Better representative data, native-speaker review and a compact model using more sentence context.</p></div></div>
<p class="references">Lines show Wilson 95% sampling intervals. Similar sentence templates may remain; everyday-message performance is untested. <a href="https://sonja242.github.io/data-science-capstone-swiftkey/phraseflow-guide.html" target="_blank">Full results, earlier prototypes, timing and instructions</a> · <a href="https://sonja242.github.io/data-science-capstone-swiftkey/phraseflow-data-credits.html" target="_blank">Data credits</a>. Sonja Sahebzad · Utrecht, the Netherlands.</p>
