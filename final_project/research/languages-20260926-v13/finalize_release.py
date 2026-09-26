"""Check saved release artifacts without reading private or reserved test text."""
from pathlib import Path
from html.parser import HTMLParser
import csv, hashlib, json, shutil

root = Path.cwd()
app = root / 'final_project/phraseflow_multilingual'
out = root / 'final_project/research/languages-20260926-v13'
backup = root / 'final_project/phraseflow_multilingual_v12_20260926'

def digest(p, algorithm='sha256'):
    return hashlib.new(algorithm, p.read_bytes()).hexdigest()

class Deck(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sections = 0
        self.images = []
        self.links = []
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'section': self.sections += 1
        if tag == 'img': self.images.append(a.get('src', ''))
        if tag == 'a': self.links.append(a.get('href', ''))

deck = Deck()
deck.feed((app / 'PhraseFlow_Pitch.html').read_text(encoding='utf-8'))
assert deck.sections == 5, deck.sections
assert len(deck.images) == 3 and all(x.startswith('data:image/') for x in deck.images)
assert deck.links.count('Multilingual_Guide.html') == 3
assert 'Third_Party_Notices.html' in deck.links
assert digest(app / 'Multilingual_Guide.html') == digest(app / 'www/Multilingual_Guide.html')
assert digest(app / 'model.rds', 'md5') == 'bd58c642df1759c4ce1c39b9ace8927c'
assert digest(app / 'model.rds') == digest(root / 'final_project/app/model.rds')
cfg = json.loads((app / 'language-config.json').read_text(encoding='utf-8'))
metrics = json.loads((out / 'metrics.json').read_text(encoding='utf-8'))
assert len(cfg) == 12 and len(metrics) == 8
for code, m in metrics.items():
    rows = list(csv.DictReader((out / f'{code}_case_metrics.csv').open(encoding='utf-8')))
    assert len(rows) == 600
    assert sum(int(r['rank']) == 1 for r in rows) == m['overall']['top1_count']
    assert sum(1 <= int(r['rank']) <= 3 for r in rows) == m['overall']['top3_count']
    assert digest(app / 'languages' / f'{code}.rds', 'md5') == m['model_md5']
assert (backup / 'preview_manifest.json').exists()

# Snapshot final sources as used, while retaining the one-time migration scripts.
for name in ['app.R','predictor.R','PhraseFlow_Pitch.Rpres','Multilingual_Guide.Rmd','pitch.css']:
    shutil.copy2(app / name, out / name)
for name in ['input.js','styles.css']:
    shutil.copy2(app / 'www' / name, out / name)

manifest = dict(author='Sonja Sahebzad', location='Utrecht, the Netherlands',
    brand='Sonja Projects', title='Sonja PhraseFlow', version='1.3 preview', date='2026-09-26',
    status='Local preview; not deployed', languages=list(cfg),
    english_model_unchanged=True, english_model_md5=digest(app / 'model.rds','md5'),
    new_languages_experimental=True, individual_confidence_calibrated=False,
    guide='Multilingual_Guide.html', guide_source='Multilingual_Guide.Rmd',
    slide_deck='PhraseFlow_Pitch.html', slide_source='PhraseFlow_Pitch.Rpres', slides=5,
    previous_preview=str(backup), production_changed=False,
    publication_note='Replace relative guide and credits links with verified public URLs before RPubs publication.',
    historical_artifacts=['PhraseFlow_Pitch-rpubs.html (superseded local export)',
                          'www/PhraseFlow_User_Guide.pdf (original English 1.1 guide)'])
(app / 'preview_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
report = dict(date='2026-09-26', automated=dict(slides=deck.sections,
    embedded_images=len(deck.images), same_guide_served_by_shiny=True,
    languages=12, additional_language_cases=4800, metrics_match_per_case_rows=True,
    english_model_matches_production=True, model_hashes_match_metrics=True),
    browser_review=dict(all_five_slides_visually_checked=True,
        guide_tables_and_charts_checked=True, slide_guide_link_opens=True,
        chinese_no_space_append_checked=True, hindi_combining_marks_checked=True,
        dutch_results_match_saved_metrics=True, author_location_visible=True),
    validation_evidence='checks.json; test_extension.R; source_manifest.json; protocol.json',
    limits=['No native-speaker review yet.',
        'IME pause checked through Shiny tests; full native IME interaction not independently tested.',
        'Local app and deck checked; no production deployment or RPubs publication for version 1.3.',
        'Sampling intervals do not establish per-suggestion confidence.'])
(out / 'release_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
paths=[p for p in app.rglob('*') if p.is_file()]
(out / 'release_sha256.json').write_text(json.dumps({str(p.relative_to(app)).replace('\\','/'):digest(p) for p in sorted(paths)},indent=2),encoding='utf-8')
print(json.dumps(report['automated'],indent=2))
