from pathlib import Path
import json, shutil, hashlib
root=Path('C:/Users/csj50/OneDrive/Documents/Sonja report/Data Science Capstone')
pub=root/'final_project/publication-20260927'
app=root/'final_project/phraseflow_multilingual'
manifest=json.loads((app/'preview_manifest.json').read_text(encoding='utf-8'))
manifest.update(status='Published; awaiting author review before Coursera submission',production_changed=True)
(app/'preview_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
checks=[
 {'source':'news','phrase':'the all volunteer pirates offer little','prediction':'girl','server_ms':49},
 {'source':'news','phrase':"when they start playing games in akron or here we're going to",'prediction':'be','server_ms':52},
 {'source':'twitter','phrase':'when will they learn dont slide','prediction':'into','server_ms':5},
 {'source':'twitter','phrase':'i was born ready always curious','prediction':'and','server_ms':4},
 {'source':'twitter','phrase':'great day so far lots accomplished lots yet to do more in','prediction':'the','server_ms':8}]
(pub/'live-browser-checks.json').write_text(json.dumps({'date':'2026-09-27','url':'https://sonjasahebzad.shinyapps.io/sonja-next-word/','version':'1.4','purpose':'Functional rubric check, not a new accuracy benchmark. These five previously documented course-corpus prefixes were reused. No target words were loaded.','cases':checks,'all_received_single_word_prediction':True,'timing_scope':'Displayed server processing time; excludes network and browser display. Five calls are not a runtime benchmark.','additional_checks':['Twelve choices present in the language menu','Dutch: we houden van -> elkaar, with 19 alternatives when twenty choices selected','Clicking elkaar appends the word and refreshes predictions','Portfolio card links to latest app, guide and RPubs pitch','Public RPubs deck has five slides and three embedded images'],'course_submitted':False,'peer_message_sent':False},indent=2)+'\n',encoding='utf-8')
status={'date':'2026-09-27','version':'1.4','author':'Sonja Sahebzad','location':'Utrecht, the Netherlands','release_commit':'24acdb6d74872592075030069a3fa411f0704773','github':'https://github.com/Sonja242/data-science-capstone-swiftkey','pages':'https://sonja242.github.io/data-science-capstone-swiftkey/','app_url':'https://sonjasahebzad.shinyapps.io/sonja-next-word/','app_bundle_id':'12610552','deck_url':'https://rpubs.com/Sonja_Janssen/sonja-next-word','deck_static_url':'https://rstudio-pubs-static.s3.amazonaws.com/1462310_b965f3714f624eaface25f8d252f410a.html','guide_url':'https://sonja242.github.io/data-science-capstone-swiftkey/phraseflow-guide.html','portfolio_url':'https://sonjasahebzad.shinyapps.io/data-science-portfolio/','portfolio_bundle_id':'12610556','slides':5,'anonymous_deck_http_status':200,'anonymous_slide_http_status':200,'anonymous_guide_http_status':200,'app_interactive_browser_checks_passed':True,'portfolio_card_and_links_verified':True,'pages_build_conclusion':'success','pages_build_url':'https://github.com/Sonja242/data-science-capstone-swiftkey/actions/runs/36281091544','course_submission':'Not submitted; awaiting author review','peer_review_request':'Draft only; not sent','reserved_english_test_opened':False}
(pub/'publication.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
# Preserve the previous publication record before moving the product-level pointer.
previous=root/'final_project/results/publication.json'
if not (pub/'previous-publication.json').exists():shutil.copy2(previous,pub/'previous-publication.json')
previous.write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
p=root/'final_project/Submission_Details.md'
s=p.read_text(encoding='utf-8').replace('publication verification in progress. Not submitted to Coursera. No peer-review message sent.','published and verified on 27 September 2026. Ready for author review. Not submitted to Coursera. No peer-review message sent.')
s=s.replace('Evidence of live checks is retained in `publication-20260927`.','All five previously documented English news/Twitter prefixes received a single-word prediction in the live app. RPubs and the embedded five-slide deck both returned HTTP 200 without signing in. Evidence is retained in `publication-20260927/publication.json` and `live-browser-checks.json`.')
p.write_text(s,encoding='utf-8')
notes=pub/'README.md'
s=notes.read_text(encoding='utf-8')+'\nPublication is complete. `publication.json` records the verified public URLs, bundle identifiers and Pages build. `live-browser-checks.json` records five functional English checks; these do not estimate new accuracy. The initial broad token scan produced matches inside embedded image data; a case-sensitive scan excluding embedded media found no credential-shaped strings.\n'
notes.write_text(s,encoding='utf-8')
shutil.copy2(Path(__file__),pub/'finalize_publication.py')
print('Publication evidence and author review pack finalized. Coursera and peer messaging remain untouched.')
