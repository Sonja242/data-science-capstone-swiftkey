// Sonja Projects | Keep predictions tied to the current text while typing.
(function () {
  let composing = false;
  function phraseBox() { return document.getElementById('phrase'); }
  function guardSuggestions() {
    const box = phraseBox();
    const card = document.querySelector('#prediction [data-prediction-phrase]');
    if (!box || !card) return;
    const language = document.getElementById('language');
    const stale = composing || card.getAttribute('data-prediction-phrase') !== box.value || !language || card.getAttribute('data-prediction-language') !== language.value;
    card.classList.toggle('stale-prediction', stale);
    card.querySelectorAll('button').forEach(function (button) { button.disabled = stale; });
  }
  function sendCurrentText() {
    const box = phraseBox();
    if (box && window.Shiny && !composing) {
      Shiny.setInputValue('phrase_context', {text:box.value, language:document.getElementById('language').value}, {priority:'event'});
      Shiny.setInputValue('phrase', box.value, {priority: 'event'});
    }
  }
  $(document).on('shiny:connected', function () {
    Shiny.addCustomMessageHandler('phraseflow-set-text', function (message) {
      const box = phraseBox(), language = document.getElementById('language');
      if (!box || !language || language.value !== message.language) return;
      box.value = message.text;
      box.dispatchEvent(new Event('input', {bubbles:true}));
    });
  });
  document.addEventListener('compositionstart', function (event) {
    if (event.target.id !== 'phrase') return;
    composing = true;
    if (window.Shiny) Shiny.setInputValue('composing', true, {priority:'event'});
    guardSuggestions();
  }, true);
  document.addEventListener('compositionend', function (event) {
    if (event.target.id !== 'phrase') return;
    composing = false;
    if (window.Shiny) Shiny.setInputValue('composing', false, {priority:'event'});
    guardSuggestions(); sendCurrentText();
  }, true);
  document.addEventListener('input', function (event) {
    if (event.target.id !== 'phrase') return;
    guardSuggestions();
    sendCurrentText();
  }, true);
  document.addEventListener('change', function(event) {
    if (event.target.id === 'phrase') { guardSuggestions(); sendCurrentText(); return; }
    if (event.target.id !== 'language') return;
    composing = false;
    if (window.Shiny) Shiny.setInputValue('composing', false, {priority:'event'});
    if (phraseBox()) phraseBox().value = '';
    guardSuggestions(); sendCurrentText();
  }, true);
  document.addEventListener('click', function (event) {
    if (event.target.closest('#predict')) sendCurrentText();
    if (event.target.closest('button[id^=choose],button[id^=complete]')) {
      const card = document.querySelector('#prediction [data-prediction-phrase]');
      if (!card || !phraseBox() || card.getAttribute('data-prediction-phrase') !== phraseBox().value || card.getAttribute('data-prediction-language') !== document.getElementById('language').value) {
        event.preventDefault();
        event.stopImmediatePropagation();
        guardSuggestions();
      }
    }
  }, true);
  document.addEventListener('keydown', function (event) {
    if (event.target.id === 'phrase' && (event.ctrlKey || event.metaKey) && event.key === 'Enter') {
      event.preventDefault();
      document.getElementById('predict').click();
    }
  });
  document.addEventListener('DOMContentLoaded', function () {
    const output = document.getElementById('prediction');
    if (output) new MutationObserver(guardSuggestions).observe(output, {childList: true, subtree: true});
  });
}());
