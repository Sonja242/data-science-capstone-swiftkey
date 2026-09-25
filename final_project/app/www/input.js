// Sonja Projects | Keep predictions tied to the current text while typing.
(function () {
  function phraseBox() { return document.getElementById('phrase'); }
  function guardSuggestions() {
    const box = phraseBox();
    const card = document.querySelector('#prediction [data-prediction-phrase]');
    if (!box || !card) return;
    const stale = card.getAttribute('data-prediction-phrase') !== box.value;
    card.classList.toggle('stale-prediction', stale);
    card.querySelectorAll('button').forEach(function (button) { button.disabled = stale; });
  }
  function sendCurrentText() {
    const box = phraseBox();
    if (box && window.Shiny) Shiny.setInputValue('phrase', box.value, {priority: 'event'});
  }
  document.addEventListener('input', function (event) {
    if (event.target.id !== 'phrase') return;
    guardSuggestions();
    sendCurrentText();
  }, true);
  document.addEventListener('click', function (event) {
    if (event.target.closest('#predict')) sendCurrentText();
    if (event.target.closest('#choose1, #choose2, #choose3')) {
      const card = document.querySelector('#prediction [data-prediction-phrase]');
      if (!card || !phraseBox() || card.getAttribute('data-prediction-phrase') !== phraseBox().value) {
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
