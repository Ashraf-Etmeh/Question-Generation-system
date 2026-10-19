// Show a quiz one question at a time.  Without JavaScript every
// question stays on one page with a single Submit button.
(function () {
  'use strict';

  var form = document.getElementById('quiz-form');
  if (!form) { return; }

  var steps = Array.prototype.slice.call(form.querySelectorAll('.step'));
  var progress = document.getElementById('quiz-progress');
  var progressLabel = document.getElementById('quiz-progress-label');
  var progressBar = progress.querySelector('[role="progressbar"]');
  var progressFill = progressBar.querySelector('span');
  var back = document.getElementById('quiz-back');
  var next = document.getElementById('quiz-next');
  var submit = document.getElementById('quiz-submit');
  var error = document.getElementById('quiz-error');
  var current = 0;

  if (steps.length < 2) { return; }

  form.classList.add('js-stepper');
  // Hidden steps cannot show the browser's "required" bubble, so
  // answers are checked here instead.
  form.setAttribute('novalidate', '');
  progress.hidden = false;

  function answered(step) {
    var text = step.querySelector('textarea');
    if (text) { return text.value.trim().length > 0; }
    return !!step.querySelector('input[type="radio"]:checked');
  }

  function show(index, moveFocus) {
    current = Math.max(0, Math.min(index, steps.length - 1));
    steps.forEach(function (step, position) {
      step.classList.toggle('is-current', position === current);
    });
    var last = current === steps.length - 1;
    back.hidden = false;
    back.disabled = current === 0;
    next.hidden = last;
    submit.hidden = !last;
    error.hidden = true;

    progressLabel.textContent = 'Question ' + (current + 1) + ' of ' + steps.length;
    progressBar.setAttribute('aria-valuenow', String(current + 1));
    progressFill.style.width = ((current + 1) / steps.length * 100) + '%';

    if (moveFocus) {
      var heading = steps[current].querySelector('.stem');
      heading.setAttribute('tabindex', '-1');
      heading.focus();
    }
  }

  back.addEventListener('click', function () { show(current - 1, true); });
  next.addEventListener('click', function () { show(current + 1, true); });

  form.addEventListener('submit', function (event) {
    var missing = steps.filter(function (step) { return !answered(step); });
    if (missing.length) {
      event.preventDefault();
      show(steps.indexOf(missing[0]), true);
      error.textContent = missing.length === 1
        ? 'This question still needs an answer.'
        : 'This question still needs an answer (' + missing.length + ' unanswered in total).';
      error.hidden = false;
      return;
    }
    submit.disabled = true;     // one submission only
  });

  show(0, false);
})();
