// New quiz form: show the chosen file, accept drag-and-drop, and show
// progress while the (synchronous) generation request runs.
(function () {
  'use strict';

  var form = document.getElementById('new-quiz-form');
  if (!form) { return; }

  var input = document.getElementById('policy-file');
  var dropzone = document.getElementById('dropzone');
  var chosen = document.getElementById('file-chosen');
  var nameLabel = document.getElementById('file-name');
  var sizeLabel = document.getElementById('file-size');
  var clearButton = document.getElementById('file-clear');
  var generateButton = document.getElementById('generate-button');

  // The file input is visually hidden, so the browser's own "please
  // select a file" bubble would point at nothing: validate here instead.
  form.setAttribute('novalidate', '');

  function formatSize(bytes) {
    if (bytes < 1024 * 1024) { return Math.max(1, Math.round(bytes / 1024)) + ' KB'; }
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  }

  function showFile() {
    var file = input.files && input.files[0];
    chosen.hidden = !file;
    dropzone.hidden = !!file;
    if (file) {
      var extension = file.name.split('.').pop().toUpperCase();
      nameLabel.textContent = file.name;
      sizeLabel.textContent = extension + ' · ' + formatSize(file.size);
      removeError();
    }
  }

  function removeError() {
    var existing = document.getElementById('file-error');
    if (existing) { existing.remove(); }
  }

  function showError(message) {
    removeError();
    var error = document.createElement('p');
    error.id = 'file-error';
    error.className = 'notice notice-error';
    error.setAttribute('role', 'alert');
    error.textContent = message;
    dropzone.insertAdjacentElement('afterend', error);
  }

  input.addEventListener('change', showFile);

  clearButton.addEventListener('click', function () {
    input.value = '';
    showFile();
    input.click();
  });

  ['dragenter', 'dragover'].forEach(function (type) {
    dropzone.addEventListener(type, function (event) {
      event.preventDefault();
      dropzone.classList.add('is-dragging');
    });
  });
  ['dragleave', 'drop'].forEach(function (type) {
    dropzone.addEventListener(type, function () {
      dropzone.classList.remove('is-dragging');
    });
  });
  dropzone.addEventListener('drop', function (event) {
    event.preventDefault();
    if (event.dataTransfer && event.dataTransfer.files.length) {
      input.files = event.dataTransfer.files;
      showFile();
    }
  });

  form.addEventListener('submit', function (event) {
    if (!input.files || !input.files.length) {
      event.preventDefault();
      showError('Choose a PDF or TXT file first.');
      return;
    }
    // Generation takes a few seconds and the page would otherwise sit
    // still; also stops a second click from starting a second run.
    document.body.classList.add('is-busy');
    generateButton.disabled = true;
  });

  // Coming back with the browser's Back button restores the page from
  // cache, overlay and all.
  window.addEventListener('pageshow', function () {
    document.body.classList.remove('is-busy');
    generateButton.disabled = false;
    showFile();
  });
})();
