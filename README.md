# Question-Generation-system

A Flask web app that turns a company policy document — a code of conduct,
an employee handbook — into an onboarding quiz. Upload a PDF or TXT file
and it builds five questions from the document's own sentences, with
plausible wrong options, then grades the answers.

Two generation modes:

- **Blank-based**: fill-in-the-blank multiple-choice questions.
- **Free-based**: one written-answer question, graded by meaning rather
  than exact wording, plus multiple-choice questions phrased as real
  questions ("Who must …?", "What is …?").

How each step works — text cleaning, sentence selection, answer ranking,
distractor choice and grading — is described in
[ALGORITHMS.md](ALGORITHMS.md).

## Setup

Requires Python 3.10 or newer (tested with 3.14).

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md
python -m nltk.downloader wordnet
```

The sentence-embedding model (`all-MiniLM-L6-v2`, about 90 MB) downloads
from Hugging Face on first start. To work offline, put a copy in
`models/all-MiniLM-L6-v2/` and it is used instead.

## Running

```bash
python app.py
```

Then open http://127.0.0.1:5000.

Status messages are signed with the `QG_SECRET_KEY` environment variable.
Without it a random key is created on each start, which is fine for local
use; set a fixed key for anything longer-lived:

```powershell
$env:QG_SECRET_KEY = "<any long random string>"   # PowerShell
```

```bash
export QG_SECRET_KEY="<any long random string>"   # bash
```

## Using it

The app has two sides: HR prepares each quiz, then an employee takes it.

1. **Quizzes** (the home page) lists every quiz with its status: draft,
   published or completed.
2. **New quiz**: upload a policy PDF or TXT file, then choose the question
   style, the number of questions (5, 8 or 10) and the pass mark.
3. **Review**: every generated question is shown with the passage it came
   from. HR can:
   - fix the wording or the options, and change which option is correct
   - set how strictly written answers are marked
   - remove a question, or swap it for one of the spare questions
   
   Questions that are likely to need an edit are marked *To check*.
4. **Publish**: the questions are locked, and from then on the answer key
   stays hidden until the quiz has been submitted.
5. **Take the quiz**: one question per screen with a progress bar. The
   first submission is final. The result shows the policy passage behind
   each answer.

Sample documents to try are in `pdf/`. Scanned PDFs (images with no text
layer) and password-protected PDFs can't be read and are rejected with a
message saying so.

There is no sign-in yet: anyone who can open the app can open a draft
and see its answer key. Individual HR accounts and shareable quiz links
are the next phase.

## Tests

```bash
python -m unittest discover tests -v
```

Takes about a minute, most of it loading the language models.

## Project layout

| Path | Role |
|---|---|
| `app.py` | Flask routes: quizzes, upload, review, publish, take, grading |
| `workers.py` | PDF/TXT reading; `generate_quiz()` returns questions plus spares |
| `text_cleaning.py` | Removes PDF layout noise, repairs broken words, picks usable sentences |
| `candidate_selection.py` | Finds and ranks answer phrases |
| `question_extraction.py` | Fill-in-the-blank questions |
| `hybrid_assessment_system.py` | Written-answer and question-form multiple-choice items |
| `incorrect_answer_generation.py` | Wrong options (distractors) |
| `question_processing.py` | Final quality check, review flags, and written-answer grading |
| `nlp_models.py` | Loads the spaCy and sentence-embedding models once |
| `templates/` | Page templates, all extending `base.html` |
| `static/` | The shared stylesheet, plus scripts for the upload form and the one-question-per-screen quiz |
| `tests/` | Unit and end-to-end tests |
| `pdf/` | Sample documents (uploads are read from a temporary file and not kept) |
| `quiz_store/` | One record per quiz: questions, spares, answer key, status and result (created at runtime, not committed) |
