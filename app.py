import json
import os
import re
import secrets
import tempfile
from datetime import datetime

from flask import (
    Flask, abort, flash, redirect, render_template, request, url_for
)

from candidate_selection import normalise_phrase
from hybrid_assessment_system import document_title
from question_processing import evaluate_free_response, review_flags
from workers import UnreadableDocumentError, generate_quiz, pdf2text

# Quiz records, answer keys included.  Never sent to the browser:
# Flask's session cookie is signed but not encrypted, so anything stored
# in it is readable by whoever holds it.
QUIZ_STORE_FOLDER = './quiz_store/'

ALLOWED_EXTENSIONS = ('pdf', 'txt')

GENERATORS = {
    'blank_based': 'Fill in the blank',
    'free_based': 'Mixed',
}
QUESTION_COUNTS = (5, 8, 10)
PASS_MARKS = (60, 70, 80, 100)
DEFAULT_QUESTION_COUNT = 5
DEFAULT_PASS_MARK = 80

# How close in meaning a written answer must be to its reference: the
# cosine similarity of their sentence embeddings.
STRICTNESS = {
    'lenient': ('Lenient: the main idea is enough', 0.50),
    'standard': ('Standard: close in meaning', 0.60),
    'strict': ('Strict: nearly the same wording', 0.70),
}
DEFAULT_STRICTNESS = 'standard'

app = Flask(__name__)

app.config['QUIZ_STORE_FOLDER'] = QUIZ_STORE_FOLDER
# Without QG_SECRET_KEY a fresh key is drawn on every start, which
# drops pending messages on restart but never ships a known key.
app.secret_key = os.environ.get('QG_SECRET_KEY') or secrets.token_hex(32)

_QUIZ_ID = re.compile(r'[0-9a-f]{32}')
_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'


# ---------------------------------------------------------------- #
# Uploads
# ---------------------------------------------------------------- #

def _read_upload(uploaded_file, file_exten):
    """ Extract an upload's text through a private temporary file.

    Uploads used to be saved into pdf/ under their own name, which
    overwrote any sample with the same name and failed outright when
    Windows still held the previous copy open (OneDrive syncing it).
    Nothing reads the saved copy afterwards, so none is kept. """

    handle, temp_path = tempfile.mkstemp(suffix=f'.{file_exten}')
    try:
        with os.fdopen(handle, 'wb') as temp_file:
            uploaded_file.save(temp_file)
        return pdf2text(temp_path, file_exten)
    finally:
        os.remove(temp_path)


def _title_from_filename(filename):
    """ A readable default title: "hcrg_employee_handbook.pdf" becomes
    "Hcrg Employee Handbook".  The reviewer can type a better one. """

    stem = os.path.splitext(filename or '')[0]
    stem = re.sub(r'\.pdf\b', ' ', stem, flags=re.IGNORECASE)
    words = re.sub(r'[_\-]+', ' ', stem).split()
    small = {'a', 'an', 'and', 'of', 'the', 'for', 'in', 'on', 'to', 'or'}
    titled = [
        word.capitalize() if word.islower()
        and (index == 0 or word not in small) else word
        for index, word in enumerate(words)
    ]
    return ' '.join(titled) or 'Untitled quiz'


def _choice(value, allowed, default):
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number in allowed else default


# ---------------------------------------------------------------- #
# Quiz store
# ---------------------------------------------------------------- #

def _now():
    return datetime.now().strftime(_TIME_FORMAT)


def _quiz_path(quiz_id):
    """ Path of a stored quiz, or None for anything but a well-formed
    id, so a crafted URL cannot point outside the store. """

    if not isinstance(quiz_id, str) or not _QUIZ_ID.fullmatch(quiz_id):
        return None
    return os.path.join(app.config['QUIZ_STORE_FOLDER'], f'{quiz_id}.json')


def _save_quiz(quiz):
    """ Write a quiz record (questions, answer key, result) to the store """

    os.makedirs(app.config['QUIZ_STORE_FOLDER'], exist_ok=True)
    record = {key: value for key, value in quiz.items() if key != 'id'}

    with open(_quiz_path(quiz['id']), 'w', encoding='utf-8') as quiz_file:
        json.dump(record, quiz_file, ensure_ascii=False)


def _normalise(record, quiz_id):
    """ Fill in what records written by earlier versions lack: they kept
    questions as a numbered dict and had no review step, so they count
    as published. """

    questions = record.get('questions') or []
    if isinstance(questions, dict):
        questions = [
            dict(question, qid=str(number))
            for number, question in sorted(
                questions.items(),
                key=lambda item: int(item[0]) if str(item[0]).isdigit() else 0)
        ]
    for question in questions:
        question.setdefault('type', 'Multiple-Choice')
    record['questions'] = questions

    record.setdefault('spares', [])
    record.setdefault('status', 'published')
    record.setdefault('title', _title_from_filename(record.get('source')))
    record.setdefault('doc_title', 'the document')
    record.setdefault('pass_mark', None)
    record.setdefault('generator', 'blank_based')

    result = record.get('result')
    if result and 'review' not in result:
        result['review'] = [
            dict(item, qid=str(qid))
            for qid, item in (result.get('review_data') or {}).items()
        ]

    record['id'] = quiz_id
    return record


def _load_quiz(quiz_id):
    """ Return a stored quiz record, or None when it cannot be found """

    quiz_path = _quiz_path(quiz_id)
    if quiz_path is None:
        return None

    try:
        with open(quiz_path, 'r', encoding='utf-8') as quiz_file:
            return _normalise(json.load(quiz_file), quiz_id)
    except (OSError, ValueError):
        return None


def _load_or_404(quiz_id):
    quiz = _load_quiz(quiz_id)
    if quiz is None:
        abort(404)
    return quiz


def _all_quizzes():
    folder = app.config['QUIZ_STORE_FOLDER']
    if not os.path.isdir(folder):
        return []

    quizzes = []
    for filename in os.listdir(folder):
        quiz_id, extension = os.path.splitext(filename)
        quiz = _load_quiz(quiz_id) if extension == '.json' else None
        if quiz:
            if not quiz.get('created'):
                quiz['created'] = datetime.fromtimestamp(
                    os.path.getmtime(_quiz_path(quiz_id))
                ).strftime(_TIME_FORMAT)
            quizzes.append(quiz)
    quizzes.sort(key=lambda quiz: quiz['created'], reverse=True)
    return quizzes


def _state(quiz):
    """ draft -> published -> completed (the first submission is final) """

    if quiz.get('result'):
        return 'completed'
    return 'draft' if quiz.get('status') == 'draft' else 'published'


def _quiz_home(quiz):
    """ Where a quiz's link leads: its review, its quiz page, or its result """

    endpoint = {
        'draft': 'review_quiz',
        'published': 'take_quiz',
        'completed': 'quiz_result',
    }[_state(quiz)]
    return url_for(endpoint, quiz_id=quiz['id'])


def _find(questions, qid):
    return next((index for index, question in enumerate(questions)
                 if question.get('qid') == qid), None)


def _create_draft(questions, spares, settings, source_filename, content):
    for question in questions + spares:
        question['qid'] = secrets.token_hex(4)
        question['flags'] = review_flags(question)
        if question['type'] == 'Free-Response':
            question['strictness'] = DEFAULT_STRICTNESS

    quiz = {
        'id': secrets.token_hex(16),
        'created': _now(),
        'source': source_filename,
        'title': settings['title'] or _title_from_filename(source_filename),
        # "the Handbook", so answers can cite "From the Handbook".
        'doc_title': document_title(content)[0],
        'generator': settings['generation_method'],
        'pass_mark': settings['pass_mark'],
        'status': 'draft',
        'questions': questions,
        'spares': spares,
        'result': None,
    }
    _save_quiz(quiz)
    return quiz


def _grade(quiz, answers):
    review = []
    for question in quiz['questions']:
        given = (answers.get(question['qid']) or '').strip()
        item = {
            'qid': question['qid'],
            'type': question['type'],
            'question': question['question'],
            'user_answer': given,
            'correct_answer': question['answer'],
        }

        if question['type'] == 'Free-Response':
            _label, threshold = STRICTNESS.get(
                question.get('strictness'), STRICTNESS[DEFAULT_STRICTNESS])
            evaluation = evaluate_free_response(
                given, question['answer'], threshold=threshold)
            item.update(
                correct=evaluation['status'] == 'Correct',
                score=evaluation['score'],
                status=evaluation['status'],
                passage=question['answer'],
            )
        else:
            item.update(
                correct=given.casefold()
                == question['answer'].strip().casefold(),
                options=question.get('options', []),
                passage=question.get('source_sentence', ''),
            )
        review.append(item)

    total = len(review)
    score = sum(item['correct'] for item in review)
    pass_mark = quiz.get('pass_mark')
    return {
        'submitted': _now(),
        'total': total,
        'score': score,
        'passed': None if pass_mark is None
        else score * 100 >= pass_mark * total,
        'review': review,
    }


# ---------------------------------------------------------------- #
# Template helpers
# ---------------------------------------------------------------- #

@app.context_processor
def _template_constants():
    return {
        'GENERATORS': GENERATORS,
        'STRICTNESS': STRICTNESS,
        'QUESTION_COUNTS': QUESTION_COUNTS,
        'PASS_MARKS': PASS_MARKS,
        'quiz_state': _state,
        'quiz_home': _quiz_home,
    }


@app.template_filter('nice_date')
def _nice_date(timestamp):
    try:
        moment = datetime.strptime(timestamp, _TIME_FORMAT)
    except (TypeError, ValueError):
        return timestamp or ''
    return f'{moment:%b} {moment.day}, {moment:%Y}'


@app.template_filter('nice_time')
def _nice_time(timestamp):
    try:
        moment = datetime.strptime(timestamp, _TIME_FORMAT)
    except (TypeError, ValueError):
        return timestamp or ''
    return f'{moment:%b} {moment.day}, {moment:%H:%M}'


# ---------------------------------------------------------------- #
# HR: quizzes, new quiz, review
# ---------------------------------------------------------------- #

@app.route('/')
def dashboard():
    """ Every quiz, newest first, filterable by draft / published """

    quizzes = _all_quizzes()
    counts = {
        'all': len(quizzes),
        'draft': sum(_state(quiz) == 'draft' for quiz in quizzes),
    }
    counts['published'] = counts['all'] - counts['draft']

    show = request.args.get('show', 'all')
    if show == 'draft':
        quizzes = [quiz for quiz in quizzes if _state(quiz) == 'draft']
    elif show == 'published':
        quizzes = [quiz for quiz in quizzes if _state(quiz) != 'draft']
    else:
        show = 'all'

    return render_template(
        'dashboard.html', quizzes=quizzes, counts=counts, show=show)


def _new_quiz_settings(form=None):
    form = form or {}
    method = form.get('generation_method')
    return {
        'generation_method': method if method in GENERATORS else 'blank_based',
        'question_count': _choice(form.get('question_count'),
                                  QUESTION_COUNTS, DEFAULT_QUESTION_COUNT),
        'pass_mark': _choice(form.get('pass_mark'),
                             PASS_MARKS, DEFAULT_PASS_MARK),
        'title': (form.get('title') or '').strip()[:120],
    }


@app.route('/quizzes/new')
def new_quiz():
    return render_template('new_quiz.html', settings=_new_quiz_settings())


@app.route('/quizzes', methods=['POST'])
def create_quiz():
    """ Generate a draft from an upload, then send HR to review it """

    settings = _new_quiz_settings(request.form)
    error = None
    status = 400

    uploaded_file = request.files.get('file')
    source_filename = uploaded_file.filename if uploaded_file else ''
    file_exten = os.path.splitext(source_filename)[1].lstrip('.').lower()

    # Each failure gets its own message: the page used to report success
    # whenever extraction did not crash, so a scanned PDF or an
    # unsupported file showed an empty quiz with only a Submit button.
    if not source_filename:
        error = 'No file was uploaded. Please choose a PDF or TXT file.'
    elif file_exten not in ALLOWED_EXTENSIONS:
        error = (f'"{source_filename}" is not a PDF or TXT file. '
                 'Please upload one of those.')
    else:
        try:
            content = _read_upload(uploaded_file, file_exten)

            if not content.strip():
                # Typically a scanned PDF: its pages are images with no
                # text layer for the extractor to read.
                error = ('No text could be read from this file. If it is '
                         'a scanned PDF, run it through OCR first so its '
                         'text can be extracted.')
            else:
                questions, spares = generate_quiz(
                    content, settings['generation_method'],
                    n=settings['question_count'])
                if questions:
                    quiz = _create_draft(questions, spares, settings,
                                         source_filename, content)
                    return redirect(url_for('review_quiz', quiz_id=quiz['id']))
                error = ('No questions could be generated from this '
                         'document. It may be too short, or made up mostly '
                         'of headings, lists or slides rather than full '
                         'sentences.')
        except UnreadableDocumentError as e:
            error = str(e)
        except Exception:
            # Full traceback for whoever runs the server; the user gets
            # a message, not a stack trace.
            app.logger.exception(
                'Quiz generation failed for %r', source_filename
            )
            error = ('Something went wrong while processing this file. '
                     'The error has been logged on the server.')
            status = 500

    return render_template(
        'new_quiz.html', settings=settings, error=error), status


def _draft_or_redirect(quiz_id):
    """ (quiz, None) for a draft; (quiz, redirect) once it is published.

    The answer key is only shown while a quiz is a draft: after that it
    stays hidden until the quiz has been submitted. """

    quiz = _load_or_404(quiz_id)
    if _state(quiz) != 'draft':
        flash('This quiz has been published, so its questions can no '
              'longer be changed.', 'info')
        return quiz, redirect(_quiz_home(quiz))
    return quiz, None


def _say(message, category, qid=None):
    """ Flash a message.  One about a question is shown inside its card:
    the review page scrolls to that card, out of sight of the page top. """

    flash(message, f'{category} q-{qid}' if qid else category)


def _back_to_review(quiz, qid=None):
    anchor = f'q-{qid}' if qid else None
    return redirect(url_for('review_quiz', quiz_id=quiz['id'], _anchor=anchor))


@app.route('/quizzes/<quiz_id>/review')
def review_quiz(quiz_id):
    quiz, response = _draft_or_redirect(quiz_id)
    return response or render_template('review.html', quiz=quiz)


@app.route('/quizzes/<quiz_id>/questions/<qid>', methods=['POST'])
def save_question(quiz_id, qid):
    """ Save a reviewer's edits; a saved question counts as checked """

    quiz, response = _draft_or_redirect(quiz_id)
    if response:
        return response
    index = _find(quiz['questions'], qid)
    if index is None:
        abort(404)
    question = quiz['questions'][index]
    form = request.form

    text = (form.get('question') or '').strip()
    if not text:
        _say('The question cannot be empty.', 'error', qid)
        return _back_to_review(quiz, qid)

    if question['type'] == 'Free-Response':
        reference = (form.get('answer') or '').strip()
        if not reference:
            _say('The reference answer cannot be empty.', 'error', qid)
            return _back_to_review(quiz, qid)
        strictness = form.get('strictness')
        question.update(
            question=text,
            answer=reference,
            strictness=strictness if strictness in STRICTNESS
            else question.get('strictness', DEFAULT_STRICTNESS),
        )
    else:
        filled = [
            (str(index_), (form.get(f'option-{index_}') or '').strip())
            for index_ in range(len(question.get('options', [])))
        ]
        filled = [(key, option) for key, option in filled if option]
        if len(filled) < 2:
            _say('A multiple-choice question needs at least two options.',
                 'error', qid)
            return _back_to_review(quiz, qid)

        normalised = [normalise_phrase(option) for _key, option in filled]
        if len(set(normalised)) != len(normalised):
            _say('Two options say the same thing. Make every option '
                 'different.', 'error', qid)
            return _back_to_review(quiz, qid)

        correct = next((option for key, option in filled
                        if key == form.get('correct')), None)
        if correct is None:
            _say('Choose which option is the correct answer.', 'error', qid)
            return _back_to_review(quiz, qid)

        question.update(question=text,
                        options=[option for _key, option in filled],
                        answer=correct)

    question['flags'] = []
    _save_quiz(quiz)
    _say('Saved. This question now counts as checked.', 'success', qid)
    return _back_to_review(quiz, qid)


@app.route('/quizzes/<quiz_id>/questions/<qid>/remove', methods=['POST'])
def remove_question(quiz_id, qid):
    """ Move a question to the spares, from where it can be added back """

    quiz, response = _draft_or_redirect(quiz_id)
    if response:
        return response
    index = _find(quiz['questions'], qid)
    if index is None:
        abort(404)
    if len(quiz['questions']) == 1:
        _say('A quiz needs at least one question.', 'error', qid)
        return _back_to_review(quiz, qid)

    quiz['spares'].insert(0, quiz['questions'].pop(index))
    _save_quiz(quiz)
    flash(f'Question {index + 1} removed. It is now first in the spare '
          'questions if you want it back.', 'success')
    return _back_to_review(quiz)


@app.route('/quizzes/<quiz_id>/questions/<qid>/swap', methods=['POST'])
def swap_question(quiz_id, qid):
    """ Put a spare in a question's place; the question becomes a spare """

    quiz, response = _draft_or_redirect(quiz_id)
    if response:
        return response
    index = _find(quiz['questions'], qid)
    spare_index = _find(quiz['spares'], request.form.get('spare'))
    if index is None or spare_index is None:
        abort(404)

    spare = quiz['spares'][spare_index]
    quiz['spares'][spare_index] = quiz['questions'][index]
    quiz['questions'][index] = spare
    _save_quiz(quiz)
    _say('Swapped in from the spares. The question it replaced is now a '
         'spare.', 'success', spare['qid'])
    return _back_to_review(quiz, spare['qid'])


@app.route('/quizzes/<quiz_id>/spares/<qid>/add', methods=['POST'])
def add_spare(quiz_id, qid):
    quiz, response = _draft_or_redirect(quiz_id)
    if response:
        return response
    spare_index = _find(quiz['spares'], qid)
    if spare_index is None:
        abort(404)

    quiz['questions'].append(quiz['spares'].pop(spare_index))
    _save_quiz(quiz)
    _say('Added from the spares.', 'success', qid)
    return _back_to_review(quiz, qid)


@app.route('/quizzes/<quiz_id>/publish', methods=['POST'])
def publish_quiz(quiz_id):
    quiz, response = _draft_or_redirect(quiz_id)
    if response:
        return response

    quiz['status'] = 'published'
    quiz['published'] = _now()
    _save_quiz(quiz)
    flash('Quiz published. Questions can no longer be changed.', 'success')
    return redirect(url_for('take_quiz', quiz_id=quiz_id))


# ---------------------------------------------------------------- #
# Employee: take the quiz, see the result
# ---------------------------------------------------------------- #

@app.route('/quizzes/<quiz_id>/take')
def take_quiz(quiz_id):
    quiz = _load_or_404(quiz_id)
    if _state(quiz) != 'published':
        return redirect(_quiz_home(quiz))
    return render_template('take.html', quiz=quiz)


@app.route('/quizzes/<quiz_id>/submit', methods=['POST'])
def submit_quiz(quiz_id):
    """ Grade the first submission; later ones just show that result.

    The result page reveals the correct answers, so regrading a
    resubmission would hand out full marks. """

    quiz = _load_or_404(quiz_id)
    if _state(quiz) == 'published':
        quiz['result'] = _grade(quiz, request.form)
        _save_quiz(quiz)
    return redirect(_quiz_home(quiz))


@app.route('/quizzes/<quiz_id>/result')
def quiz_result(quiz_id):
    quiz = _load_or_404(quiz_id)
    if _state(quiz) != 'completed':
        return redirect(_quiz_home(quiz))
    return render_template('result.html', quiz=quiz, result=quiz['result'])


# ---------------------------------------------------------------- #
# Earlier URLs and errors
# ---------------------------------------------------------------- #

@app.route('/quiz')
def legacy_quiz():
    return redirect(url_for('new_quiz'))


@app.route('/quiz/archive')
def legacy_archive():
    return redirect(url_for('dashboard'))


@app.route('/quiz/review/<quiz_id>')
def legacy_review(quiz_id):
    return redirect(_quiz_home(_load_or_404(quiz_id)))


@app.errorhandler(404)
def not_found(_error):
    return render_template(
        'message.html',
        heading='Page not found',
        text='This quiz or page does not exist. It may have been removed.'
    ), 404


if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)
