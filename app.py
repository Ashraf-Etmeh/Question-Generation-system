import os
from datetime import datetime
from flask import (
    Flask, render_template, redirect, url_for, request, session,
    send_from_directory, abort
)
from werkzeug.utils import secure_filename
from workers import pdf2text, txt2questions, txt2questions_free
from question_processing import evaluate_free_response

UPLOAD_FOLDER = './pdf/'
GENERATED_QUIZZES_FOLDER = './generated_quizzes/'

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GENERATED_QUIZZES_FOLDER'] = GENERATED_QUIZZES_FOLDER
app.secret_key = '20032026FAG'


def _save_quiz_snapshot(html, source_filename, generation_method):
    """ Persist a rendered quiz page so past test runs stay reviewable
    from disk instead of only living in the (overwritten) session. """

    os.makedirs(app.config['GENERATED_QUIZZES_FOLDER'], exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    source_stem = 'quiz'
    if source_filename:
        source_stem = secure_filename(
            os.path.splitext(source_filename)[0]
        ) or 'quiz'

    filename = f'{timestamp}_{generation_method}_{source_stem}.html'
    snapshot_path = os.path.join(
        app.config['GENERATED_QUIZZES_FOLDER'], filename
    )

    with open(snapshot_path, 'w', encoding='utf-8') as snapshot_file:
        snapshot_file.write(html)

    return filename


@ app.route('/')
def index():
    """ The landing page for the app """
    return render_template('index.html')


@ app.route('/quiz', methods=['GET', 'POST'])
def quiz():
    """ Handle upload and conversion of file + other stuff """

    UPLOAD_STATUS = False
    questions = dict()
    source_filename = None
    # choose question generator
    generation_method = request.form.get('generation_method', 'blank_based')

    # Make directory to store uploaded files, if not exists
    if not os.path.isdir('./pdf'):
        os.mkdir('./pdf')

    if request.method == 'POST':
        try:
            # Retrieve file from request
            uploaded_file = request.files['file']
            source_filename = uploaded_file.filename
            file_path = os.path.join(
                app.config['UPLOAD_FOLDER'],
                secure_filename(
                    uploaded_file.filename))
            file_exten = uploaded_file.filename.rsplit('.', 1)[1].lower()

            uploaded_file.save(file_path)
            # Get contents of file
            uploaded_content = pdf2text(file_path, file_exten)

            if generation_method == 'blank_based':
                questions = txt2questions(uploaded_content)
                for question_id, question_data in questions.items():
                    question_data['type'] = 'Multiple-Choice'

            elif generation_method == 'free_based':
                questions = txt2questions_free(uploaded_content)
                # questions = process_new_questions(questions)

            else:
                raise ValueError(
                    f'Unknown generator: {generation_method}'
                )

            # Store questions for /result
            session['questions'] = questions

            session['generator'] = generation_method

            # File upload + convert success
            if uploaded_content is not None:
                UPLOAD_STATUS = True
        except Exception as e:
            print(e)

    rendered_quiz = render_template(
        'quiz.html',
        uploaded=UPLOAD_STATUS,
        questions=questions,
        size=len(questions))

    # Keep a standalone copy of every generated quiz so the output of
    # each test run stays reviewable later from /quiz/archive.
    if UPLOAD_STATUS and questions:
        _save_quiz_snapshot(rendered_quiz, source_filename, generation_method)

    return rendered_quiz


@app.route('/quiz/archive')
def quiz_archive():
    """ List every quiz snapshot saved from past test runs """

    os.makedirs(app.config['GENERATED_QUIZZES_FOLDER'], exist_ok=True)

    filenames = sorted(
        (f for f in os.listdir(app.config['GENERATED_QUIZZES_FOLDER'])
         if f.endswith('.html')),
        reverse=True
    )

    snapshots = []
    for filename in filenames:
        stem = filename[:-len('.html')]
        timestamp, _, rest = stem.partition('_')
        method, _, source = rest.partition('_')

        try:
            display_time = datetime.strptime(
                timestamp, '%Y%m%d-%H%M%S'
            ).strftime('%Y-%m-%d %H:%M:%S')
        except ValueError:
            display_time = timestamp

        snapshots.append({
            'filename': filename,
            'time': display_time,
            'method': method or 'unknown',
            'source': source or 'unknown',
        })

    return render_template('quiz_archive.html', snapshots=snapshots)


@app.route('/quiz/archive/<path:filename>')
def quiz_archive_file(filename):
    """ Serve a single saved quiz snapshot as a standalone page """

    safe_name = secure_filename(filename)
    if safe_name != filename:
        abort(404)

    return send_from_directory(
        app.config['GENERATED_QUIZZES_FOLDER'],
        safe_name
    )


@app.route('/result', methods=['POST'])
def result():

    questions = session.get('questions', {})

    if not questions:
        return "No active quiz.", 400

    correct_q = 0
    total = len(questions)

    review_data = {}

    for question_id, question_data in questions.items():

        q_type = question_data.get(
            'type',
            'Multiple-Choice'
        )

        correct_answer = question_data.get(
            'answer',
            ''
        )

        user_answer = request.form.get(
            str(question_id),
            ''
        )

        if q_type == 'Multiple-Choice':

            is_correct = (
                user_answer.strip().casefold()
                == correct_answer.strip().casefold()
            )

            if is_correct:
                correct_q += 1

            review_data[question_id] = {

                'question': question_data.get(
                    'question',
                    ''
                ),

                'type': 'Multiple-Choice',

                'options': question_data.get(
                    'options',
                    []
                ),

                'user_answer': user_answer,

                'correct_answer': correct_answer,

                'correct': is_correct
            }


        elif q_type == 'Free-Response':

            evaluation = evaluate_free_response(
                user_answer,
                correct_answer
            )

            is_correct = (
                evaluation['status']
                == 'Correct'
            )

            if is_correct:
                correct_q += 1

            review_data[question_id] = {

                'question': question_data.get(
                    'question',
                    ''
                ),

                'type': 'Free-Response',

                'user_answer': user_answer,

                'correct_answer': correct_answer,

                'score': evaluation['score'],

                'status': evaluation['status'],

                'correct': is_correct
            }

    return render_template(
        'result.html',
        total=total,
        score=correct_q,
        review_data=review_data
    )

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)
