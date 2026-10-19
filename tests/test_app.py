'''The Flask app: dashboard, HR review, taking a quiz, upload errors.

Most tests work on hand-built quiz records, so only the GenerationTest
cases pay for real question generation.  Records go to a temporary
folder.
'''
import copy
import hashlib
import io
import json
import logging
import os
import re
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
logging.getLogger('pypdf').setLevel(logging.ERROR)

from markupsafe import escape  # noqa: E402
from pypdf import PdfWriter  # noqa: E402

import app as app_module  # noqa: E402

app = app_module.app
_tmp = None

REFERENCE = ('Employees must report suspected violations of the Code to the '
             'Law Department.')
PASSAGE = ('Gifts above the limit need approval from the Law Department '
           'before they are accepted.')
QUESTIONS = [
    {'qid': 'fr000001', 'type': 'Free-Response',
     'question': 'What does the Code of Conduct require of employees?',
     'answer': REFERENCE, 'strictness': 'standard', 'flags': []},
    {'qid': 'mc000002', 'type': 'Multiple-Choice',
     'question': 'Gifts above the limit need approval from the __________.',
     'answer': 'Law Department', 'source_sentence': PASSAGE,
     'options': ['Internal Audit', 'Law Department', 'Payroll Office',
                 'Board of Directors'],
     'flags': ['Reads awkwardly: the source sentence joins two statements '
               'with a comma.']},
]
SPARES = [
    {'qid': 'sp000003', 'type': 'Multiple-Choice',
     'question': 'Report concerns to the __________ or your manager.',
     'answer': 'Ethics Line', 'source_sentence': 'Report concerns to the '
     'Ethics Line or your manager.',
     'options': ['Ethics Line', 'Help Desk', 'Front Desk', 'Payroll Office'],
     'flags': []},
]


def setUpModule():
    global _tmp
    _tmp = tempfile.TemporaryDirectory()
    app.config['QUIZ_STORE_FOLDER'] = os.path.join(_tmp.name, 'quiz_store')
    # Exercise the error handlers, not Flask's debug re-raise.
    app.config['TESTING'] = False


def tearDownModule():
    _tmp.cleanup()


def make_quiz(status='draft', result=None, pass_mark=80, title='Code test'):
    quiz = {
        'id': os.urandom(16).hex(),
        'created': '2026-10-19 09:00:00',
        'source': 'code.pdf',
        'title': title,
        'doc_title': 'the Code of Conduct',
        'generator': 'free_based',
        'pass_mark': pass_mark,
        'status': status,
        'questions': copy.deepcopy(QUESTIONS),
        'spares': copy.deepcopy(SPARES),
        'result': result,
    }
    app_module._save_quiz(quiz)
    return quiz['id']


def stored(quiz_id):
    path = os.path.join(app.config['QUIZ_STORE_FOLDER'], f'{quiz_id}.json')
    with open(path, encoding='utf-8') as quiz_file:
        return json.load(quiz_file)


def page(response):
    return response.get_data(as_text=True)


def score_of(html):
    match = re.search(r'class="score-value">\s*(\d+) of (\d+)\s*<', html)
    return (int(match.group(1)), int(match.group(2))) if match else None


def mcq_form(item, **overrides):
    form = {'question': item['question'],
            'correct': str(item['options'].index(item['answer']))}
    form.update({f'option-{i}': option
                 for i, option in enumerate(item['options'])})
    form.update(overrides)
    return form


def upload(client, filename, data, method='blank_based', **settings):
    form = {'generation_method': method, **settings}
    if filename is not None:
        form['file'] = (io.BytesIO(data), filename)
    return client.post('/quizzes', data=form,
                       content_type='multipart/form-data')


def pdf_bytes(password=None):
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    if password:
        writer.encrypt(password)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


class DashboardTest(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()
        self.draft = make_quiz('draft', title='Draft quiz')
        self.published = make_quiz('published', title='Published quiz')
        self.completed = make_quiz(
            'published', title='Completed quiz',
            result={'total': 2, 'score': 1, 'passed': False, 'review': []})

    def test_each_quiz_links_to_its_next_step(self):
        html = page(self.client.get('/'))
        self.assertIn(f'/quizzes/{self.draft}/review', html)
        self.assertIn(f'/quizzes/{self.published}/take', html)
        self.assertIn(f'/quizzes/{self.completed}/result', html)
        self.assertIn('Draft · 1 to check', html)
        self.assertIn('1 / 2', html)

    def test_filters(self):
        drafts = page(self.client.get('/?show=draft'))
        self.assertIn('Draft quiz', drafts)
        self.assertNotIn('Published quiz', drafts)
        published = page(self.client.get('/?show=published'))
        self.assertIn('Published quiz', published)
        self.assertIn('Completed quiz', published)
        self.assertNotIn('Draft quiz', published)


class ReviewTest(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()
        self.quiz_id = make_quiz('draft')
        self.base = f'/quizzes/{self.quiz_id}'

    def test_review_shows_answer_key_flags_and_passages(self):
        html = page(self.client.get(f'{self.base}/review'))
        self.assertIn(str(escape(REFERENCE)), html)
        self.assertIn(str(escape(PASSAGE)), html)
        self.assertIn('From the Code of Conduct', html)
        self.assertIn('Reads awkwardly', html)
        self.assertIn('Spare questions · 1', html)

    def test_saving_edits_updates_the_question_and_clears_flags(self):
        question = QUESTIONS[1]
        form = mcq_form(question, question='Who approves gifts? __________',
                        correct='0', **{'option-0': 'Legal team'})
        response = self.client.post(
            f'{self.base}/questions/{question["qid"]}', data=form)
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers['Location'].endswith(
            f'#q-{question["qid"]}'))

        saved = stored(self.quiz_id)['questions'][1]
        self.assertEqual(saved['question'], 'Who approves gifts? __________')
        self.assertEqual(saved['answer'], 'Legal team')
        self.assertEqual(saved['options'][0], 'Legal team')
        self.assertEqual(saved['flags'], [])
        self.assertIn('now counts as checked',
                      page(self.client.get(f'{self.base}/review')))

    def test_invalid_edits_are_refused(self):
        question = QUESTIONS[1]
        cases = {
            'empty question': mcq_form(question, question='  '),
            'duplicate options': mcq_form(
                question, **{'option-0': 'the Law Department'}),
            'no correct answer': mcq_form(question, correct=''),
            'one option left': {'question': 'Q?', 'correct': '1',
                                'option-1': 'Law Department'},
        }
        for label, form in cases.items():
            with self.subTest(label):
                self.client.post(f'{self.base}/questions/{question["qid"]}',
                                 data=form)
                self.assertEqual(stored(self.quiz_id)['questions'][1],
                                 QUESTIONS[1])
                html = page(self.client.get(f'{self.base}/review'))
                self.assertIn('notice-error', html)

    def test_free_response_marking_is_saved(self):
        question = QUESTIONS[0]
        self.client.post(f'{self.base}/questions/{question["qid"]}', data={
            'question': question['question'], 'answer': REFERENCE,
            'strictness': 'strict'})
        self.assertEqual(stored(self.quiz_id)['questions'][0]['strictness'],
                         'strict')

    def test_remove_moves_the_question_to_the_spares(self):
        self.client.post(f'{self.base}/questions/mc000002/remove')
        quiz = stored(self.quiz_id)
        self.assertEqual([q['qid'] for q in quiz['questions']], ['fr000001'])
        self.assertEqual(quiz['spares'][0]['qid'], 'mc000002')

        self.client.post(f'{self.base}/questions/fr000001/remove')
        self.assertEqual(len(stored(self.quiz_id)['questions']), 1,
                         'the last question cannot be removed')

    def test_swap_and_add_spares(self):
        self.client.post(f'{self.base}/questions/mc000002/swap',
                         data={'spare': 'sp000003'})
        quiz = stored(self.quiz_id)
        self.assertEqual(quiz['questions'][1]['qid'], 'sp000003')
        self.assertEqual(quiz['spares'][0]['qid'], 'mc000002')

        self.client.post(f'{self.base}/spares/mc000002/add')
        quiz = stored(self.quiz_id)
        self.assertEqual([q['qid'] for q in quiz['questions']],
                         ['fr000001', 'sp000003', 'mc000002'])
        self.assertEqual(quiz['spares'], [])

    def test_publishing_locks_the_questions(self):
        response = self.client.post(f'{self.base}/publish')
        self.assertTrue(response.headers['Location'].endswith('/take'))
        self.assertEqual(stored(self.quiz_id)['status'], 'published')

        response = self.client.get(f'{self.base}/review')
        self.assertTrue(response.headers['Location'].endswith('/take'))
        self.client.post(f'{self.base}/questions/mc000002/remove')
        self.assertEqual(len(stored(self.quiz_id)['questions']), 2)


class TakeQuizTest(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()
        self.quiz_id = make_quiz('published')
        self.base = f'/quizzes/{self.quiz_id}'
        self.correct = {q['qid']: q['answer'] for q in QUESTIONS}

    def test_quiz_page_never_shows_the_answer_key(self):
        html = page(self.client.get(f'{self.base}/take'))
        self.assertIn('name="mc000002"', html)
        self.assertNotIn(str(escape(REFERENCE)), html)
        self.assertNotIn(str(escape(PASSAGE)), html)
        self.assertNotIn('Correct answer', html)

    def test_drafts_cannot_be_taken(self):
        draft = make_quiz('draft')
        response = self.client.get(f'/quizzes/{draft}/take')
        self.assertTrue(response.headers['Location'].endswith('/review'))
        self.client.post(f'/quizzes/{draft}/submit', data=self.correct)
        self.assertIsNone(stored(draft)['result'])

    def test_first_submission_is_graded_and_final(self):
        response = self.client.post(f'{self.base}/submit',
                                    data={**self.correct,
                                          'mc000002': 'Payroll Office'})
        self.assertTrue(response.headers['Location'].endswith('/result'))
        html = page(self.client.get(f'{self.base}/result'))
        self.assertEqual(score_of(html), (1, 2))
        self.assertIn('Not passed this time', html)
        self.assertIn(str(escape(PASSAGE)), html)

        self.client.post(f'{self.base}/submit', data=self.correct)
        self.assertEqual(stored(self.quiz_id)['result']['score'], 1)
        self.assertTrue(self.client.get(f'{self.base}/take')
                        .headers['Location'].endswith('/result'))

    def test_pass_mark_decides_the_verdict(self):
        self.client.post(f'{self.base}/submit', data=self.correct)
        html = page(self.client.get(f'{self.base}/result'))
        self.assertEqual(score_of(html), (2, 2))
        self.assertIn('You passed', html)
        self.assertIn('Close match to the Code of Conduct', html)

    def test_result_waits_for_a_submission(self):
        response = self.client.get(f'{self.base}/result')
        self.assertTrue(response.headers['Location'].endswith('/take'))


class GenerationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open('pdf/Code-of-Conduct-English_KFC.pdf', 'rb') as pdf:
            cls.kfc = pdf.read()

    def created_quiz(self, response):
        self.assertEqual(response.status_code, 302, page(response))
        match = re.search(r'/quizzes/([0-9a-f]{32})/review',
                          response.headers['Location'])
        self.assertIsNotNone(match)
        return stored(match.group(1))

    def test_upload_creates_a_draft_with_spares(self):
        quiz = self.created_quiz(upload(
            app.test_client(), 'Code-of-Conduct-English_KFC.pdf', self.kfc,
            method='free_based', question_count='5', pass_mark='70',
            title='KFC policy check'))
        self.assertEqual(quiz['status'], 'draft')
        self.assertEqual(quiz['title'], 'KFC policy check')
        self.assertEqual(quiz['pass_mark'], 70)
        self.assertEqual(quiz['doc_title'], 'the Code of Conduct')
        self.assertEqual(len(quiz['questions']), 5)
        self.assertEqual(quiz['questions'][0]['type'], 'Free-Response')
        self.assertTrue(quiz['spares'])
        for question in quiz['questions'] + quiz['spares']:
            self.assertRegex(question['qid'], r'^[0-9a-f]{8}$')
            self.assertIsInstance(question['flags'], list)

    def test_question_count_setting(self):
        quiz = self.created_quiz(upload(
            app.test_client(), 'Code-of-Conduct-English_KFC.pdf', self.kfc,
            question_count='8'))
        self.assertEqual(len(quiz['questions']), 8)

    def test_utf16_text_file_builds_a_quiz(self):
        # Saved by Notepad as "Unicode" or by PowerShell's ">"; read as
        # UTF-8 this used to fail with "No questions could be generated".
        text = app_module.pdf2text('pdf/Code-of-Conduct-English_KFC.pdf',
                                   'pdf')
        quiz = self.created_quiz(upload(
            app.test_client(), 'policy.txt', text.encode('utf-16')))
        self.assertEqual(len(quiz['questions']), 5)
        self.assertEqual(quiz['title'], 'Policy')

    def test_uploads_never_touch_the_sample_folder(self):
        # Uploads were once saved into pdf/ under their own name, which
        # overwrote samples - and failed while OneDrive held the file.
        def snapshot():
            hashes = {}
            for name in os.listdir('pdf'):
                with open(os.path.join('pdf', name), 'rb') as sample:
                    hashes[name] = hashlib.sha256(sample.read()).hexdigest()
            return hashes

        before = snapshot()
        client = app.test_client()
        for _ in range(2):      # same name twice in a row
            upload(client, 'netflix-culture.pdf', pdf_bytes())
        self.assertEqual(snapshot(), before)


class UploadErrorTest(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def assertUploadFails(self, filename, data, message, status=400):
        before = set(os.listdir(app.config['QUIZ_STORE_FOLDER'])) \
            if os.path.isdir(app.config['QUIZ_STORE_FOLDER']) else set()
        response = upload(self.client, filename, data, method='free_based')
        html = page(response)
        self.assertEqual(response.status_code, status, html)
        self.assertIn(str(escape(message)), html)
        # The form comes back with the chosen settings, and no record.
        self.assertIn('value="free_based" checked', html)
        after = set(os.listdir(app.config['QUIZ_STORE_FOLDER'])) \
            if os.path.isdir(app.config['QUIZ_STORE_FOLDER']) else set()
        self.assertEqual(after, before)

    def test_no_file(self):
        self.assertUploadFails(None, b'', 'No file was uploaded')

    def test_unsupported_types(self):
        for name in ('notes.docx', 'README'):
            with self.subTest(name=name):
                self.assertUploadFails(name, b'data', 'is not a PDF or TXT')

    def test_files_without_text(self):
        for name, data in (('scanned.pdf', pdf_bytes()),
                           ('empty.txt', b'  \n ')):
            with self.subTest(name=name):
                self.assertUploadFails(name, data, 'No text could be read')

    def test_password_protected_pdf(self):
        self.assertUploadFails('locked.pdf', pdf_bytes(password='secret'),
                               'This PDF is password-protected')

    def test_damaged_pdf(self):
        self.assertUploadFails('broken.pdf', b'this is not a pdf',
                               'could not be read as a PDF')

    def test_document_without_usable_sentences(self):
        with open('pdf/netflix-culture.pdf', 'rb') as pdf:
            self.assertUploadFails('netflix-culture.pdf', pdf.read(),
                                   'No questions could be generated')


class RoutingTest(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_old_urls_redirect(self):
        self.assertTrue(self.client.get('/quiz').headers['Location']
                        .endswith('/quizzes/new'))
        self.assertTrue(self.client.get('/quiz/archive').headers['Location']
                        .endswith('/'))
        quiz_id = make_quiz('published')
        self.assertTrue(self.client.get(f'/quiz/review/{quiz_id}')
                        .headers['Location'].endswith('/take'))

    def test_unknown_or_malformed_ids_are_404(self):
        for quiz_id in ('0' * 32, 'A' * 32, 'a' * 33, '..%2Fapp'):
            for path in ('review', 'take', 'result'):
                with self.subTest(quiz_id=quiz_id, path=path):
                    response = self.client.get(f'/quizzes/{quiz_id}/{path}')
                    self.assertEqual(response.status_code, 404)
            response = self.client.post(f'/quizzes/{quiz_id}/submit')
            self.assertEqual(response.status_code, 404)

    def test_legacy_records_still_load(self):
        # Before the review step, questions were a numbered dict and the
        # result kept "review_data"; such quizzes count as published.
        legacy = {
            'created': '2026-10-18 10:00:00', 'source': 'old.pdf',
            'generator': 'blank_based',
            'questions': {'1': {'question': 'Old __________ stem.',
                                'answer': 'Old answer',
                                'options': ['Old answer', 'B', 'C', 'D'],
                                'type': 'Multiple-Choice'}},
            'result': None,
        }
        quiz_id = os.urandom(16).hex()
        os.makedirs(app.config['QUIZ_STORE_FOLDER'], exist_ok=True)
        with open(os.path.join(app.config['QUIZ_STORE_FOLDER'],
                               f'{quiz_id}.json'), 'w') as record:
            json.dump(legacy, record)

        html = page(self.client.get(f'/quizzes/{quiz_id}/take'))
        self.assertIn('Old __________ stem.', html)
        self.client.post(f'/quizzes/{quiz_id}/submit', data={'1': 'Old answer'})
        self.assertEqual(score_of(page(self.client.get(
            f'/quizzes/{quiz_id}/result'))), (1, 1))


if __name__ == '__main__':
    unittest.main()
