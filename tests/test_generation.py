'''Question generation: sentence filter, stems, options, sample PDFs.

Loads the spaCy and SBERT models once (about 15 s), then generates
quizzes from the sample PDFs in pdf/.
'''
import logging
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
logging.getLogger('pypdf').setLevel(logging.ERROR)

from candidate_selection import Candidate, extract_candidates  # noqa: E402
from hybrid_assessment_system import (  # noqa: E402
    HybridAssessmentSystem, document_title,
)
from incorrect_answer_generation import IncorrectAnswerGenerator  # noqa: E402
from nlp_models import nlp  # noqa: E402
from question_processing import evaluate_free_response  # noqa: E402
from text_cleaning import clean_document, is_quality_sentence  # noqa: E402
from workers import pdf2text, txt2questions, txt2questions_free  # noqa: E402


def first_sentence(text):
    return next(nlp(text).sents)


class SentenceFilterTest(unittest.TestCase):

    def assertAccepted(self, text):
        self.assertTrue(is_quality_sentence(first_sentence(text)), text)

    def assertRejected(self, text):
        self.assertFalse(is_quality_sentence(first_sentence(text)), text)

    def test_accepts_a_plain_obligation(self):
        self.assertAccepted('Employees must report any suspected violation '
                            'of the Code to the Law Department.')

    def test_accepts_an_imperative(self):
        # Its only finite verb sits in a relative clause, like the
        # fragments below - the verb root is what makes it a sentence.
        self.assertAccepted('Avoid contacts with competitors that may '
                            'create the appearance of improper arrangements, '
                            'side deals, or informal agreements.')

    def test_rejects_a_relative_clause_fragment(self):
        self.assertRejected('Members of internal groups that specialize in '
                            'handling reports, including Human Resources '
                            'and Legal.')

    def test_rejects_a_gerund_list_item(self):
        self.assertRejected('Failing to monitor the conduct of vendors for '
                            'whom you have oversight responsibility.')

    def test_rejects_an_unresolved_demonstrative(self):
        self.assertRejected('Keep this in mind and exercise care when you '
                            'use Company electronic devices.')

    def test_accepts_a_modified_demonstrative(self):
        self.assertAccepted('We expect those with whom we do business to '
                            'respect human rights in every country.')

    def test_rejects_a_cross_reference(self):
        self.assertRejected('See the guidance given in the section titled '
                            'Anti-Corruption for more specific direction.')


class DocumentTitleTest(unittest.TestCase):

    def test_names_the_title_the_document_uses_most(self):
        text = 'This Handbook explains our rules. Keep the handbook nearby.'
        self.assertEqual(document_title(text), ('the Handbook', 'does'))

    def test_plural_title_takes_plural_verb(self):
        text = ('The Standards of Business Conduct apply to all of us. Read '
                'the Standards of Business Conduct carefully.')
        self.assertEqual(document_title(text),
                         ('the Standards of Business Conduct', 'do'))

    def test_falls_back_when_no_title_is_mentioned_enough(self):
        text = 'See the Code of Conduct. Our culture values freedom.'
        self.assertEqual(document_title(text), ('the document', 'does'))


class SubjectQuestionTest(unittest.TestCase):

    def build(self, text, start, end):
        doc = nlp(text)
        span = doc[start:end]
        candidate = Candidate(span.text, span, next(doc.sents), '', 0.0)
        return HybridAssessmentSystem(5).build_subject_question(candidate)

    def test_replaces_the_whole_subject(self):
        self.assertEqual(
            self.build('New employees will be asked to certify their '
                       'commitment to the standards in this Code.', 0, 2),
            'Who will be asked to certify their commitment to the '
            'standards in this Code?')

    def test_skips_a_subject_the_answer_only_partly_covers(self):
        # Used to give "What of confidential information requires ...?"
        self.assertEqual(
            self.build('Disclosure of confidential information requires a '
                       'clear business need and authorization.', 0, 1),
            '')

    def definition(self, text, start, end):
        doc = nlp(text)
        span = doc[start:end]
        candidate = Candidate(span.text, span, next(doc.sents), '', 0.0)
        return HybridAssessmentSystem(5).build_definition_question(candidate)

    def test_definition_asks_for_the_whole_complement(self):
        self.assertEqual(
            self.definition("Harry Caray's Management Corporation is our "
                            'management company.', 7, 9),
            "What is Harry Caray's Management Corporation?")

    def test_definition_skips_a_partial_complement(self):
        # Used to give "What is your immediate supervisor or manager?"
        # with the answer "best place".
        self.assertEqual(
            self.definition('Your immediate supervisor or manager is the '
                            'best place to start when you have a question.',
                            7, 9),
            '')


class OptionTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.generator = IncorrectAnswerGenerator(
            'The Law Department approves gifts.', candidates=[])

    def test_rejects_options_naming_the_same_thing(self):
        same = [
            ("McDonald's Business Integrity", 'Business Integrity line'),
            ("non-exempt employee's regular rate",
             "employee's regular hourly rate"),
            ('initial period', 'initial employment period'),
            ('The Us', 'The United States'),
        ]
        for option, answer in same:
            with self.subTest(option=option):
                self.assertTrue(self.generator._is_alias(option, answer))

    def test_keeps_genuinely_different_options(self):
        different = [
            ('Internal Audit Department', 'Law Department'),
            ('perceived conflict', 'actual conflict'),
        ]
        for option, answer in different:
            with self.subTest(option=option):
                self.assertFalse(self.generator._is_alias(option, answer))

    def test_contentless_phrases_are_not_answers(self):
        # spaCy lemmatises "following" to "follow", which once made
        # "following situations" an answer key.
        doc = nlp('The following situations may give rise to competition '
                  'law risks for the Company.')
        texts = [c.text.lower()
                 for c in extract_candidates(doc, list(doc.sents))]
        self.assertNotIn('following situations', texts)

    def test_entity_answers_drop_the_leading_article(self):
        doc = nlp('No employee may accept outside compensation without the '
                  'prior approval of the Audit Committee of the Board.')
        texts = [c.text for c in extract_candidates(doc, list(doc.sents))]
        self.assertTrue(texts)
        self.assertFalse([t for t in texts if t.lower().startswith('the ')])


class FreeResponseGradingTest(unittest.TestCase):

    def test_reference_answer_is_correct(self):
        answer = 'Employees must report violations to the Law Department.'
        self.assertEqual(
            evaluate_free_response(answer, answer)['status'], 'Correct')

    def test_blank_answer_is_incorrect(self):
        self.assertEqual(
            evaluate_free_response('  ', 'Report violations.')['status'],
            'Incorrect')


class TextFileTest(unittest.TestCase):
    '''.txt uploads come in whatever encoding the editor chose.'''

    TEXT = ("Yum!’s employees must report concerns to the Law "
            'Department.\nNo one may retaliate against a person who reports '
            'a concern in good faith.\n')

    def read_back(self, data, extension='txt'):
        with tempfile.NamedTemporaryFile('wb', suffix='.txt',
                                         delete=False) as txt_file:
            txt_file.write(data)
        try:
            return pdf2text(txt_file.name, extension)
        finally:
            os.remove(txt_file.name)

    def test_reads_every_common_windows_encoding(self):
        # UTF-16 is Notepad's "Unicode" and PowerShell's default for ">";
        # read as UTF-8 it used to yield no questions at all.
        for encoding in ('utf-8', 'utf-8-sig', 'utf-16', 'cp1252'):
            with self.subTest(encoding=encoding):
                self.assertEqual(
                    self.read_back(self.TEXT.encode(encoding)), self.TEXT)

    def test_normalises_windows_line_endings(self):
        data = self.TEXT.replace('\n', '\r\n').encode('utf-16')
        self.assertEqual(self.read_back(data), self.TEXT)

    def test_extension_case_does_not_matter(self):
        self.assertEqual(
            self.read_back(self.TEXT.encode('utf-8'), 'TXT'), self.TEXT)


class SamplePdfTest(unittest.TestCase):
    '''End to end on the documents shipped in pdf/.'''

    @classmethod
    def setUpClass(cls):
        cls.kfc = pdf2text('pdf/Code-of-Conduct-English_KFC.pdf', 'pdf')
        cls.hcrg = pdf2text('pdf/hcrg_employee_handbook.pdf', 'pdf')

    def assertSoundMultipleChoice(self, question):
        options = question['options']
        self.assertEqual(len(options), 4, question)
        self.assertEqual(len(set(options)), 4, question)
        self.assertEqual(options.count(question['answer']), 1, question)
        self.assertFalse(
            [o for o in options if o.lower().startswith('the ')], question)

    def test_blank_based_quiz_is_complete_and_sound(self):
        questions = txt2questions(self.kfc)
        self.assertEqual(len(questions), 5)
        for question in questions.values():
            self.assertIn('__________', question['question'])
            self.assertSoundMultipleChoice(question)

    def test_free_based_quiz_names_the_right_document(self):
        questions = list(txt2questions_free(self.hcrg).values())
        self.assertEqual(len(questions), 5)

        free = [q for q in questions if q['type'] == 'Free-Response']
        self.assertEqual(len(free), 1)
        self.assertIn('the Handbook', free[0]['question'])
        self.assertNotIn('Code of Conduct', free[0]['question'])

        for question in questions:
            self.assertNotRegex(question['question'], r'^What (of|in|to) ')
            if question['type'] == 'Multiple-Choice':
                self.assertSoundMultipleChoice(question)

    def test_split_words_are_repaired(self):
        cleaned = clean_document(self.hcrg)
        for repaired in ('Management Corporation', 'extreme emergency',
                         'Wednesday through Tuesday', 'paid-time-off'):
            self.assertIn(repaired, cleaned)
        for broken in ('M anagement', 'extrem e', 'thro ugh', 'paid -time'):
            self.assertNotIn(broken, cleaned)

    def test_document_without_sentences_yields_no_questions(self):
        netflix = pdf2text('pdf/netflix-culture.pdf', 'pdf')
        self.assertEqual(txt2questions(netflix), {})


if __name__ == '__main__':
    unittest.main()
