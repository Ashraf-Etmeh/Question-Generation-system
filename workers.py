try:  # pypdf is the maintained successor of PyPDF2
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - legacy installs
    from PyPDF2 import PdfReader

from question_generation_main import QuestionGeneration, QuestionGeneration_free
from question_processing import drop_incomplete_questions


def pdf2text(file_path: str, file_exten: str) -> str:
    """ Converts a given file to text content """

    _content = ''

    # Identify file type and get its contents
    if file_exten.lower() == 'pdf':
        with open(file_path, 'rb') as pdf_file:
            reader = PdfReader(pdf_file)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    _content += text + "\n"
            print('PDF operation done!')

    elif file_exten == 'txt':
        with open(file_path, 'r', encoding='utf-8', errors='replace') as txt_file:
            _content = txt_file.read()
            print('TXT operation done!')

    return _content


def _options_to_list(questions: dict) -> dict:
    """Normalise option dicts to lists for the template layer."""

    for question_data in questions.values():
        options = question_data.get('options')
        if isinstance(options, dict):
            question_data['options'] = list(options.values())
    return questions


# The quality gate drops any item it cannot vouch for, so ask the
# generators for a few spares and trim back to n afterwards.  Without
# this a single rejected question leaves the quiz short.
_OVERGENERATE = 4


def _trim(questions: dict, n: int) -> dict:
    return {number: questions[number]
            for number in sorted(questions)[:n]}


def txt2questions(doc: str, n=5, o=4) -> dict:
    """ Get all fill-in-the-blank questions and their options """

    generator = QuestionGeneration(n + _OVERGENERATE, o)
    questions = generator.generate_questions_dict(doc)
    questions = _options_to_list(questions)
    return _trim(drop_incomplete_questions(questions, min_options=o), n)


def txt2questions_free(doc: str, n=5, o=4) -> dict:
    """ Get a mixed free-response / multiple-choice assessment """

    generator = QuestionGeneration_free(n + _OVERGENERATE, o)
    questions = generator.generate_questions_dict(doc)
    questions = _options_to_list(questions)
    return _trim(drop_incomplete_questions(questions, min_options=o), n)
