import codecs

try:  # pypdf is the maintained successor of PyPDF2
    from pypdf import PdfReader
    from pypdf.errors import FileNotDecryptedError, PdfReadError
except ImportError:  # pragma: no cover - legacy installs
    from PyPDF2 import PdfReader
    from PyPDF2.errors import FileNotDecryptedError, PdfReadError

from question_generation_main import QuestionGeneration, QuestionGeneration_free
from question_processing import drop_incomplete_questions


class UnreadableDocumentError(Exception):
    """ The upload cannot be read; the message is safe to show the user. """


# UTF-32 first: its little-endian mark begins with UTF-16's.
_BYTE_ORDER_MARKS = (
    (codecs.BOM_UTF32_LE, 'utf-32'),
    (codecs.BOM_UTF32_BE, 'utf-32'),
    (codecs.BOM_UTF16_LE, 'utf-16'),
    (codecs.BOM_UTF16_BE, 'utf-16'),
)


def _decode_text(data: bytes) -> str:
    """ Decode a .txt upload in whichever encoding saved it.

    Reading every file as UTF-8 turned a UTF-16 one - Notepad's
    "Unicode", PowerShell's default for `>` - into text with a NUL after
    every letter, from which no question could be built, and turned the
    curly quotes of an ANSI (cp1252) file into replacement characters. """

    for mark, encoding in _BYTE_ORDER_MARKS:
        if data.startswith(mark):
            return data.decode(encoding, errors='replace')
    try:
        return data.decode('utf-8-sig')     # also drops a UTF-8 BOM
    except UnicodeDecodeError:
        return data.decode('cp1252', errors='replace')


def pdf2text(file_path: str, file_exten: str) -> str:
    """ Converts a given file to text content """

    _content = ''

    # Identify file type and get its contents
    if file_exten.lower() == 'pdf':
        try:
            with open(file_path, 'rb') as pdf_file:
                reader = PdfReader(pdf_file)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        _content += text + "\n"
        # FileNotDecryptedError subclasses PdfReadError, so it goes first.
        except FileNotDecryptedError as exc:
            raise UnreadableDocumentError(
                'This PDF is password-protected. Remove the password and '
                'upload it again.'
            ) from exc
        except PdfReadError as exc:
            raise UnreadableDocumentError(
                'This file could not be read as a PDF. It may be damaged, '
                'or not actually a PDF.'
            ) from exc
        print('PDF operation done!')

    elif file_exten.lower() == 'txt':
        with open(file_path, 'rb') as txt_file:
            # Universal newlines, as text-mode reading gave before.
            _content = _decode_text(txt_file.read())
            _content = _content.replace('\r\n', '\n').replace('\r', '\n')
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
# this a single rejected question leaves the quiz short.  The sound
# spares are kept too: a reviewer can swap one in for a weak question.
_OVERGENERATE = 4

_GENERATORS = {
    'blank_based': QuestionGeneration,
    'free_based': QuestionGeneration_free,
}


def generate_quiz(doc: str, method: str, n=5, o=4):
    """ Return (questions, spares) for a document.

    questions : the n best sound questions, in order
    spares    : the remaining sound questions, best first
    """

    if method not in _GENERATORS:
        raise ValueError(f'Unknown generator: {method}')

    generator = _GENERATORS[method](n + _OVERGENERATE, o)
    questions = _options_to_list(generator.generate_questions_dict(doc))
    sound = drop_incomplete_questions(questions, min_options=o)

    ordered = [sound[number] for number in sorted(sound)]
    for question_data in ordered:
        question_data.setdefault('type', 'Multiple-Choice')
    return ordered[:n], ordered[n:]


def _numbered(questions: list) -> dict:
    return {number: question for number, question in enumerate(questions, 1)}


def txt2questions(doc: str, n=5, o=4) -> dict:
    """ Get all fill-in-the-blank questions and their options """

    return _numbered(generate_quiz(doc, 'blank_based', n, o)[0])


def txt2questions_free(doc: str, n=5, o=4) -> dict:
    """ Get a mixed free-response / multiple-choice assessment """

    return _numbered(generate_quiz(doc, 'free_based', n, o)[0])
