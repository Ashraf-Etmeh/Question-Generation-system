'''Fill-in-the-blank question extraction.

The cloze form is the safest multiple-choice frame for a policy
document: the stem is a real sentence from the source, so it can never
be ungrammatical, and the surrounding words supply the context the
reader needs to choose between the options.

What makes a cloze item good or bad is *where* the blank goes.  A blank
at the very start of a fragment - "____ brand restaurant in Asia to ship
supplies from." - leaves nothing to reason from.  So a blank is only
accepted when enough of the sentence survives on both sides of it.
'''
import re

from candidate_selection import (
    content_key,
    extract_candidates,
    is_redundant_with,
    rank_candidates,
)
from nlp_models import nlp
from text_cleaning import select_quality_sentences

BLANK = '__________'

# The stem has to retain enough words to be answerable.
_MIN_CONTEXT_WORDS = 6
# ... and enough of them must come before the blank, otherwise the reader
# is guessing at a sentence that has not started yet.
_MIN_LEADING_WORDS = 2


class QuestionExtractor:
    '''Extracts fill-in-the-blank questions from a document.'''

    def __init__(self, num_questions):
        self.num_questions = num_questions
        self.ner_tagger = nlp
        self.questions_dict = dict()
        self.doc = None
        self.sentences = []
        self.candidates = []

    def get_questions_dict(self, document):
        '''Return {number: {question, answer, answer_label}}.

        Params:
            * document : cleaned document text
        Returns:
            * dict
        '''
        self.doc = self.ner_tagger(document)
        self.sentences = select_quality_sentences(self.doc)

        raw_candidates = extract_candidates(self.doc, self.sentences)
        self.candidates = rank_candidates(raw_candidates, self.sentences)

        self.form_questions()
        return self.questions_dict

    def build_cloze(self, candidate):
        '''Blank the candidate out of its sentence.

        Returns the stem, or '' when the resulting item would not give
        the reader enough to work with.
        '''
        sentence = candidate.sentence
        sentence_text = sentence.text.strip()

        # Work in sentence-relative character offsets.
        start = candidate.span.start_char - sentence.start_char
        end = candidate.span.end_char - sentence.start_char
        if start < 0 or end > len(sentence_text):
            return ''

        before = sentence_text[:start]
        after = sentence_text[end:]

        leading_words = len(before.split())
        trailing_words = len(after.split())
        if leading_words < _MIN_LEADING_WORDS:
            return ''
        if leading_words + trailing_words < _MIN_CONTEXT_WORDS:
            return ''

        stem = f'{before}{BLANK}{after}'
        stem = re.sub(r'\s+', ' ', stem).strip()

        # A determiner immediately before the blank leaks the answer's
        # article ("a __________" when the answer starts with "the").
        stem = re.sub(r'\b(a|an)\s+' + re.escape(BLANK), BLANK, stem)

        if not stem.endswith(('.', '!', '?')):
            stem += '.'
        return stem

    def form_questions(self):
        '''Populate questions_dict, one question per sentence.

        Candidates are also spread across topics: the highest-scoring
        phrases in a corporate document are all variations on the company
        name, and five questions about "Yum! business" / "Yum!'s
        business" / "Yum! employees" test nothing.
        '''
        used_sentences = set()
        used_keys = []
        counter = 1

        for candidate in self.candidates:
            if counter > self.num_questions:
                break

            sentence_key = candidate.sentence.text.strip()
            if sentence_key in used_sentences:
                continue
            if is_redundant_with(candidate.text, used_keys):
                continue

            stem = self.build_cloze(candidate)
            if not stem:
                continue

            used_sentences.add(sentence_key)
            used_keys.append(content_key(candidate.text))
            self.questions_dict[counter] = {
                'question': stem,
                # Keep the document's own surface form - re-casing with
                # .title() produced answers like "The Companys Board".
                'answer': candidate.text,
                'answer_label': candidate.label,
                'source_sentence': sentence_key,
            }
            counter += 1
