'''Final quality gate and free-response scoring.

This module previously tried to repair broken questions after the fact:
a blocklist of regexes deleted known-bad phrases from stems, and any MCQ
short of options was padded with invented answers ("General company
policy", "Standard administrative procedure").  Both are worse than
having one fewer question - the padding in particular is trivially
recognisable, so every question it touched became a free point.

The generators now produce sound items or none, so the job here is to
verify, not to repair.  Anything that fails verification is dropped and
the remaining questions are renumbered.
'''
import re

from sentence_transformers import util

from candidate_selection import normalise_phrase
from nlp_models import nlp, semantic_model

from question_extraction import BLANK


def _is_sound_question(question_data, min_options):
    '''True when an item is complete and internally consistent.'''
    question = (question_data.get('question') or '').strip()
    answer = (question_data.get('answer') or '').strip()

    if not question or not answer:
        return False
    if len(question.split()) < 5:
        return False
    if not (question.endswith(('?', '.', '!')) or BLANK in question):
        return False

    if question_data.get('type') == 'Free-Response':
        # The reference answer must be a real sentence, not a fragment.
        return len(answer.split()) >= 6

    options = question_data.get('options') or []
    if len(options) < min_options:
        return False

    # Exactly one option may match the answer, and no two may coincide.
    normalised = [normalise_phrase(str(option)) for option in options]
    if len(set(normalised)) != len(normalised):
        return False
    if normalised.count(normalise_phrase(answer)) != 1:
        return False

    # The answer must not also be sitting in the stem.
    stem_without_blank = question.replace(BLANK, ' ')
    if re.search(r'\b' + re.escape(answer) + r'\b',
                 stem_without_blank, re.IGNORECASE):
        return False

    return True


def drop_incomplete_questions(questions, min_options=4):
    '''Keep only sound questions and renumber them from 1.

    Params:
        * questions   : dict<int, question record>
        * min_options : options an MCQ must have to be usable
    Returns:
        * dict<int, question record>
    '''
    kept = {}
    counter = 1

    for _number, question_data in sorted(questions.items()):
        if not _is_sound_question(question_data, min_options):
            continue
        kept[counter] = question_data
        counter += 1

    return kept


# ---------------------------------------------------------------- #
# Review flags: reasons for a reviewer to look twice
# ---------------------------------------------------------------- #

LONG_QUESTION_WORDS = 35
LONG_REFERENCE_WORDS = 40

# A clause opening with one of these is subordinate, not spliced on:
# "When you are hired, we are required ...".  spaCy does not always
# mark them as such ("when" is an adverb, "should" an inverted auxiliary).
_SUBORDINATE_OPENERS = frozenset({
    'when', 'whenever', 'while', 'once', 'if', 'unless', 'should', 'where',
    'wherever', 'as', 'since', 'after', 'before', 'until', 'although',
    'though', 'because', 'whether', 'so', 'that', 'what', 'which', 'who',
    'how', 'why', 'to', 'for',
})


def _has_subject(token):
    return any(child.dep_ in ('nsubj', 'nsubjpass', 'expl')
               for child in token.children)


def joins_clauses_with_comma(sentence):
    '''True for a comma splice: two clauses, each with its own subject,
    joined by a bare comma.

    "Our scheduled work week runs Monday through Sunday for Lombard and
    Rosemont, other locations are Wednesday through Tuesday." turned into
    a question that reads awkwardly.  On the sample documents this fires
    on about 4 of 900 usable sentences, so it suits an advisory flag.
    '''
    doc = sentence.doc
    for token in sentence:
        if token.pos_ not in ('VERB', 'AUX') or token.dep_ == 'ROOT':
            continue
        if not _has_subject(token):
            continue
        if any(child.dep_ in ('mark', 'cc') or child.tag_ in ('WRB', 'WDT', 'WP')
               for child in token.children):
            continue
        if token.left_edge.lower_ in _SUBORDINATE_OPENERS:
            continue
        head = token.head
        if head.pos_ not in ('VERB', 'AUX') or not _has_subject(head):
            continue

        # Clause, comma, main clause.
        if token.i < head.i and token.dep_ in ('advcl', 'ccomp', 'parataxis'):
            after = token.right_edge.i + 1
            if (after + 1 < sentence.end and doc[after].text == ','
                    and doc[after + 1].lower_ not in _SUBORDINATE_OPENERS):
                return True
        # Main clause, comma, clause.
        if token.i > head.i and token.dep_ in ('conj', 'ccomp', 'parataxis'):
            before = token.left_edge.i - 1
            if before >= sentence.start and doc[before].text == ',':
                return True
    return False


def review_flags(question_data):
    '''Reasons a reviewer should look twice at a question; [] when none.

    Advisory only: every question is shown to the reviewer anyway, these
    just mark the ones most likely to need an edit.
    '''
    flags = []
    question = question_data.get('question') or ''
    is_free = question_data.get('type') == 'Free-Response'

    if len(question.split()) > LONG_QUESTION_WORDS:
        flags.append('Long question: consider trimming it.')

    if is_free:
        reference = question_data.get('answer') or ''
        if len(reference.split()) > LONG_REFERENCE_WORDS:
            flags.append('Long reference answer: hard to recall in full.')
        return flags

    # Near-duplicate options are not flagged here: the distractor
    # generator already rejects them (incorrect_answer_generation).
    source = question_data.get('source_sentence') or ''
    if source:
        sentence = next(nlp(source).sents, None)
        if sentence is not None and joins_clauses_with_comma(sentence):
            flags.append('Reads awkwardly: the source sentence joins two '
                         'statements with a comma.')

    return flags


def evaluate_free_response(
    employee_answer,
    reference_answer,
    threshold=0.60
):
    '''Evaluate a free-response answer using semantic similarity.'''

    if not employee_answer or not employee_answer.strip():
        return {'score': 0.0, 'status': 'Incorrect'}

    if not reference_answer or not reference_answer.strip():
        return {'score': 0.0, 'status': 'Cannot evaluate'}

    embeddings = semantic_model.encode(
        [employee_answer, reference_answer],
        convert_to_tensor=True
    )

    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()

    status = 'Correct' if similarity >= threshold else 'Incorrect'

    return {'score': similarity, 'status': status}
