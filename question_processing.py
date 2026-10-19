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
from nlp_models import semantic_model

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
