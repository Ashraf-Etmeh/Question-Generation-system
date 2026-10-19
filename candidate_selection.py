'''Selection and ranking of answer keys.

A multiple-choice question is only as good as the thing it asks about.
The previous selection step accepted any named entity or noun chunk,
which is how answer keys such as "A Country", "The Companys Board" and
"This Business Relationship" ended up on the quiz: they are grammatical
noun phrases but they carry no testable content, and several of them are
merely different surface forms of the same idea.

This module produces `Candidate` records that are known to be

  * anchored - the phrase occurs verbatim in a quality sentence, so the
    question can always be rendered against real context;
  * contentful - determiners, pronouns and vacuous head nouns are gone;
  * salient - ranked by the TF-IDF mass of the words they contain.
'''
import re
from collections import namedtuple

from sklearn.feature_extraction.text import TfidfVectorizer

# text, the spaCy span of the phrase, the sentence span containing it,
# the entity label ('' when the candidate came from a noun chunk) and the
# salience score assigned by rank_candidates().
Candidate = namedtuple(
    'Candidate', ['text', 'span', 'sentence', 'label', 'score']
)

# Entity labels worth asking about.  ORDINAL/CARDINAL are excluded on
# their own because bare numbers rarely make a fair question.
INFORMATIVE_LABELS = frozenset({
    'PERSON', 'ORG', 'GPE', 'LOC', 'NORP', 'FAC', 'EVENT',
    'LAW', 'PRODUCT', 'DATE', 'TIME', 'MONEY', 'PERCENT', 'QUANTITY',
})

_LEADING_DETERMINER = re.compile(
    r'^(the|a|an|this|that|these|those|its|their|our|my|your|his|her|'
    r'any|all|each|every|some|such|no)\s+',
    re.IGNORECASE,
)

# Head nouns that say nothing on their own.  A phrase is rejected when
# every content word it has is drawn from this set - "a country",
# "the following", "this business relationship".
_VACUOUS_WORDS = frozenset({
    'thing', 'things', 'matter', 'matters', 'item', 'items', 'way', 'ways',
    'case', 'cases', 'example', 'examples', 'part', 'parts', 'kind', 'kinds',
    'type', 'types', 'area', 'areas', 'aspect', 'aspects', 'point', 'points',
    'following', 'above', 'below', 'other', 'others', 'one', 'ones',
    'person', 'people', 'someone', 'anyone', 'everyone', 'something',
    'country', 'countries', 'place', 'places', 'situation', 'situations',
    'business', 'company', 'companies', 'relationship', 'relationships',
    'use', 'uses', 'time', 'times', 'number', 'numbers', 'order', 'orders',
    'question', 'questions', 'answer', 'answers', 'issue', 'issues',
    'information', 'detail', 'details', 'fact', 'facts', 'reason', 'reasons',
    'result', 'results', 'action', 'actions', 'decision', 'decisions',
    'policy', 'policies', 'provision', 'provisions', 'code', 'codes',
    'employee', 'employees', 'member', 'members', 'group', 'groups',
    'year', 'years', 'day', 'days', 'month', 'months', 'week', 'weeks',
})

_COORDINATION = re.compile(r'\s+(?:and|or|nor|but)\s+', re.IGNORECASE)

_MIN_CHARS = 4
_MAX_CHARS = 45
_MAX_WORDS = 5


def content_key(text):
    '''Inflection-insensitive set of the content words in a phrase.

    Two candidates that share most of their content key are about the
    same thing; used to keep a five-question quiz from asking five
    variations of one topic.
    '''
    words = normalise_phrase(text).replace("'s", '').replace("'", '').split()
    return {word.rstrip('s') for word in words if len(word) > 2}


def is_redundant_with(text, previous_keys, overlap=0.5):
    '''True when a phrase largely repeats an already-used answer.'''
    key = content_key(text)
    if not key:
        return True
    for previous in previous_keys:
        if not previous:
            continue
        shared = len(key & previous)
        if shared / min(len(key), len(previous)) >= overlap:
            return True
    return False


def normalise_phrase(text):
    '''Lower-case, determiner-free, punctuation-free form of a phrase.

    Used for equality and containment tests so that "the Board" and
    "The Board." are recognised as the same phrase.
    '''
    text = text.strip().strip('.,;:!?"\'()[]')
    text = _LEADING_DETERMINER.sub('', text)
    text = re.sub(r"[^\w\s']", ' ', text)
    return re.sub(r'\s+', ' ', text).strip().lower()


def strip_determiner(text):
    '''Remove a leading determiner while preserving the original casing.'''
    return _LEADING_DETERMINER.sub('', text.strip()).strip()


def _is_acceptable_phrase(text, lemmas):
    '''Reject phrases that cannot serve as an answer key.'''
    if not (_MIN_CHARS <= len(text) <= _MAX_CHARS):
        return False

    words = text.split()
    if not (1 <= len(words) <= _MAX_WORDS):
        return False

    # Must contain a letter; must not trail on a function word.
    if not any(c.isalpha() for c in text):
        return False
    if words[-1].lower() in ('of', 'in', 'to', 'for', 'and', 'or', 'with',
                             'the', 'a', 'an', 'that', 'this'):
        return False

    # A chunk spanning a coordination ("Yum! or other companies") is a
    # parse artefact, not a phrase anyone would offer as an answer.
    if _COORDINATION.search(text):
        return False

    # A dangling possessive ("Law Department's") is a truncated phrase.
    if re.search(r"'s?$", text, re.IGNORECASE):
        return False

    # Unbalanced brackets mean the phrase was cut mid-token, which is
    # how "Employee(s" reached the option list.
    if text.count('(') != text.count(')') or text.count('[') != text.count(']'):
        return False

    # Every content word vacuous -> the phrase tests nothing.
    content = [lemma for lemma in lemmas if lemma not in ('the', 'a', 'an')]
    if content and all(lemma in _VACUOUS_WORDS for lemma in content):
        return False

    return True


def _phrase_lemmas(span):
    return [
        token.lemma_.lower()
        for token in span
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]


def extract_candidates(doc, sentences):
    '''Collect answer-key candidates anchored in the given sentences.

    Params:
        * doc       : the parsed spaCy Doc
        * sentences : list<Span> of sentences that passed the quality gate
    Returns:
        * list<Candidate> with score still unset (0.0)
    '''
    # Map a character offset to the quality sentence that contains it, so
    # a candidate can be tied to the sentence it will be asked about.
    sentence_ranges = [
        (sentence.start_char, sentence.end_char, sentence)
        for sentence in sentences
    ]

    def containing_sentence(span):
        for start, end, sentence in sentence_ranges:
            if start <= span.start_char and span.end_char <= end:
                return sentence
        return None

    candidates = []
    seen = set()

    def add(span, label):
        text = span.text.strip().strip('.,;:!?"\'()[]')
        if not text:
            return
        sentence = containing_sentence(span)
        if sentence is None:
            return

        lemmas = _phrase_lemmas(span)
        if not _is_acceptable_phrase(text, lemmas):
            return

        key = normalise_phrase(text)
        if not key or key in seen:
            return
        seen.add(key)
        candidates.append(Candidate(text, span, sentence, label, 0.0))

    for ent in doc.ents:
        if ent.label_ in INFORMATIVE_LABELS:
            add(ent, ent.label_)

    for chunk in doc.noun_chunks:
        # Drop the determiner by re-slicing, so the answer key reads
        # "Board of Directors" rather than "the Board of Directors".
        span = chunk
        while len(span) > 1 and span[0].pos_ in ('DET', 'PRON'):
            span = span[1:]
        if len(span) == 0 or span.root.pos_ == 'PRON':
            continue
        if span.root.pos_ not in ('NOUN', 'PROPN'):
            continue
        add(span, '')

    return candidates


def rank_candidates(candidates, sentences, top_n=None):
    '''Score candidates by the TF-IDF mass of the words they contain.

    Params:
        * candidates : list<Candidate>
        * sentences  : list<Span> used as the TF-IDF corpus
        * top_n      : optional cap on the number returned
    Returns:
        * list<Candidate> sorted by descending score
    '''
    if not candidates or not sentences:
        return []

    corpus = [sentence.text for sentence in sentences]
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        matrix = vectorizer.fit_transform(corpus)
    except ValueError:          # corpus reduced to stop words only
        return list(candidates)

    vocabulary = vectorizer.vocabulary_
    # Mean TF-IDF weight of every term across the corpus.
    mean_weights = matrix.mean(axis=0).A1

    scored = []
    for candidate in candidates:
        score = 0.0
        for token in candidate.span:
            index = vocabulary.get(token.lower_)
            if index is not None:
                score += float(mean_weights[index])

        # A named entity is a better answer key than a bare noun chunk,
        # and a two-to-three word phrase is more specific than one word.
        if candidate.label:
            score *= 1.4
        word_count = len(candidate.text.split())
        if 2 <= word_count <= 3:
            score *= 1.15

        scored.append(candidate._replace(score=score))

    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:top_n] if top_n else scored
