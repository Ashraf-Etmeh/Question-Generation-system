'''Document clean-up and sentence selection.

Raw PDF text is not prose. A `pypdf` extraction of a corporate policy
document interleaves real sentences with table-of-contents dot leaders,
running headers and footers, cover-page slogans, page numbers and
hard-wrapped lines that split words across a hyphen.

Every downstream stage (TF-IDF scoring, spaCy parsing, keyword ranking,
question framing) assumes it is reading sentences. Feeding it raw
extraction output is the single largest source of nonsensical questions:
a fragment such as "brand restaurant in Asia to ship supplies from" is
not a sentence, so no amount of clever templating turns it into a
sensible question.

This module therefore does two things:

    clean_document(raw_text) -> str
        Strip PDF furniture and rebuild wrapped lines into paragraphs
        while *preserving punctuation* - sentence boundaries, possessives
        and brand names such as "Yum!" all depend on it.

    select_quality_sentences(doc) -> list[Span]
        Keep only sentences that can actually carry a question: a real
        subject and finite verb, a sane length, and no layout noise.
'''
import re
import unicodedata
from collections import Counter
from functools import lru_cache

from nltk.corpus import wordnet as wn
from spacy.lang.en.stop_words import STOP_WORDS

# A line is table-of-contents furniture when it carries dot leaders
# ("Our Commitment to Integrity......5") or is a bare page number.
_DOT_LEADER = re.compile(r'\.{4,}\s*\d*\s*$')
_BARE_NUMBER = re.compile(r'^\s*\d{1,4}\s*$')

# Running headers / footers and cover-page furniture.
_FURNITURE_PATTERNS = (
    r'^\s*go\s+to\s*$',
    r'^\s*table\s+of\s+contents\s*$',
    r'^\s*(go\s+to\s+)?table\s+of\s*$',
    r'^\s*contents\s*$',
    r'^\s*page\s+\d+',
    r'^\s*\d+\s*\|',
    r'^\s*[-–—•*·]\s*$',
)
_FURNITURE = re.compile('|'.join(_FURNITURE_PATTERNS), re.IGNORECASE)

# Word split across a line break: "responsi-\nbility" -> "responsibility".
_HYPHEN_BREAK = re.compile(r'(\w)-\s*\n\s*(\w)')

# Terminal punctuation marks the end of a real sentence.
_SENTENCE_END = re.compile(r'[.!?]["\')\]]?\s*$')

_BULLET_PREFIX = re.compile(
    r'^\s*(?:[-–—•*·▪●>»]+|\(?[a-z]\)|\(?\d+[.)])\s+'
)

# Characters that only ever arrive from PDF layout, never from prose.
_UNICODE_FIXES = {
    '‘': "'", '’': "'", '“': '"', '”': '"',
    '–': '-', '—': '-', '…': '...', ' ': ' ',
    'ﬁ': 'fi', 'ﬂ': 'fl', '•': ' ', '▪': ' ',
}


def _normalise_unicode(text):
    text = unicodedata.normalize('NFKC', text)
    for bad, good in _UNICODE_FIXES.items():
        text = text.replace(bad, good)
    return text


def _find_repeated_lines(lines, min_repeats=4):
    '''Return lines that recur often enough to be running headers/footers.

    A real sentence is not repeated verbatim on a dozen pages; a header
    such as "YUM! GLOBAL CODE OF CONDUCT" is.
    '''
    counts = Counter(
        line.strip().lower()
        for line in lines
        if 3 <= len(line.strip()) <= 90
    )
    return {line for line, count in counts.items() if count >= min_repeats}


def _is_heading(line):
    '''True for short, unterminated lines - titles, labels, cover slogans.'''
    stripped = line.strip()
    if not stripped:
        return False
    if _SENTENCE_END.search(stripped):
        return False

    words = stripped.split()
    if len(words) > 12:
        return False

    letters = [c for c in stripped if c.isalpha()]
    if not letters:
        return True

    # Mostly-uppercase short line: a banner, not prose.
    upper_ratio = sum(c.isupper() for c in letters) / len(letters)
    return upper_ratio > 0.6 or len(words) <= 8


# A word followed by a lower-case word, matched with a lookahead so that
# every neighbouring pair is examined - "Wednesday thro ugh" must still
# reach "thro ugh".  The lookbehind keeps possessive tails ("'s") out.
_SPLIT_PAIR = re.compile(r"(?<![\w'’-])([A-Za-z]+) (?=([a-z]+)\b)")
_NEXT_WORD = re.compile(r' ([a-z]+)\b')
# WordNet's SemCor frequency above which a word counts as everyday
# English.  Separates "men" (35) or "web" (3) from dictionary oddities
# that pypdf fragments happen to spell - "thro", "comp", "th" (all 0).
_COMMON_WORD_COUNT = 2

# "paid -time-off", "non -exempt": a hyphen that lost its left neighbour.
_DETACHED_HYPHEN = re.compile(r'(?<=[A-Za-z]) -(?=[A-Za-z])')
# "Tuesday . Schedules", "Security , Information".
_SPACE_BEFORE_PUNCT = re.compile(r'(?<=\w) +([,;:.!?])(?=\s|$)')


@lru_cache(maxsize=None)
def _in_dictionary(word):
    return bool(wn.synsets(word))


@lru_cache(maxsize=None)
def _is_common_word(word):
    '''Stop word, or a word in everyday use according to WordNet.'''
    if word in STOP_WORDS:
        return True
    base = wn.morphy(word) or word
    usage = sum(
        lemma.count()
        for synset in wn.synsets(word)
        for lemma in synset.lemmas()
        if lemma.name().lower() in (word, base)
    )
    return usage >= _COMMON_WORD_COUNT


def _repair_split_words(text):
    '''Re-join words the PDF extractor broke apart.

    Some PDFs place glyph runs individually and pypdf inserts a space
    wherever the gap between runs looks wide, giving "M anagement",
    "tak ing" and "thro ugh".  Two neighbouring fragments are joined when

      * the joined word occurs intact elsewhere in this document, and
      * at least one fragment is not a word in its own right: neither
        everyday English nor a dictionary word seen outside this split.

    The second test is what keeps genuine pairs such as "over time" or
    "a rate" apart.  pypdf's layout mode avoids the splits, but it
    interleaves the columns of multi-column pages, which is worse.
    '''
    words = Counter(word.lower() for word in re.findall(r'[A-Za-z]+', text))
    pairs = Counter(
        (match.group(1).lower(), match.group(2))
        for match in _SPLIT_PAIR.finditer(text)
    )

    def is_word_here(fragment, split_count):
        # WordNet lists every letter ("n": nitrogen), and other splits
        # leave stray letters all over the text, so neither is evidence.
        if len(fragment) == 1:
            return fragment in ('a', 'i')
        return _is_common_word(fragment) or (
            _in_dictionary(fragment) and words[fragment] > split_count
        )

    def can_join(first, second):
        if not words[first + second]:
            return False
        split_count = pairs[(first, second)]
        return not (is_word_here(first, split_count)
                    and is_word_here(second, split_count))

    # Start of the last fragment that was joined to its left neighbour.
    joined_left = -1

    def join(match):
        nonlocal joined_left
        first, second = match.group(1), match.group(2)

        # A fragment joins one neighbour at most: "So n or" is "Son or",
        # not "Sonor".
        if match.start(1) == joined_left:
            return match.group(0)
        if not can_join(first.lower(), second):
            return match.group(0)

        # When the second fragment completes the word after it instead,
        # leave it for that join: "employee s hoes" is "employee shoes".
        following = _NEXT_WORD.match(text, match.end(2))
        if following:
            third = following.group(1)
            if can_join(second, third) and not is_word_here(
                    third, pairs[(second, third)]):
                return match.group(0)

        joined_left = match.start(2)
        return first                        # drop the space

    try:
        return _SPLIT_PAIR.sub(join, text)
    except LookupError:     # WordNet data not installed: leave text as is
        return text


def clean_document(raw_text):
    '''Turn raw PDF extraction output into paragraph text.

    Preserves punctuation - sentence segmentation downstream depends on
    it - while removing layout artefacts.

    Params:
        * raw_text : string straight out of the PDF/TXT reader
    Returns:
        * string : paragraphs separated by blank lines
    '''
    if not raw_text:
        return ''

    text = _normalise_unicode(raw_text)
    text = _HYPHEN_BREAK.sub(r'\1\2', text)

    lines = text.split('\n')
    repeated = _find_repeated_lines(lines)

    kept = []
    for line in lines:
        stripped = line.strip()

        if not stripped:
            kept.append('')                      # paragraph boundary
            continue
        if stripped.lower() in repeated:
            continue
        if _DOT_LEADER.search(stripped):
            continue
        if _BARE_NUMBER.match(stripped):
            continue
        if _FURNITURE.match(stripped):
            continue

        stripped = _BULLET_PREFIX.sub('', stripped)

        if _is_heading(stripped):
            kept.append('')                      # a heading acts as a break
            continue

        kept.append(stripped)

    # Re-flow: consecutive non-empty lines belong to one paragraph.
    paragraphs = []
    buffer = []
    for line in kept:
        if line:
            buffer.append(line)
        elif buffer:
            paragraphs.append(' '.join(buffer))
            buffer = []
    if buffer:
        paragraphs.append(' '.join(buffer))

    text = '\n\n'.join(paragraphs)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' *\n\n *', '\n\n', text)

    # Spacing damage from extraction, repaired once lines are re-flowed
    # so splits that straddled a line break are caught too.
    text = _repair_split_words(text)
    text = _DETACHED_HYPHEN.sub('-', text)
    text = _SPACE_BEFORE_PUNCT.sub(r'\1', text)
    return text.strip()


# ---------------------------------------------------------------- #
# Sentence-level quality gate
# ---------------------------------------------------------------- #

_MIN_WORDS = 8
_MAX_WORDS = 45

# Deictic openers make a sentence unusable out of context: the question
# would refer to a "this"/"they" the reader cannot resolve.
_DANGLING_OPENERS = (
    'this', 'that', 'these', 'those', 'it', 'they', 'he', 'she',
    'such', 'them', 'their', 'his', 'her', 'its', 'there',
)

# Cross-reference and navigation sentences carry no testable content.
_NON_CONTENT = re.compile(
    r'\b(see page|refer to page|click here|go to table|table of contents|'
    r'for (?:more|further) (?:specific |detailed )?'
    r'(?:information|guidance|direction|details))\b',
    re.IGNORECASE,
)
# "See the guidance given in the section titled ..." points elsewhere.
_CROSS_REFERENCE_OPENER = re.compile(
    r'^(?:please\s+)?(?:see|refer\s+to)\b', re.IGNORECASE
)

_DEMONSTRATIVES = frozenset({'this', 'that', 'these', 'those'})

_ALL_CAPS_RUN = re.compile(r'\b[A-Z]{3,}(?:\s+[A-Z]{2,}){2,}')


def _alpha_ratio(sentence):
    meaningful = [c for c in sentence if not c.isspace()]
    if not meaningful:
        return 0.0
    allowed = sum(c.isalpha() or c in ".,';!?-()$%" or c.isdigit()
                  for c in meaningful)
    return allowed / len(meaningful)


def _has_unresolved_demonstrative(span):
    '''A bare "this"/"those" refers to something outside the sentence.

    "Keep this in mind and exercise care when you use ..." cannot be
    answered without the sentence before it.  Determiners ("this Code"),
    the conjunction "in that", and modified pronouns ("those with whom
    we do business") are self-contained and allowed.
    '''
    for token in span:
        if (token.lower_ in _DEMONSTRATIVES and token.tag_ == 'DT'
                and token.dep_ not in ('det', 'mark')
                and not any(child.dep_ in ('relcl', 'acl', 'prep')
                            for child in token.children)):
            return True
    return False


def _is_finite(token):
    return token.tag_ == 'MD' or (
        token.pos_ in ('VERB', 'AUX')
        and 'Fin' in token.morph.get('VerbForm')
    )


def _inside_relative_clause(token, root):
    while token.i != root.i and token.head.i != token.i:
        if token.dep_ in ('relcl', 'acl'):
            return True
        token = token.head
    return False


def _lacks_main_clause(span):
    '''True for list items that only look like sentences.

    "Members of internal Intel groups that specialize in handling
    reports." and "Failing to monitor vendors for whom you have
    oversight." both carry a subject and a finite verb - but only inside
    a relative clause.  The sentence itself has no main clause, so a
    blank in it has nothing to hang on.
    '''
    root = span.root
    # A verb root other than a gerund is a clause - imperatives included.
    if root.pos_ in ('VERB', 'AUX') and root.tag_ != 'VBG':
        return False

    finite = [token for token in span if _is_finite(token)]
    # A noun root next to a modal is a misparse of a real sentence:
    # "Employees who are involved in an accident ... must report it."
    if root.pos_ != 'VERB' and any(token.tag_ == 'MD' for token in finite):
        return False
    return all(_inside_relative_clause(token, root) for token in finite)


def is_quality_sentence(span):
    '''Decide whether a parsed spaCy sentence can carry a question.

    Params:
        * span : spacy.tokens.Span - one sentence from doc.sents
    Returns:
        * bool
    '''
    text = span.text.strip()
    words = text.split()

    # A newline inside a "sentence" means the segmenter ran across a
    # paragraph break and glued two unrelated passages together.
    if '\n' in span.text:
        return False

    # Likewise a full stop followed by a lower-case word: the sentence
    # boundary was missed and the tail belongs to another sentence.
    if re.search(r'[.!?]\s+[a-z]', text):
        return False

    if not (_MIN_WORDS <= len(words) <= _MAX_WORDS):
        return False
    if not text[:1].isupper():
        return False
    if not _SENTENCE_END.search(text):
        return False
    if _NON_CONTENT.search(text) or _CROSS_REFERENCE_OPENER.search(text):
        return False
    if _alpha_ratio(text) < 0.90:
        return False
    if _ALL_CAPS_RUN.search(text):
        return False
    # A sentence that is itself a question (a section heading such as
    # "What is the purpose of this Code?") cannot be turned into one.
    if text.endswith('?'):
        return False

    if words[0].lower().strip('.,') in _DANGLING_OPENERS:
        return False

    # Require a real clause: a finite verb together with a subject.
    has_subject = any(
        tok.dep_ in ('nsubj', 'nsubjpass', 'expl') for tok in span
    )
    has_finite_verb = any(
        tok.pos_ in ('VERB', 'AUX')
        and 'Ger' not in tok.morph.get('VerbForm')
        for tok in span
    )
    if not (has_subject and has_finite_verb):
        return False

    if _lacks_main_clause(span) or _has_unresolved_demonstrative(span):
        return False

    return span.root.pos_ in ('VERB', 'AUX', 'NOUN', 'PROPN', 'ADJ')


_BRAND_TOKEN = re.compile(r'\b([A-Z][A-Za-z]{1,15}!)')


def register_brand_tokens(spacy_nlp, text):
    '''Teach the tokeniser that "Yum!" is one token, not "Yum" + "!".

    Brand names ending in punctuation break both tokenisation and
    sentence segmentation: spaCy treats the "!" as sentence-final, so
    "The Yum! Brands Social Media Code describes ..." is split, and the
    noun chunk arrives as "Brands Social Media Code" inside a sentence
    that starts mid-phrase.

    The decision is made from the document itself rather than a
    hard-coded list: a token is merged only when it usually appears in
    the middle of a sentence.  Genuine sentence endings are unaffected
    because the document writes its own full stop after the brand
    ("... work for or with Yum!.").

    Params:
        * spacy_nlp : the shared spaCy pipeline
        * text      : the cleaned document
    '''
    totals = Counter()
    terminal = Counter()

    for match in _BRAND_TOKEN.finditer(text):
        token = match.group(1)
        totals[token] += 1
        following = text[match.end():match.end() + 2].lstrip()[:1]
        if not following or following in '.,;:?!':
            terminal[token] += 1

    for token, total in totals.items():
        if total >= 5 and terminal[token] / total < 0.5:
            spacy_nlp.tokenizer.add_special_case(token, [{'ORTH': token}])


def select_quality_sentences(doc):
    '''Return the subset of doc.sents that pass the quality gate.

    Params:
        * doc : a parsed spaCy Doc
    Returns:
        * list<spacy.tokens.Span>
    '''
    seen = set()
    selected = []
    for span in doc.sents:
        if not is_quality_sentence(span):
            continue
        key = span.text.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        selected.append(span)
    return selected
