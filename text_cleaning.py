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
    r'\b(see page|refer to page|click here|go to table|'
    r'for more information,? see|table of contents)\b',
    re.IGNORECASE,
)

_ALL_CAPS_RUN = re.compile(r'\b[A-Z]{3,}(?:\s+[A-Z]{2,}){2,}')


def _alpha_ratio(sentence):
    meaningful = [c for c in sentence if not c.isspace()]
    if not meaningful:
        return 0.0
    allowed = sum(c.isalpha() or c in ".,';!?-()$%" or c.isdigit()
                  for c in meaningful)
    return allowed / len(meaningful)


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
    if _NON_CONTENT.search(text):
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
