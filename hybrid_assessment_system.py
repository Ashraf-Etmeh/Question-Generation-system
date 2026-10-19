'''Mixed free-response and multiple-choice assessment generation.

The previous version built every stem from one template:

    "What is the applicable policy or provision regarding {fragment}?"

where {fragment} was the source sentence with the answer deleted and
then truncated to twelve words.  Deleting a phrase from the middle of a
sentence does not leave a noun phrase behind, so the slot was filled
with things like "as an yum. recognizes its responsibility to the
countries where we do" - grammatical debris.

The replacement never assembles a stem out of fragments.  A question is
produced only when one of three transformations applies:

  1. Subject substitution - the answer is the grammatical subject, so
     replacing it with Who/What leaves the rest of the sentence intact
     and the result is guaranteed grammatical.
  2. Copular definition - "X is <answer>" becomes "What is X?".
  3. Cloze - everything else falls back to a fill-in-the-blank against
     the untouched sentence.

Free-response items follow the same rule: the stem is built from the
sentence's real subject and modal verb, never from leftover text.
'''
import re

from candidate_selection import (
    content_key,
    extract_candidates,
    is_redundant_with,
    normalise_phrase,
    rank_candidates,
    strip_determiner,
)
from nlp_models import nlp
from question_extraction import BLANK, QuestionExtractor
from text_cleaning import select_quality_sentences

# Entity label -> the interrogative pronoun that fits it.
_LABEL_TO_WH = {
    'PERSON': 'Who',
    'ORG': 'Which organisation',
    'NORP': 'Which group',
    'GPE': 'Which country or region',
    'LOC': 'Which location',
    'FAC': 'Which facility',
    'DATE': 'What period',
    'TIME': 'What time',
    'MONEY': 'What amount',
    'PERCENT': 'What proportion',
    'QUANTITY': 'What quantity',
    'LAW': 'Which law or regulation',
    'EVENT': 'Which event',
    'PRODUCT': 'Which product',
}

# Words that signal a sentence states a duty - the best raw material for
# a free-response question about a code of conduct.
_OBLIGATION_MARKERS = (
    'must', 'shall', 'required', 'require', 'responsible', 'responsibility',
    'obligation', 'obligated', 'expected', 'prohibited', 'forbidden',
    'may not', 'must not', 'should', 'never', 'always', 'entitled',
)

_MODAL_LEMMAS = frozenset({'must', 'shall', 'should', 'may', 'can', 'will'})

# Head nouns that denote people.  spaCy labels "local contact" and
# "new employees" as plain noun chunks, so the entity label alone is not
# enough to choose between "Who" and "What".
_PERSON_NOUNS = frozenset({
    'employee', 'employer', 'director', 'manager', 'supervisor', 'officer',
    'contact', 'official', 'representative', 'agent', 'contractor',
    'consultant', 'auditor', 'partner', 'supplier', 'vendor', 'franchisee',
    'shareholder', 'stakeholder', 'customer', 'candidate', 'applicant',
    'personnel', 'staff', 'worker', 'colleague', 'leader', 'executive',
    'member', 'board', 'committee', 'team', 'department',
})


class HybridAssessmentSystem:
    '''Builds a mixed free-response / multiple-choice assessment.'''

    def __init__(self, num_questions):
        self.num_questions = num_questions
        self.ner_tagger = nlp
        self.assessment_dict = dict()
        self.doc = None
        self.sentences = []
        self.candidates = []

    def get_assessment(self, document):
        '''Return {number: question record} for the whole assessment.

        Params:
            * document : cleaned document text
        Returns:
            * dict
        '''
        self.doc = self.ner_tagger(document)
        self.sentences = select_quality_sentences(self.doc)

        raw_candidates = extract_candidates(self.doc, self.sentences)
        self.candidates = rank_candidates(raw_candidates, self.sentences)

        self.assessment_dict = dict()
        used_sentences = set()

        counter = self._add_free_response(used_sentences, start=1)
        self._add_multiple_choice(used_sentences, start=counter)
        return self.assessment_dict

    # ------------------------------------------------------------ #
    # Free response
    # ------------------------------------------------------------ #

    def _obligation_sentences(self):
        '''Quality sentences that state a duty, most salient first.'''
        scored = []
        for sentence in self.sentences:
            lowered = sentence.text.lower()
            hits = sum(marker in lowered for marker in _OBLIGATION_MARKERS)
            if hits:
                scored.append((hits, len(sentence.text.split()), sentence))
        # Prefer many obligation cues and a substantial sentence.
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [sentence for _hits, _length, sentence in scored]

    @staticmethod
    def _expand_left(sentence, chunk):
        '''Re-attach proper-noun material the chunker left behind.

        "Yum!" tokenises as "Yum" + "!", and the punctuation cuts the
        noun chunk short: "The Yum! Brands Social Media Code" arrives as
        just "Brands Social Media Code".  Walk back over any proper
        nouns, attached punctuation and the determiner that opens them.
        '''
        start = chunk.start
        while start > sentence.start:
            previous = sentence.doc[start - 1]
            if previous.pos_ == 'PROPN' or previous.text in ('!', '&', '-'):
                start -= 1
            elif previous.pos_ == 'DET' and start - 1 > sentence.start:
                start -= 1
                break
            else:
                break
        return sentence.doc[start:chunk.end].text.strip()

    @staticmethod
    def _subject_phrase(sentence):
        '''The sentence's subject noun phrase, or '' when it is a pronoun.

        A pronoun subject ("they must report it") produces a question
        the reader cannot resolve, so those sentences are skipped.  The
        determiner is deliberately kept: dropping it yielded stems such
        as "what must franchisee do" instead of "what must a franchisee
        do".
        '''
        for chunk in sentence.noun_chunks:
            if chunk.root.dep_ not in ('nsubj', 'nsubjpass'):
                continue
            if chunk.root.pos_ == 'PRON':
                return ''
            text = HybridAssessmentSystem._expand_left(sentence, chunk)
            if len(text.split()) > 7 or len(strip_determiner(text)) < 3:
                return ''
            # The phrase appears mid-question, so de-capitalise it -
            # unless it opens with a proper noun ("Yum! employees").
            if chunk[0].pos_ != 'PROPN':
                text = text[:1].lower() + text[1:]
            return text
        return ''

    @staticmethod
    def _modal_of(sentence):
        for token in sentence:
            if token.lemma_.lower() in _MODAL_LEMMAS and token.pos_ == 'AUX':
                return token.lemma_.lower()
        return ''

    def build_free_response_question(self, sentence):
        '''Build an open question whose reference answer is the sentence.

        Returns (question, reference_answer) or ('', '') when no sound
        stem can be built from this sentence.
        '''
        subject = self._subject_phrase(sentence)
        if not subject:
            return '', ''

        modal = self._modal_of(sentence)
        lowered = sentence.text.lower()

        # "What must a franchisee do" only makes sense when the subject
        # is someone who can act.  A document title as subject produced
        # "What responsibility does the Code place on Social Media Code?"
        subject_root = next(
            (chunk.root for chunk in sentence.noun_chunks
             if chunk.root.dep_ in ('nsubj', 'nsubjpass')),
            None,
        )
        actor_subject = (
            subject_root is not None
            and (self._denotes_person(subject_root)
                 or subject_root.ent_type_ == 'ORG')
        )

        if not actor_subject:
            question = (
                f'What does the Code of Conduct set out in relation to '
                f'{subject}?'
            )
            question = re.sub(r'\s+', ' ', question).strip()
            if not self.is_valid_free_response_question(question):
                return '', ''
            return question, sentence.text.strip()

        if modal in ('must', 'shall'):
            question = (
                f'According to the Code of Conduct, what {modal} '
                f'{subject} do in this situation?'
            )
        elif 'prohibited' in lowered or 'may not' in lowered \
                or 'must not' in lowered or 'forbidden' in lowered:
            question = (
                f'What does the Code of Conduct prohibit in relation to '
                f'{subject}?'
            )
        elif 'responsible' in lowered or 'responsibility' in lowered:
            question = (
                f'What responsibility does the Code of Conduct place on '
                f'{subject}?'
            )
        else:
            question = (
                f'What requirement does the Code of Conduct set out for '
                f'{subject}?'
            )

        question = re.sub(r'\s+', ' ', question).strip()
        if not self.is_valid_free_response_question(question):
            return '', ''
        return question, sentence.text.strip()

    @staticmethod
    def is_valid_free_response_question(question):
        '''Reject stems that are too short, too long or ungrammatical.'''
        words = question.split()
        if not 6 <= len(words) <= 30:
            return False
        if not question.endswith('?'):
            return False
        if not question[0].isupper():
            return False
        # Leftover connectives betray a fragment-built stem.
        if re.search(r'\b(and|or|of|to|for|the|a|an)\s*\?$', question, re.I):
            return False
        if re.search(r'\b(\w+)\s+\1\b', question, re.IGNORECASE):
            return False
        return True

    def _add_free_response(self, used_sentences, start):
        '''Add one free-response item; return the next question number.'''
        for sentence in self._obligation_sentences():
            key = sentence.text.strip()
            if key in used_sentences:
                continue

            question, answer = self.build_free_response_question(sentence)
            if not question:
                continue

            used_sentences.add(key)
            self.assessment_dict[start] = {
                'type': 'Free-Response',
                'question': question,
                'answer': answer,
            }
            return start + 1

        return start

    # ------------------------------------------------------------ #
    # Multiple choice
    # ------------------------------------------------------------ #

    @staticmethod
    def _denotes_person(token):
        '''True when the head noun of an answer refers to people.'''
        if token.ent_type_ in ('PERSON', 'NORP'):
            return True
        return token.lemma_.lower() in _PERSON_NOUNS

    def build_subject_question(self, candidate):
        '''Turn "<answer> must report X." into "Who must report X?".

        Only applies when the answer is the whole subject noun phrase, so
        substituting the interrogative leaves a grammatical sentence.
        Returns '' when the pattern does not apply.
        '''
        sentence = candidate.sentence
        span = candidate.span

        root = span.root
        if root.dep_ not in ('nsubj', 'nsubjpass'):
            return ''

        # Quoted speech and first-person case-study narration do not
        # survive the substitution: "What has asked me to send the
        # supplies ... and 'he will take care of it from there.'?"
        if '"' in sentence.text or "'" in sentence.text.replace("'s", ''):
            return ''
        if re.search(r'\b(I|me|my|we|us|our)\b', sentence.text):
            return ''

        # The answer must cover the subject noun phrase in full, or the
        # remainder of that phrase would dangle in front of the verb.
        subject_chunk = None
        for chunk in sentence.noun_chunks:
            if chunk.root == root:
                subject_chunk = chunk
                break
        if subject_chunk is None:
            return ''
        if normalise_phrase(subject_chunk.text) != normalise_phrase(span.text):
            return ''

        # The subject has to open the sentence; substituting mid-sentence
        # would need the rest re-ordered.
        if subject_chunk.start_char - sentence.start_char > 0:
            return ''

        # spaCy splits a coordinated subject ("relevant laws or
        # requirements") into separate chunks, so replacing only the
        # first one strands the conjunction: "What or requirements are
        # complex?".  Skip any subject that is part of a coordination.
        if any(child.dep_ == 'conj' for child in root.children):
            return ''
        next_index = subject_chunk.end - sentence.start
        if next_index < len(sentence) and sentence[next_index].pos_ == 'CCONJ':
            return ''

        wh_word = _LABEL_TO_WH.get(candidate.label)
        if not wh_word:
            wh_word = 'Who' if self._denotes_person(root) else 'What'

        # A terse sentence leaves a stem with no substance behind the
        # interrogative ("What do more than keep Yum?").
        if len(sentence.text.split()) < 10:
            return ''

        remainder = sentence.text[
            subject_chunk.end_char - sentence.start_char:
        ].strip()
        if len(remainder.split()) < 5:
            return ''

        # A colon introduces a list, which reads as a fragment once the
        # subject is gone: "What take many forms, including bans on:
        # Exports to a prohibited country?"
        if ':' in remainder:
            return ''

        # "What"/"Who" take singular agreement.  Replacing a plural
        # subject in front of a finite verb leaves the verb stranded in
        # the plural - "What take many forms", "What were resolved on
        # January 2nd".  A modal carries no agreement, so "Who will be
        # asked to certify ..." is fine.
        next_index = subject_chunk.end - sentence.start
        if next_index < len(sentence) and root.tag_ in ('NNS', 'NNPS'):
            following = sentence[next_index]
            if following.pos_ in ('VERB', 'AUX') and following.tag_ != 'MD':
                return ''

        question = re.sub(r'\s+', ' ', f'{wh_word} {remainder}').strip()
        # Strip only the sentence's own full stop.  A blanket
        # rstrip('.!?') also ate the "!" in the brand name "Yum!".
        question = re.sub(r'\s*\.\s*$', '', question)
        if question.endswith(('!', '?')):
            return ''
        return question + '?'

    def build_definition_question(self, candidate):
        '''Turn "The Code is <answer>." into "What is the Code?".

        Applies when the answer is the complement of a copular verb.
        '''
        sentence = candidate.sentence
        root = candidate.span.root

        if root.dep_ != 'attr':
            return ''

        verb = root.head
        if verb.lemma_.lower() != 'be':
            return ''

        subject = None
        for child in verb.children:
            if child.dep_ in ('nsubj', 'nsubjpass'):
                subject = child
                break
        if subject is None or subject.pos_ == 'PRON':
            return ''

        subject_text = ''
        for chunk in sentence.noun_chunks:
            if chunk.root == subject:
                subject_text = chunk.text.strip()
                break
        if not subject_text or len(subject_text.split()) > 6:
            return ''

        copula = 'are' if verb.tag_ == 'VBP' else 'is'
        return f'What {copula} {subject_text}?'

    def build_mcq_stem(self, candidate, cloze_builder, allow_cloze=True):
        '''Pick the best available frame for this candidate.

        Returns (stem, form) where form is one of 'subject',
        'definition' or 'cloze'; ('', '') when nothing applies.
        '''
        stem = self.build_subject_question(candidate)
        if stem and self.is_valid_question(stem):
            return stem, 'subject'

        stem = self.build_definition_question(candidate)
        if stem and self.is_valid_question(stem):
            return stem, 'definition'

        if allow_cloze:
            stem = cloze_builder(candidate)
            if stem:
                return stem, 'cloze'

        return '', ''

    @staticmethod
    def is_valid_question(question):
        '''Structural check for an interrogative stem.'''
        words = question.split()
        if not 5 <= len(words) <= 30:
            return False
        if not (question.endswith('?') or BLANK in question):
            return False
        if '  ' in question:
            return False

        lowered = question.lower()
        valid_starters = (
            'how', 'what', 'who', 'where', 'when', 'why',
            'which', 'is', 'are', 'can', 'do', 'does',
        )
        if not lowered.startswith(valid_starters):
            return False
        # Repeated word ("can can", "the the") means a botched splice.
        if re.search(r'\b(\w+)\s+\1\b', lowered):
            return False
        # A stem ending on a preposition or connective is a fragment.
        if re.search(r'\b(of|to|for|and|or|with|the|a|an|in|on|at)\s*\?$',
                     lowered):
            return False
        return True

    def _add_multiple_choice(self, used_sentences, start):
        '''Fill the remaining slots with multiple-choice items.'''
        # Reuse the cloze builder so both generators blank identically.
        cloze_source = QuestionExtractor(self.num_questions)
        cloze_source.doc = self.doc
        cloze_source.sentences = self.sentences

        counter = start
        used_keys = []

        # Two passes.  The first takes only candidates that support a
        # genuine interrogative, so this generator produces questions the
        # blank-based one cannot; the second fills any remaining slots
        # with cloze items rather than leaving the quiz short.
        for allow_cloze in (False, True):
            for candidate in self.candidates:
                if counter > self.num_questions:
                    break

                key = candidate.sentence.text.strip()
                if key in used_sentences:
                    continue
                if is_redundant_with(candidate.text, used_keys):
                    continue

                stem, form = self.build_mcq_stem(
                    candidate, cloze_source.build_cloze, allow_cloze
                )
                if not stem:
                    continue

                used_sentences.add(key)
                used_keys.append(content_key(candidate.text))
                self.assessment_dict[counter] = {
                    'type': 'Multiple-Choice',
                    'question': stem,
                    'answer': candidate.text,
                    'answer_label': candidate.label,
                    'form': form,
                    'source_sentence': key,
                }
                counter += 1
