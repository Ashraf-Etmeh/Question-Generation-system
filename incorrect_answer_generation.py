'''Generation of incorrect alternatives (distractors) for an answer key.

A distractor has to sit in a narrow band to be any good:

  * too close to the answer and the question has two correct options
    ("Material Inside Information" / "Material Nonpublic Information",
    "The United States" / "The Us");
  * too far from it and the answer is given away by elimination
    ("The United States" / "France" / "A Country");
  * differently formatted and the answer is given away by sight alone -
    three Title Case options and one lower-case one is a free point.

So candidates are filtered on surface form (alias, acronym, plural,
containment), on type (same entity label, same numeric-ness, comparable
length) and finally on cosine similarity inside an explicit band, before
being spread out so the distractors also differ from each other.
'''
import random
import re

import torch
from nltk.corpus import wordnet as wn
from sentence_transformers import util

from candidate_selection import normalise_phrase, strip_determiner
from nlp_models import nlp, semantic_model

# Below this the option is obviously wrong and helps by elimination.
MIN_SIMILARITY = 0.30
# Above this the option paraphrases the answer - the question has two
# defensible answers and is unfair.
MAX_SIMILARITY = 0.78
# Distractors must also differ from one another by at least this much.
MAX_MUTUAL_SIMILARITY = 0.82

# Suffix rules good enough to tell "policy"/"policies" apart from a real
# difference in meaning.  A full lemmatiser is not worth a spaCy pass per
# candidate comparison, and nlp.tokenizer() does not populate lemmas.
_PLURAL_RULES = (
    ('ies', 'y'), ('ches', 'ch'), ('shes', 'sh'), ('sses', 'ss'),
    ('xes', 'x'), ('ves', 'f'), ('es', ''), ('s', ''),
)


def _stem_word(word):
    for suffix, replacement in _PLURAL_RULES:
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[:-len(suffix)] + replacement
    return word


def _stem_set(phrase):
    '''Inflection-insensitive set of the words in a phrase.'''
    return {_stem_word(word) for word in phrase.split() if word}


class IncorrectAnswerGenerator:
    '''Builds the option list for a multiple-choice question.'''

    # Class-level caches shared across instances and requests.
    _embedding_cache_shared = {}
    _wordnet_cache_shared = {}

    def __init__(self, document, doc=None, candidates=None):
        self.document = document
        self.nlp = nlp
        self.ranker_model = semantic_model
        self.doc = doc if doc is not None else self.nlp(document)

        # Candidates vetted by candidate_selection are far better raw
        # material than every noun chunk in the document.
        self.candidate_pool = self._build_pool(candidates)

        self._embedding_cache = IncorrectAnswerGenerator._embedding_cache_shared
        self._wordnet_cache = IncorrectAnswerGenerator._wordnet_cache_shared

    # ------------------------------------------------------------ #
    # Candidate pools
    # ------------------------------------------------------------ #

    def _build_pool(self, candidates):
        '''Return [(surface_text, entity_label)] drawn from the document.'''
        pool = {}

        if candidates:
            for candidate in candidates:
                pool.setdefault(normalise_phrase(candidate.text),
                                (candidate.text, candidate.label))
        else:
            for ent in self.doc.ents:
                text = ent.text.strip()
                if 1 <= len(text.split()) <= 4:
                    pool.setdefault(normalise_phrase(text), (text, ent.label_))
            for chunk in self.doc.noun_chunks:
                text = strip_determiner(chunk.text)
                if 1 <= len(text.split()) <= 4:
                    pool.setdefault(normalise_phrase(text), (text, ''))

        pool.pop('', None)
        return list(pool.values())

    def get_wordnet_candidates(self, answer):
        '''Co-hyponyms of the answer - siblings under a shared hypernym.'''
        cache_key = answer.strip().lower()
        if cache_key in self._wordnet_cache:
            return self._wordnet_cache[cache_key]

        found = set()
        lookup = answer.lower().replace(' ', '_')

        # Only single-word or short answers have useful WordNet coverage.
        if len(answer.split()) <= 2:
            for synset in wn.synsets(lookup)[:3]:
                for hypernym in synset.hypernyms()[:3]:
                    for hyponym in hypernym.hyponyms()[:12]:
                        for lemma in hyponym.lemmas():
                            name = lemma.name().replace('_', ' ').strip()
                            if name.lower() != answer.lower():
                                found.add(name)

        result = list(found)
        self._wordnet_cache[cache_key] = result
        return result

    # ------------------------------------------------------------ #
    # Surface-form filters
    # ------------------------------------------------------------ #

    @staticmethod
    def _acronym(phrase):
        words = [w for w in re.split(r'\W+', phrase) if w]
        if len(words) < 2:
            return ''
        return ''.join(word[0] for word in words).lower()

    def _is_alias(self, candidate, answer):
        '''True when the two strings denote the same thing.

        Catches "The Us" against "The United States" (acronym), "years"
        against "The Year" (plural) and any containment pair.
        '''
        cand_norm = normalise_phrase(candidate)
        ans_norm = normalise_phrase(answer)

        if not cand_norm or not ans_norm:
            return True
        if cand_norm == ans_norm:
            return True
        if cand_norm in ans_norm or ans_norm in cand_norm:
            return True

        # Acronym in either direction: "us" vs "united states".
        flat_cand = cand_norm.replace(' ', '')
        flat_ans = ans_norm.replace(' ', '')
        if flat_cand and flat_cand == self._acronym(ans_norm):
            return True
        if flat_ans and flat_ans == self._acronym(cand_norm):
            return True

        # Same content words up to inflection: "year" vs "years".
        if _stem_set(cand_norm) == _stem_set(ans_norm):
            return True

        # The answer's head - its last two words - reappearing inside the
        # candidate means the two phrases name the same thing seen from
        # different angles: "Law Department's approval" against
        # "Yum! Law Department".
        ans_words = ans_norm.split()
        if len(ans_words) >= 2:
            head = ' '.join(ans_words[-2:])
            if head in cand_norm:
                return True

        return False

    @staticmethod
    def _shape_matches(candidate, answer):
        '''Reject options whose shape betrays which one is the answer.'''
        cand_words = candidate.split()
        ans_words = answer.split()

        # Comparable length: no one-word option next to a four-word answer.
        if abs(len(cand_words) - len(ans_words)) > 2:
            return False
        if not 0.4 <= len(candidate) / max(len(answer), 1) <= 2.5:
            return False

        # A number must be matched by a number.
        cand_has_digit = any(c.isdigit() for c in candidate)
        ans_has_digit = any(c.isdigit() for c in answer)
        if cand_has_digit != ans_has_digit:
            return False

        return True

    def filter_candidates(self, pool, answer, answer_label):
        '''Apply every surface and type filter to the raw pool.

        Params:
            * pool         : list of (text, label) tuples
            * answer       : the answer key, in its document surface form
            * answer_label : entity label of the answer ('' when none)
        Returns:
            * list<str> of surviving candidate texts
        '''
        kept = []
        seen = set()

        for text, label in pool:
            text = text.strip()
            if not text:
                continue

            # A typed answer needs typed distractors: a GPE answer must
            # not be offered against a DATE.
            if answer_label and label and label != answer_label:
                continue

            if self._is_alias(text, answer):
                continue
            if not self._shape_matches(text, answer):
                continue

            key = normalise_phrase(text)
            if key in seen:
                continue
            seen.add(key)
            kept.append(text)

        return kept

    # ------------------------------------------------------------ #
    # Semantic ranking
    # ------------------------------------------------------------ #

    def get_cached_embeddings(self, texts):
        uncached = [t for t in texts if t not in self._embedding_cache]
        if uncached:
            new_embeddings = self.ranker_model.encode(
                uncached, convert_to_tensor=True
            )
            for text, embedding in zip(uncached, new_embeddings):
                self._embedding_cache[text] = embedding
        return torch.stack([self._embedding_cache[t] for t in texts])

    def rank_candidates(self, candidates, answer):
        '''Keep candidates inside the similarity band, best first.

        Returns (texts, embeddings) where "best" means closest to the
        upper edge of the band - plausible without being a paraphrase.
        '''
        if not candidates:
            return [], None

        answer_embedding = self.get_cached_embeddings([answer])[0]
        candidate_embeddings = self.get_cached_embeddings(candidates)
        scores = util.cos_sim(candidate_embeddings, answer_embedding).squeeze(1)

        in_band = [
            index for index in range(len(candidates))
            if MIN_SIMILARITY <= scores[index].item() <= MAX_SIMILARITY
        ]
        # Nothing in band: fall back to whatever is not a paraphrase,
        # rather than returning an empty option list.
        if not in_band:
            in_band = [
                index for index in range(len(candidates))
                if scores[index].item() <= MAX_SIMILARITY
            ]

        in_band.sort(key=lambda index: scores[index].item(), reverse=True)
        return (
            [candidates[index] for index in in_band],
            candidate_embeddings[in_band],
        )

    def diversify_candidates(self, ranked, embeddings, num_needed):
        '''Pick options that differ from the answer *and* from each other.'''
        selected = []
        selected_indices = []

        if embeddings is None:
            return selected

        for index, candidate in enumerate(ranked):
            if len(selected) >= num_needed:
                break

            if selected_indices:
                similarities = util.cos_sim(
                    embeddings[index], embeddings[selected_indices]
                )[0]
                if similarities.max().item() > MAX_MUTUAL_SIMILARITY:
                    continue
                if any(self._is_alias(candidate, chosen)
                       for chosen in selected):
                    continue

            selected.append(candidate)
            selected_indices.append(index)

        return selected

    # ------------------------------------------------------------ #
    # Presentation
    # ------------------------------------------------------------ #

    @staticmethod
    def _match_case(option, answer):
        '''Align an option's case with the answer's, without mangling it.

        Mixed casing across options is a visual give-away (three Title
        Case options next to one lower-case one), but blanket re-casing
        is worse: .title() turned "Yum!'s Internal Audit Department" into
        "Yum!'S Internal Audit Department".  Options drawn from the
        document already carry the right case, so only all-lower-case
        options - the WordNet ones - are adjusted.
        '''
        if not option or option != option.lower():
            return option
        if answer.isupper():
            return option.upper()
        if answer[:1].isupper():
            return option[:1].upper() + option[1:]
        return option

    def get_all_options_dict(self, answer, num_options, answer_label='',
                             stem=''):
        '''Return {1: option, ...} containing the answer and distractors.

        Params:
            * answer       : the answer key in its document surface form
            * num_options  : total number of options wanted
            * answer_label : entity label of the answer, when known
            * stem         : the question text, so options already
                             visible in it can be excluded
        Returns:
            * dict<int, str>
        '''
        answer = answer.strip()
        num_needed = max(num_options - 1, 0)

        pool = list(self.candidate_pool)
        pool += [(text, answer_label)
                 for text in self.get_wordnet_candidates(answer)]

        filtered = self.filter_candidates(pool, answer, answer_label)

        # An option printed in the stem is not a real alternative - the
        # reader can see it belongs elsewhere in the sentence.
        if stem:
            stem_normalised = normalise_phrase(stem)
            filtered = [
                text for text in filtered
                if normalise_phrase(text) not in stem_normalised
            ]

        ranked, embeddings = self.rank_candidates(filtered, answer)
        distractors = self.diversify_candidates(ranked, embeddings, num_needed)

        # Top up from the ranked list if diversification was too strict.
        for candidate in ranked:
            if len(distractors) >= num_needed:
                break
            if candidate not in distractors:
                distractors.append(candidate)

        distractors = [self._match_case(d, answer) for d in distractors]
        distractors = [d for d in distractors
                       if normalise_phrase(d) != normalise_phrase(answer)]

        options = distractors[:num_needed] + [answer]
        random.shuffle(options)

        return {index: option for index, option in enumerate(options, 1)}
