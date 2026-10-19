'''Shared NLP model instances.

Loaded once at import time and reused across requests instead of being
reloaded (spaCy) or resolved over the network (SentenceTransformer) on
every question-generation call.

Both loaders degrade gracefully so the project can be cloned and run
without the large local artefacts checked in:
  * spaCy  : ``en_core_web_md`` is preferred (it carries word vectors),
             ``en_core_web_sm`` is used when the md model is absent.
  * SBERT  : ``./models/all-MiniLM-L6-v2`` is used when present, otherwise
             the model is resolved from the HuggingFace hub / local cache.
'''
import os

import spacy
from sentence_transformers import SentenceTransformer

_SPACY_PREFERENCE = ('en_core_web_md', 'en_core_web_sm')

_LOCAL_SBERT_PATH = './models/all-MiniLM-L6-v2'
_HUB_SBERT_NAME = 'sentence-transformers/all-MiniLM-L6-v2'


def _load_spacy():
    last_error = None
    for model_name in _SPACY_PREFERENCE:
        try:
            return spacy.load(model_name)
        except OSError as exc:  # model not installed
            last_error = exc
    raise OSError(
        'No spaCy English model found. Install one with:\n'
        '    python -m spacy download en_core_web_md'
    ) from last_error


def _load_sbert():
    if os.path.isdir(_LOCAL_SBERT_PATH):
        return SentenceTransformer(_LOCAL_SBERT_PATH)
    return SentenceTransformer(_HUB_SBERT_NAME)


nlp = _load_spacy()
semantic_model = _load_sbert()

# True when the loaded pipeline ships word vectors (en_core_web_md and up).
HAS_VECTORS = nlp.meta.get('vectors', {}).get('vectors', 0) > 0
