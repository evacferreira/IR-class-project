import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from unidecode import unidecode
import string
from enum import Enum

# Required NLTK resources download
nltk.download('punkt_tab', quiet=True) # for tokenization
nltk.download('stopwords', quiet=True) # for filtering common words in PT/EN
nltk.download('wordnet', quiet=True) # for lemmatization logic
nltk.download('omw-1.4', quiet=True) # for lemmatization logic


class ReductionMode(str, Enum):
    """
    Controls which lexical reduction strategy is applied during preprocessing.

    - STEMMING:      Heuristic-based root extraction (faster, less precise).
    - LEMMATIZATION: Dictionary-based canonical form (slower, more accurate).
    - BOTH:          Lemmatization followed by stemming (default legacy behaviour).
    - NONE:          No reduction applied; tokens are kept as-is after filtering.
    """
    STEMMING = "stemming"
    LEMMATIZATION = "lemmatization"
    BOTH = "both"
    NONE = "none"


def preprocess(
    text: str,
    reduction_mode: ReductionMode = ReductionMode.BOTH,
    remove_stopwords: bool = True,
) -> list[str]:
    """
    Performs full text preprocessing for Bilingual Information Retrieval.
    Includes: Normalization, Tokenization, optional Stopword Removal,
    and configurable Lemmatization / Stemming.

    Args:
        text:
            Raw input string to process.
        reduction_mode:
            Lexical reduction strategy to apply. One of:
              - ReductionMode.STEMMING      – Porter Stemmer only.
              - ReductionMode.LEMMATIZATION – WordNet Lemmatizer only.
              - ReductionMode.BOTH          – Lemmatize then stem (default).
              - ReductionMode.NONE          – Keep tokens unchanged.
        remove_stopwords:
            When True (default) bilingual PT+EN stop words are filtered out.
            Set to False to retain stop words in the output token list.

    Returns:
        List of processed tokens.
    """
    # Validation: Ensure input is a valid non-empty string
    if not text or not isinstance(text, str):
        return []

    # Normalization & ASCII conversion
    # Lowercase text and remove accents/diacritics
    text = unidecode(text.lower())

    # Tokenization
    # Splits the string into individual words (tokens)
    tokens = word_tokenize(text)

    # Bilingual Stopword Removal (REQ-B20)
    # Combines Portuguese and English stopword lists to support bilingual datasets.
    # Only built when removal is requested, to avoid unnecessary computation.
    if remove_stopwords:
        stop_words = set(stopwords.words('portuguese')).union(set(stopwords.words('english')))
        # Normalize stopwords to match our lowercased/unidecoded tokens
        stop_words = {unidecode(sw) for sw in stop_words}
    else:
        stop_words = set()

    # Initialize Linguistic Reducers (REQ-B18)
    # Instantiated lazily based on the requested reduction_mode.
    stemmer = PorterStemmer() if reduction_mode in (ReductionMode.STEMMING, ReductionMode.BOTH) else None
    lemmatizer = WordNetLemmatizer() if reduction_mode in (ReductionMode.LEMMATIZATION, ReductionMode.BOTH) else None

    filtered_tokens = []
    for w in tokens:
        # Remove stopwords (if enabled), punctuation marks, and very short tokens
        if w in stop_words or w in string.punctuation or len(w) <= 2:
            continue

        # REQ-B18 — Apply the configured reduction strategy
        if reduction_mode == ReductionMode.BOTH:
            # Lemmatize first, then stem the result
            w = lemmatizer.lemmatize(w)
            w = stemmer.stem(w)

        elif reduction_mode == ReductionMode.LEMMATIZATION:
            # Dictionary-based canonical form only
            w = lemmatizer.lemmatize(w)

        elif reduction_mode == ReductionMode.STEMMING:
            # Heuristic root extraction only
            w = stemmer.stem(w)

        # ReductionMode.NONE: token is kept as-is

        filtered_tokens.append(w)

    return filtered_tokens