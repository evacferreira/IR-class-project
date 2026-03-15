import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from unidecode import unidecode
import string

# Required NLTK resources download
nltk.download('punkt_tab', quiet=True) # for tokenization
nltk.download('stopwords', quiet=True) # for filtering common words in PT/EN
nltk.download('wordnet', quiet=True) # for lemmatization logic
nltk.download('omw-1.4', quiet=True) # for lemmatization logic

def preprocess(text):
    """
    Performs full text preprocessing for Bilingual Information Retrieval.
    Includes: Normalization, Tokenization, Stopword Removal, Lemmatization, and Stemming.
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

    # Bilingual Stopword Removal
    # Combines Portuguese and English stopword lists to support bilingual datasets
    stop_words = set(stopwords.words('portuguese')).union(set(stopwords.words('english')))
    # Normalize stopwords to match our lowercased/unidecoded tokens
    stop_words = {unidecode(sw) for sw in stop_words}

    # Initialize Linguistic Reducers
    # Stemmer: Reduces words to their root form (faster, heuristic-based)
    # Lemmatizer: Reduces words to their dictionary base form (more accurate)
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    filtered_tokens = []
    for w in tokens:
        # Remove stopwords, punctuation marks, and very short tokens 
        if w not in stop_words and w not in string.punctuation and len(w) > 2:
            
            # Lemmatization
            # Returns the canonical form of the word
            lemma = lemmatizer.lemmatize(w)
            
            # Stemming
            # Trims the word to its structural root
            stem = stemmer.stem(lemma)
            
            filtered_tokens.append(stem)

    return filtered_tokens