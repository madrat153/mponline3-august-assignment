"""
text_utils.py -- Module B1 deliverable: text preprocessing.

Dependency-light (no NLTK/spaCy corpus downloads required): lowercasing,
punctuation/number stripping, stopword removal, tokenization, and a light
suffix-stripping stemmer as a stand-in for full lemmatization.
"""
import re
import string

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "to", "of", "in",
    "on", "for", "with", "at", "by", "from", "up", "about", "into", "over",
    "after", "is", "are", "was", "were", "be", "been", "being", "am", "i",
    "you", "he", "she", "it", "we", "they", "this", "that", "these", "those",
    "my", "your", "his", "her", "its", "our", "their", "as", "do", "does",
    "did", "have", "has", "had", "not", "no", "will", "would", "can", "could",
    "should", "just", "very", "there", "here", "than", "too", "also", "s",
    "t", "m", "re", "ve", "ll", "d",
}

_SUFFIXES = ("ing", "edly", "ed", "ly", "es", "s")


def light_stem(word: str) -> str:
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    return text.split()


def remove_stopwords(tokens):
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def preprocess(text: str, stem: bool = True) -> str:
    cleaned = clean_text(text)
    tokens = remove_stopwords(tokenize(cleaned))
    if stem:
        tokens = [light_stem(t) for t in tokens]
    return " ".join(tokens)
