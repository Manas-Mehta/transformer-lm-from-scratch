import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


# Probability knobs for the transformation.
# Tuned to land a 5-10 point accuracy drop vs the original test set (full credit = >4 pt drop).
P_SYNONYM = 0.30
P_TYPO = 0.10

# QWERTY neighbors for typo injection. Only letters with well-defined adjacency.
QWERTY_NEIGHBORS = {
    'a': 'sqwz', 'b': 'vghn', 'c': 'xdfv', 'd': 'serfcx', 'e': 'wsdr',
    'f': 'drtgvc', 'g': 'ftyhbv', 'h': 'gyujnb', 'i': 'ujko', 'j': 'huikmn',
    'k': 'jiolm', 'l': 'kop', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp',
    'p': 'ol', 'q': 'wa', 'r': 'edft', 's': 'awedxz', 't': 'rfgy',
    'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu',
    'z': 'asx',
}

# Short, high-frequency function words that shouldn't be swapped - swapping "the"/"a"/"of"
# produces ungrammatical noise rather than a reasonable paraphrase.
STOPWORDS = {
    "a", "an", "the", "is", "am", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "by", "for", "with", "from", "as", "into",
    "and", "or", "but", "if", "not", "no", "so", "do", "does", "did", "has",
    "have", "had", "i", "you", "he", "she", "it", "we", "they", "me", "him",
    "her", "us", "them", "my", "your", "his", "its", "our", "their", "this",
    "that", "these", "those", "there", "here",
}

_detokenizer = TreebankWordDetokenizer()


def _inject_typo(word):
    # Pick a random lowercase letter position and swap it with a QWERTY neighbor.
    # Preserve original capitalization of the first letter if the original was capitalized.
    candidates = [i for i, ch in enumerate(word) if ch.lower() in QWERTY_NEIGHBORS]
    if not candidates:
        return word
    idx = random.choice(candidates)
    original_char = word[idx]
    neighbors = QWERTY_NEIGHBORS[original_char.lower()]
    new_char = random.choice(neighbors)
    if original_char.isupper():
        new_char = new_char.upper()
    return word[:idx] + new_char + word[idx + 1:]


def _get_synonym(word):
    # Return a WordNet synonym different from the input word, preserving leading capitalization.
    synsets = wordnet.synsets(word)
    if not synsets:
        return None
    lemmas = set()
    for syn in synsets:
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ")
            if name.lower() != word.lower() and name.isascii() and all(
                c.isalpha() or c == " " for c in name
            ):
                lemmas.add(name)
    if not lemmas:
        return None
    chosen = random.choice(sorted(lemmas))
    if word[0].isupper():
        chosen = chosen[0].upper() + chosen[1:]
    return chosen


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    tokens = word_tokenize(example["text"])
    new_tokens = []
    for tok in tokens:
        new_tok = tok
        is_word = tok.isalpha()
        lower = tok.lower()

        # Synonym replacement: content words only, length >= 4, not a stopword.
        if is_word and len(tok) >= 4 and lower not in STOPWORDS and random.random() < P_SYNONYM:
            syn = _get_synonym(tok)
            if syn is not None:
                new_tok = syn

        # Typo injection on the (possibly replaced) token. Keep typos modest — a whole
        # phrase from a synonym will count as one token here.
        if new_tok.isalpha() and len(new_tok) >= 3 and random.random() < P_TYPO:
            new_tok = _inject_typo(new_tok)

        new_tokens.append(new_tok)

    example["text"] = _detokenizer.detokenize(new_tokens)

    ##### YOUR CODE ENDS HERE ######

    return example
