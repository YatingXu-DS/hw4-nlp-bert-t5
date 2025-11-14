import datasets
from datasets.load import load_dataset
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

def get_synonym(word):

    synsets = wordnet.synsets(word)
    if not synsets:
        return word
    lemmas = synsets[0].lemmas()
    if not lemmas:
        return word
    synonym = lemmas[0].name().replace("_", " ")
    return synonym if synonym.lower() != word.lower() else word

def introduce_typo(word):
    if len(word) <= 3:
        return word
    if random.random() < 0.5:
        i = random.randint(0, len(word) - 2)
        return word[:i] + word[i+1] + word[i] + word[i+2:]
    else:
        letters = "abcdefghijklmnopqrstuvwxyz"
        i = random.randint(1, len(word) - 2)
        c = random.choice(letters)
        return word[:i] + c + word[i:]


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    text = example["text"]
    words = word_tokenize(text)
    new_words = []

    for w in words:
        if w.isalpha():
            r = random.random()
            if r < 0.3:                      
                new_words.append(get_synonym(w))
            elif r < 0.45:                    
                new_words.append(introduce_typo(w))
            else:
                new_words.append(w)
        else:
            new_words.append(w)


    detok = TreebankWordDetokenizer()
    transformed_text = detok.detokenize(new_words)
    example["text"] = transformed_text

    ##### YOUR CODE ENDS HERE ######

    return example
