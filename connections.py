import gensim.downloader as api
import random
import json
import re

# --- NLP tools ---
import nltk
# nltk.download('words')
# nltk.download('wordnet')
from nltk.corpus import words as nltk_words
from nltk.stem import WordNetLemmatizer

# --- Load model ---
model = api.load("word2vec-google-news-300")

# --- Setup ---
valid_words = set(w.lower() for w in nltk_words.words())
lemmatizer = WordNetLemmatizer()

def is_clean_word(word):
    return (
        word.isalpha()
        and word.islower()
        and 3 <= len(word) <= 12
        and word in valid_words
        and not word.endswith("s")  # optional: remove plurals
        and word in model.key_to_index  # valid in Word2Vec
    )

def avg_similarity(words):
    sims = []
    for i in range(len(words)):
        for j in range(i+1, len(words)):
            try:
                sims.append(model.similarity(words[i], words[j]))
            except KeyError:
                return 0  # if one word isn't in vocab
    return sum(sims) / len(sims) if sims else 0

# --- Generate puzzle ---
themes = ["fruit", "beans", "french","bread"]
puzzle = {"words": [], "groups": []}
used_words = set()
used_lemmas = set()

for theme in themes:
    try:
        similar = model.most_similar(theme, topn=50)
    except KeyError:
        continue  # skip if theme isn't in vocab

    # Clean and filter similar words
    filtered = []
    for word, _ in similar:
        word = word.lower()
        lemma = lemmatizer.lemmatize(word)
        if is_clean_word(word) and lemma not in used_lemmas:
            used_lemmas.add(lemma)
            filtered.append(word)

    if len(filtered) < 4:
        continue  # not enough valid words

    selected = random.sample(filtered, 4)

    if avg_similarity(selected) < 0.35:
        continue  # discard weak group

    puzzle["words"].extend(selected)
    puzzle["groups"].append({"theme": theme, "words": selected})
    used_words.update(selected)

print("Words:")
print(" | ".join(puzzle["words"]))
input("\nFind the 4 groups of 4. Press Enter to reveal the answers...\n")

print(json.dumps(puzzle, indent=2))
