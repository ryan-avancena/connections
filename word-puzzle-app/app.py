from flask import Flask, render_template, jsonify
import gensim.downloader as api
import random
import re
import json

import nltk
from nltk.corpus import words as nltk_words
from nltk.stem import WordNetLemmatizer

# Load NLTK data
# nltk.download('words')
# nltk.download('wordnet')

# Load model
model = api.load("word2vec-google-news-300")

app = Flask(__name__)

valid_words = set(w.lower() for w in nltk_words.words())
lemmatizer = WordNetLemmatizer()

def is_clean_word(word):
    return (
        word.isalpha()
        and word.islower()
        and 3 <= len(word) <= 12
        and word in valid_words
        and not word.endswith("s")
        and word in model.key_to_index
    )

def avg_similarity(words):
    sims = []
    for i in range(len(words)):
        for j in range(i+1, len(words)):
            try:
                sims.append(model.similarity(words[i], words[j]))
            except KeyError:
                return 0
    return sum(sims) / len(sims) if sims else 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate')
def generate():
    themes = ["fruit", "beans", "french", "bread"]
    puzzle = {"words": [], "groups": []}
    used_lemmas = set()

    for theme in themes:
        try:
            similar = model.most_similar(theme, topn=50)
        except KeyError:
            continue

        filtered = []
        for word, _ in similar:
            word = word.lower()
            lemma = lemmatizer.lemmatize(word)
            if is_clean_word(word) and lemma not in used_lemmas:
                used_lemmas.add(lemma)
                filtered.append(word)

        if len(filtered) < 4:
            continue

        selected = random.sample(filtered, 4)

        if avg_similarity(selected) < 0.35:
            continue

        puzzle["words"].extend(selected)
        puzzle["groups"].append({"theme": theme, "words": selected})

    random.shuffle(puzzle["words"])
    return jsonify(puzzle)

if __name__ == '__main__':
    app.run(debug=True)
