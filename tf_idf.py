import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
import re
import math
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

def preprocess(words, remove_special=False, remove_stopwords=False, lemmatize=False):
    pattern = r'[^a-zA-Z0-9\s]'
    processed_words = []
    wordlemmatizer = WordNetLemmatizer()
    Stopwords = set(stopwords.words('english'))
    words = [w.lower() for w in words]
    for w in words:
        if remove_special:
            w = re.sub(pattern, '', w)
            w = re.sub(r'\d+', '', w)
        if remove_stopwords:
            if w in Stopwords or len(w) < 2:
                continue
        if lemmatize:
            processed_words.append(wordlemmatizer.lemmatize(w))
    return processed_words


def noun_and_verbs(text):
    pos_tagged = nltk.pos_tag(text)
    nouns_verbs = []
    for word, tag in pos_tagged:
        if tag == "NN" or tag == "NNS" or tag == "NNP" or tag == "NNPS" or tag == "VBZ" or tag == "VBP" or tag == "VBN" or tag == "VBG" or tag == "VBD" or tag == "VB":
            nouns_verbs.append(word)
    return nouns_verbs


def words_frequency(words):
    words = [w.lower() for w in words]
    freq = Counter(words)
    return dict(freq)


def tf(word, sentence):
    word_frequency = 0
    len_sentence = len(sentence)
    for word_sentence in sentence:
        if word == word_sentence:
            word_frequency +=  1
    tf = word_frequency / len_sentence
    return tf


def idf(word, sentences):
    n_sentences = len(sentences)
    sentences_with_word = 0
    for s in sentences:
        s = preprocess(s.split(), remove_special=True, remove_stopwords=True, lemmatize=True)
        if word in s:
            sentences_with_word += 1
    idf = math.log10(n_sentences / sentences_with_word)
    return idf


def word_score(word, s, sentences):
    tf_ = tf(word, s)
    idf_ = idf(word, sentences)
    return tf_ * idf_


def sentence_score(s, sentences):
    pattern = r'[^a-zA-Z0-9\s]'
    wordlemmatizer = WordNetLemmatizer()
    score = 0
    s = re.sub(pattern, '', s)
    s = re.sub(r'\d+', '', s)
    s = noun_and_verbs(s.split())
    Stopwords = set(stopwords.words('english'))
    for w in s:
        if w.lower() not in Stopwords and len(w) > 1:
            w = wordlemmatizer.lemmatize(w.lower())
            score += word_score(w, s, sentences)
    return score


def get_summary(text, n):
    n_retained_sentences = n
    sentences_scores = {}
    i = 0
    for s in text:
        score = sentence_score(s, text)
        sentences_scores[i] = score
        i += 1

    sentences_scores = sorted(sentences_scores.items(), key=lambda x: x[1], reverse=True)
    index_used_sentences = []
    i = 0
    for s in sentences_scores:
        if i < n_retained_sentences:
            index_used_sentences.append(s[0])
            i += 1
        else:
            break

    index_used_sentences = sorted(index_used_sentences)
    summary = []
    for i in index_used_sentences:
        summary.append(text[i])
    print(" ".join(summary))
    return summary
