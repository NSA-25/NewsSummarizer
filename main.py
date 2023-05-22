import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
import text_rank
import k_means
import tf_idf
import nltk
from gensim.models import Word2Vec
import csv
from rouge import Rouge
nltk.download('punkt')
nltk.download('stopwords')
# CNN dataset https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail
# glove.6B.100d.txt: https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip


def get_sentences(file_name, nr):
    ls = []
    with open(file_name, encoding="utf8") as f:
        file = csv.reader(f)
        file = list(file)[1:nr]
        for row in file:
            text = sent_tokenize(row[1])
            highlight = sent_tokenize(row[2])
            ls.append((text, highlight))

    return ls


def get_clean_sentences(sentences):
    stop_words = stopwords.words('english')
    lines = [line.lower() for line in sentences]
    lines = [line.translate(str.maketrans('', '', string.punctuation)) for line in lines]
    lines = [word_tokenize(line) for line in lines]
    lines = [[word for word in line if word not in stop_words] for line in
                       lines]
    lemmatizer = WordNetLemmatizer()
    clean_sentences = [" ".join([lemmatizer.lemmatize(token) for token in line]) for line in lines]
    return clean_sentences


def glove(clean_sentences):
    glove_model = {}
    with open('glove.6B.100d.txt', encoding='utf-8') as file:
        for line in file:
            split_line = line.split()
            word = split_line[0]
            embedding = np.asarray(split_line[1:], dtype='float32')
            glove_model[word] = embedding

    sentence_vectors = []
    for sentence in clean_sentences:
        vectors = []
        for token in sentence.split():
            if token in glove_model:
                vectors.append(glove_model[token])
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            sentence_vectors.append(avg_vec)
        else:
            sentence_vectors.append(np.zeros((100,)))

    return sentence_vectors


def word_2_vec(clean_sentences):
    all_words = [i.split() for i in clean_sentences]
    w2v_model = Word2Vec(all_words)
    sentence_vectors = []
    for sentence in clean_sentences:
        zero_vector = np.zeros(w2v_model.vector_size)
        vectors = []
        for token in sentence.split():
            if token in w2v_model.wv:
                vectors.append(w2v_model.wv[token])
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            sentence_vectors.append(avg_vec)
        else:
            sentence_vectors.append(zero_vector)

    return sentence_vectors


def rouge(predicted_summary, original_summary):
    rouge = Rouge()
    scores = rouge.get_scores(predicted_summary, original_summary)
    print(scores)

    return scores


if __name__ == "__main__":
    text_path = 'cnn_dailymail.csv'
    file_number = 5
    texts = get_sentences(text_path, file_number + 1)
    for article in texts:
        original_sentences = article[0]
        original_summary = article[1]
        clean_sentences = get_clean_sentences(original_sentences)
        sentence_vectors = word_2_vec(clean_sentences)

        n = len(original_summary) + 1
        tf_idf_summary = tf_idf.get_summary(original_sentences, n)
        k_means_summary = k_means.get_summary(original_sentences, sentence_vectors, n=n)
        text_rank_summary = text_rank.get_summary(original_sentences, sentence_vectors, n=n)

        # for sentence in tf_idf_summary:
        #     print(sentence)
        print('TF-IDF Score: ', end='')
        rouge(" ".join(tf_idf_summary), " ".join(original_summary))

        # for sentence in k_means_summary:
        #     print(sentence)
        print('K-means Score: ', end='')
        rouge(" ".join(k_means_summary), " ".join(original_summary))

        # for sentence in text_rank_summary:
            # print(sentence)
        print('TextRank Score: ', end='')
        if text_rank_summary:
            rouge(" ".join(text_rank_summary), " ".join(original_summary))

        print('-------------------')
