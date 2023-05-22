import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx


def compute_similarity_matrix(sentences, sentence_vectors):
    n = len(sentences)
    sim_mat = np.zeros((n, n))
    for row in range(n):
        for column in range(n):
            if row == column:
                continue

            first_sentence_embedding = np.array(sentence_vectors[row]).reshape(1, 100)
            second_sentence_embedding = np.array(sentence_vectors[column]).reshape(1, 100)
            similarity_score = cosine_similarity(first_sentence_embedding, second_sentence_embedding)
            sim_mat[row][column] = similarity_score[0, 0]

    return sim_mat


def compute_ranked_sentences(sentences, similiraity_mat):
    nx_graph = nx.from_numpy_array(similiraity_mat)
    try:
        scores = nx.pagerank(nx_graph)
        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
        return ranked_sentences
    except:
        return []


def get_summary(original_sentences, sentence_vectors, n=5):
    similiraity_mat = compute_similarity_matrix(original_sentences, sentence_vectors)
    ranked_sentences = compute_ranked_sentences(original_sentences, similiraity_mat)

    summary = []
    for i in range(n):
        if ranked_sentences:
            summary.append(ranked_sentences[i][1])
    return summary
