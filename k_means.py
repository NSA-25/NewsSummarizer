from sklearn.cluster import KMeans
from scipy.spatial import distance


def get_summary(sentences, sentence_vectors, n=5):
    kmeans = KMeans(n)
    y_kmeans = kmeans.fit_predict(sentence_vectors)

    my_list = []
    for i in range(n):
        my_dict = {}
        for j in range(len(y_kmeans)):
            if y_kmeans[j] == i:
                my_dict[j] = distance.euclidean(kmeans.cluster_centers_[i], sentence_vectors[j])
        try:
            my_list.append(min(my_dict, key=my_dict.get))
        except:
            pass

    summary = []
    for i in sorted(my_list):
        summary.append(sentences[i])
    return summary
