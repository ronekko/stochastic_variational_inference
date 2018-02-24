# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 13:15:08 2018

@author: ryuhei
"""

import gensim
import numpy as np
import matplotlib.pyplot as plt


def generate_lda_corpus(num_docs=1000, num_tokens=100, alpha=0.1,
                        beta=0.01, num_topics=10, size_vocaburary=25,
                        return_topics=True):
    D = num_docs
    N = num_tokens
    K = num_topics
    V = size_vocaburary

    # Create artificial topics
    L = K // 2
    topics = np.zeros((K, L, L))
    for i in range(L):
        topic = topics[i]
        topic[i] = 1.0 / L
    for j in range(L):
        topic = topics[L + j]
        topic[:, j] = 1.0 / L
    for topic in topics:
        topic[:] += beta
        topic[:] /= topic.sum()
    topics = topics.reshape(K, V)

    # Generate documents
    thetas = np.random.dirichlet(np.full(K, alpha), D)
    docs = []
    for theta in thetas:
        z = np.random.choice(K, N, p=theta)
        n_z = np.bincount(z, minlength=K)
        x = np.empty((0,), int)
        for k, n_z_k in enumerate(n_z):
            x_from_k = np.random.choice(V, n_z_k, p=topics[k])
            x = np.append(x, x_from_k)
        np.random.shuffle(x)
        docs.append(x)

    # Create bag-of-words
    bow = []
    for doc in docs:
        counts = np.bincount(doc, minlength=V)
        bow.append(counts)
    bow = np.array(bow)

    corpus = gensim.matutils.Dense2Corpus(bow.T)
    corpus.num_docs = D
    corpus.num_nnz = D * N
    corpus.num_terms = V
    corpus.id2word = dict((i, i) for i in range(V))

    if return_topics:
        return corpus, topics
    else:
        return corpus


if __name__ == '__main__':
    num_docs=1000
    num_tokens=100
    alpha=0.1
    beta=0.01
    K = 10

    corpus, topics = generate_lda_corpus(num_docs, num_tokens, alpha, beta)
    V = corpus.num_terms

    # Visualize topics
    L = K // 2
    for k, topic in enumerate(topics):
        plt.subplot(2, K // 2, k + 1)
        plt.imshow(topic.reshape(int(np.sqrt(V)), -1), vmin=0, vmax=1)
        plt.axis('off')
    plt.show()