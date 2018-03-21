# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 20:28:53 2018

@author: ryuhei
"""

import gensim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from chainer.backends import cuda
from dataset_topics import generate_lda_corpus

np.random.seed(0)


def doc_to_words_counts(doc):
    bow = np.int64(doc)
    words = bow.T[0]
    counts = bow.T[1]
    return words, counts


def add_at(a, slices, value):
    xp = cuda.get_array_module(a)
    if xp is np:
        return np.add.at(a, slices, value)
    else:
        return xp.scatter_add(
            a, slices, value.astype(np.float32))


if __name__ == '__main__':
    corpus, true_topics = generate_lda_corpus()

    D = corpus.num_docs
    V = corpus.num_terms
    K = 10
    hp_alpha = 0.1
    hp_eta = 2

    use_gpu = True
    num_epochs = 2000
    batch_size = 50
    max_iter_local = 200  # max iteration for local optimization
    thresh_local_convergence = 0.001  # convergence threshold for local optim
    # learning rate \rho is scheduled as \rho_t = (t + \tau)^{-kappa}
    tau = 1.0
    kappa = 0.9

    if not use_gpu:
        xp = np
        from scipy.special import loggamma

        def digamma(x):
            eps = x * 1e-3
            return (loggamma(x + eps) - loggamma(x - eps)) / (eps + eps)

    else:
        import cupy
        xp = cupy
        digamma = cupy.ElementwiseKernel(
                'T x', 'T y',
                '''
                T eps = x * 1e-3;
                y = (lgamma(x + eps) - lgamma(x - eps)) / (eps + eps);
                ''',
                'elementwise_digamma',
            )

    docs = list(corpus)

    # Initialize lambda according to footnote 6
    p_lambda = np.random.exponential(D * 100 / float(K * V), (K, V)) + hp_eta
    p_lambda = xp.asarray(p_lambda, np.float32)

    # Step 3
    t = 0
    rho = 1
    ppl_history = []
    for epoch in range(1, num_epochs + 1):
        print('epoch', epoch)
        print('rho =', rho)
        ppls = []
        perm = np.random.permutation(D)
        num_batches = D // batch_size
        indexes = np.array_split(perm, num_batches)
        for batch in tqdm(indexes, total=num_batches):
            t += 1
            rho = (t + tau) ** -kappa  # learning rate

            # Step 5-9
            digamma_lambda = digamma(p_lambda)
            digamma_sum_lambda = digamma(p_lambda.sum(1))[:, None, None]

            B = len(batch)  # actual size of this mini-batch
            lengths = [len(docs[d]) for d in batch]
            max_length = max(lengths)
            words = np.zeros((B, max_length), np.int64)
            counts = np.zeros((B, max_length), np.float32)
            for i, d in enumerate(batch):
                words_d, counts_d = doc_to_words_counts(docs[d])
                length = len(words_d)
                words[i, :length] = words_d
                counts[i, :length] = counts_d
            xp_counts = xp.asarray(counts)

            # Step 4
            p_gamma = xp.asarray(
                np.random.gamma(100, 0.01, (K, B)), np.float32)

            for ite in range(max_iter_local):
                p_gamma_prev = p_gamma
                digamma_gamma = digamma(p_gamma)
                digamma_sum_gamma = digamma(p_gamma.sum(0))
                e_log_theta = digamma_gamma - digamma_sum_gamma[None]
                e_log_beta = digamma_lambda[:, words] - digamma_sum_lambda
                exponent = e_log_theta[..., None] + e_log_beta
                p_phi = xp.exp(exponent)
                p_phi /= p_phi.sum(0, keepdims=True)

                p_gamma = hp_alpha + np.sum(p_phi * xp_counts[None], -1)
                mean_diff = xp.abs(p_gamma_prev - p_gamma).mean(0).max()
                if mean_diff < thresh_local_convergence:
                    break

            # Step 10
            lambda_hat = xp.zeros_like(p_lambda)
            add_at(lambda_hat, (Ellipsis, words), p_phi * xp_counts[None])
            lambda_hat *= D / batch_size
            lambda_hat += hp_eta

            # Step 11
            p_lambda = (1 - rho) * p_lambda + rho * lambda_hat

            # Rough evaluation
            e_beta = p_lambda / p_lambda.sum(1, keepdims=True)
#            ppl = np.average(-np.log(np.sum(p_phi * e_beta[:, words], 0)),
#                             weights=counts)
            ppl = np.average(cuda.to_cpu(
                -xp.log(xp.sum(p_phi[:, -1] * e_beta[:, words[-1]], 0))),
                             weights=counts[-1])
            ppls.append(ppl)

        epoch_ppl = cuda.to_cpu(np.average(ppls))
        print('Perplexity:', epoch_ppl)
        ppl_history.append(epoch_ppl)
        plt.plot(ppl_history)
        plt.grid()
        plt.show()

        topics = p_lambda / p_lambda.sum(1, keepdims=True)
        topics = cuda.to_cpu(topics)
        word_ranks = [[corpus.id2word[w] for w in np.argsort(topic)[::-1]]
                      for topic in cuda.to_cpu(topics)]
        for k, word_ranks_k in enumerate(word_ranks):
            print('{:2d} {}'.format(k, word_ranks_k[:5]))

        # Visualize topics
        L = K // 2
        for k, topic in enumerate(topics):
            plt.subplot(2, K // 2, k + 1)
            plt.imshow(topic.reshape(int(np.sqrt(V)), -1), vmin=0, vmax=1)
            plt.axis('off')
        plt.show()
