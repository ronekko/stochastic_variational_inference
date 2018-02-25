# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 22:21:22 2018

@author: ryuhei
"""

import random

import gensim
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import digamma
from tqdm import tqdm


def doc_to_tokens(doc):
    return sum([[w[0]] * int(w[1]) for w in doc], [])


if __name__ == '__main__':
    corpus = gensim.corpora.UciCorpus('kos/docword.kos.txt',
                                      'kos/vocab.kos.txt')
    D = corpus.num_docs
    V = corpus.num_terms
    K = 20
#    hp_alpha = 0.1 / K
#    hp_eta = 0.001
    hp_alpha = 1 / K
    hp_eta = 0.1
    # learning rate \rho is scheduled as \rho_t = (t + \tau)^{-kappa}
    num_epochs = 100
    batch_size = 50
    tau = 1.0
    kappa = 0.9

    docs = list(corpus)

    # Initialize lambda according to footnote 6
    p_lambda = np.random.exponential(D * 100 / float(K * V), (K, V)) + hp_eta

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
            lambda_hat = np.zeros_like(p_lambda)

            # Step 5-9
            digamma_lambda = digamma(p_lambda)
            digamma_sum_lambda = digamma(p_lambda.sum(1, keepdims=True))
            for d in batch:
                # Step 4
                x = doc_to_tokens(docs[d])

                p_gamma = np.ones(K, float)
                for ite in range(15):
                    digamma_gamma = digamma(p_gamma)
                    digamma_sum_gamma = digamma(p_gamma.sum())
                    e_log_theta = digamma_gamma - digamma_sum_gamma
                    # Without for loop below
                    e_log_beta = digamma_lambda[:, x] - digamma_sum_lambda
                    exponent = e_log_theta[:, None] + e_log_beta
                    exponent -= exponent.max(0, keepdims=True)
                    p_phi = np.exp(exponent)
                    p_phi /= p_phi.sum(0, keepdims=True)

                    p_gamma = hp_alpha + np.sum(p_phi, 1)
        #            print(p_gamma)

                # Step 10
                for w, p_phi_n in zip(x, p_phi.T):
                    lambda_hat[:, w] += p_phi_n

            lambda_hat *= D / batch_size
            lambda_hat += hp_eta

            # Step 11
            p_lambda = (1 - rho) * p_lambda + rho * lambda_hat

            # Rough evaluation
#            e_theta = p_gamma / p_gamma.sum()
            e_beta = p_lambda / p_lambda.sum(1, keepdims=True)
            ppl = np.average(-np.log(np.sum(p_phi * e_beta[:, x], 0)))
#            ppl = -np.log(e_theta.dot(e_beta[:, x])).sum()
            ppls.append(ppl)
#            print('rho =', rho)
#            print('ppl =', ppl)

        epoch_ppl = np.average(ppls)
        print(epoch_ppl)
        ppl_history.append(epoch_ppl)
#        ppl_history += ppls
        plt.plot(ppl_history)
        plt.show()

        topics = p_lambda / p_lambda.sum(1, keepdims=True)
        word_ranks = [[corpus.id2word[w] for w in np.argsort(topic)[::-1]]
                  for topic in topics]
        for k, word_ranks_k in enumerate(word_ranks):
            print('{:2d} {}'.format(k, word_ranks_k[:5]))
