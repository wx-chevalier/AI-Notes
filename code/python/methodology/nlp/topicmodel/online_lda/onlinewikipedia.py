# onlinewikipedia.py: Demonstrates the use of online VB for LDA to
# analyze a bunch of random Wikipedia articles.
#
# Copyright (C) 2010  Matthew D. Hoffman
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import cPickle, string, numpy, getopt, sys, random, time, re, pprint

import onlineldavb
import wikirandom

def main():
    """
    Downloads and analyzes a bunch of random Wikipedia articles using
    online VB for LDA.
    """

    # The number of documents to analyze each iteration
    batchsize = 64
    # The total number of documents in Wikipedia
    D = 3.3e6
    # The number of topics
    K = 100

    # How many documents to look at
    if (len(sys.argv) < 2):
        documentstoanalyze = int(D/batchsize)
    else:
        documentstoanalyze = int(sys.argv[1])

    # Our vocabulary
    vocab = file('./dictnostops.txt').readlines()
    W = len(vocab)

    # Initialize the algorithm with alpha=1/K, eta=1/K, tau_0=1024, kappa=0.7
    olda = onlineldavb.OnlineLDA(vocab, K, D, 1./K, 1./K, 1024., 0.7)
    # Run until we've seen D documents. (Feel free to interrupt *much*
    # sooner than this.)
    for iteration in range(0, documentstoanalyze):
        # Download some articles
        (docset, articlenames) = \
            wikirandom.get_random_wikipedia_articles(batchsize)
        # Give them to online LDA
        (gamma, bound) = olda.update_lambda_docs(docset)
        # Compute an estimate of held-out perplexity
        (wordids, wordcts) = onlineldavb.parse_doc_list(docset, olda._vocab)
        perwordbound = bound * len(docset) / (D * sum(map(sum, wordcts)))
        print '%d:  rho_t = %f,  held-out perplexity estimate = %f' % \
            (iteration, olda._rhot, numpy.exp(-perwordbound))

        # Save lambda, the parameters to the variational distributions
        # over topics, and gamma, the parameters to the variational
        # distributions over topic weights for the articles analyzed in
        # the last iteration.
        if (iteration % 10 == 0):
            numpy.savetxt('lambda-%d.dat' % iteration, olda._lambda)
            numpy.savetxt('gamma-%d.dat' % iteration, gamma)

if __name__ == '__main__':
    main()
