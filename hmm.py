# Basic HMM framework
# TODO:
#   - Handle start/end states separately, as non-emitting states
#   - Probably commit to bigram models only and clean up indexing
#   - Work out how the backpointers actually work in decode

from collections import defaultdict
from math import log


class Hmm(object):
    """
    Hidden Markov model with state list, transition and emission probabilities

    Use learn_mle_model() to calculate transition and emission probabilities
    from labelled data and populate state list.

    Once trained:
        transition_probs[q_t][q_t-1] gives p(q_t|q_t-1)
        emissions_probs[q_t][o_t] gives p(o_t|q_t)
    """

    def __init__(self, states=None, transition_probs=None, emission_probs=None):
        if states is None:
            states = []
        if transition_probs is None:
            transition_probs = {}
        if emission_probs is None:
            emission_probs = {}
        self.states = states
        self.transition_probs = transition_probs
        self.emission_probs = emission_probs

    def learn_mle_model(self, train_data, n=1):
        """
        Learn transition and emissions probabilities from training data.

        Training data should be a list of training sequences (e.g. sentences),
        each one a list of tuples (token, label). Calculate transition and
        emission probabilities from counts using MLE. Transition probabilities
        can be calculated based on n previous states.
        """
        # for emission probabilities per label
        tokens_per_label = defaultdict(lambda: defaultdict(int))
        # for label sequences
        label_ngram = defaultdict(lambda: defaultdict(int))
        label_prev = defaultdict(int)

        for train_seq in train_data:
            # add start/end symbols (need n times for higher-order ngrams)
            seq = [('<s>', '<s>')]*n + train_seq + [('</s>', '</s>')]*n
            # want to look behind at each step:
            #   p(w_i|w_i-1) = c(w_i-1,w_i) / c(w_i-1)
            for i, (token, label) in enumerate(seq[n:]):
                token = token.lower()
                tokens_per_label[label][token] += 1
                # zip(*iterable) is the inverse of zip(iter1, iter2)
                prev_tokens, prev_labels = zip(*seq[i:i+n])
                # we use these counts for mle
                label_ngram[label][prev_labels] += 1
                label_prev[prev_labels] += 1

        # convert token/label counts to (log) prob
        # these are emission probabilities
        for label, tokens in tokens_per_label.items():
            total_tokens = float(sum(tokens.values()))
            for token, count in tokens.items():
                # tokens[token] = count / total_tokens
                tokens[token] = log(count / total_tokens)
        # calculate mle ngram probs from sequence counts
        for label, ngram_hists in label_ngram.items():
            for context, count in ngram_hists.items():
                context_total = float(label_prev[context])
                # label_ngram[label][context] = count / context_total
                label_ngram[label][context] = log(count / context_total)

        # switch defaultdict values to -1000 for log(0) prob on missing tokens when decoding
        transition_probs = dict(zip(label_ngram.keys(), [defaultdict(lambda: -1000, v) for v in label_ngram.values()]))
        emission_probs = dict(zip(tokens_per_label.keys(), [defaultdict(lambda: -1000, v) for v in tokens_per_label.values()]))
        # assign model components
        for label in tokens_per_label:
            self.states.append(label)
            self.transition_probs[label] = transition_probs[label]
            self.emission_probs[label] = emission_probs[label]
            # self.transition_probs[label] = label_ngram[label]
            # self.emission_probs[label] = tokens_per_label[label]

    def decode(self, test_seq, states=None, t_probs=None, e_probs=None, n=1):
        """Use Viterbi search to decode test sequences input as strings"""
        # don't try and generalise yet
        if n != 1:
            raise NotImplementedError("Sorry, only works with bigrams for now")

        # get learned model properties
        if states is None:
            states = self.states
        if t_probs is None:
            t_probs = self.transition_probs
        if e_probs is None:
            e_probs = self.emission_probs

        # process input sequence
        # bit dumb for now, split on whitespace and lowercase
        seq = [i.lower() for i in test_seq.split()]
        seq = ['<s>']*n + seq + ['</s>']*n

        # need to track viterbi prob per state plus backpointer to prev state
        viterbi_probs = [{} for i in range(len(seq))]
        backpointers = [{} for i in range(len(seq))]

        # initialize: transitions from start state and emissions of first token
        for q in states:
            # viterbi_probs[0][q] = t_probs[q][('<s>',)] * e_probs[q][seq[n]]
            viterbi_probs[0][q] = t_probs[q][('<s>',)] + e_probs[q][seq[n]]
            backpointers[0][q] = ('<s>',)

        # recursive calculations over rest of input sequence
        for t, o in enumerate(seq[n:], n):
            for q in states:
                v_by_t = {}
                for q_prev in states:
                    # v_by_t[q_prev] = viterbi_probs[t-1][q_prev] * t_probs[q][(q_prev,)]
                    v_by_t[q_prev] = viterbi_probs[t-1][q_prev] + t_probs[q][(q_prev,)]
                backpointers[t][q] = argmax_dict(v_by_t)
                viterbi_probs[t][q] = max(v_by_t.values()) + e_probs[q][o]

        # then wrap up and follow backpointers
        # skip emission on </s> (which has 0.0 log prob = certain)
        backpointers.reverse()
        viterbi_probs.reverse()
        q_seq = [backpointers[t][argmax_dict(i)] for t, i in enumerate(viterbi_probs[n:-n])]
        # q_seq = [backpointers[t][argmax_dict(i)] for t,i in enumerate(viterbi_probs)]
        q_seq.reverse()

        # return (seq, q_seq, viterbi_probs, backpointers)
        return zip(q_seq, test_seq.split())


# Some utility functions

def argmax_dict(mydict):
    """Return key from dict with max value"""
    return max(mydict, key=lambda x: mydict[x])


def argmax_list(mylist):
    """Return (first) index of list item with max value"""
    return mylist.index(max(mylist))
