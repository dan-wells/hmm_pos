# Basic HMM framework
# States have probabilistic transition and emission tables
# Models have state maps with transitions inherited from states

from collections import defaultdict
from math import log


class Hmm(object):
    """Hidden Markov model collecting a set of State objects"""

    def __init__(self, states=None, label_priors=None):
        if states is None:
            states = {}
        self.states = states
        self.label_priors = label_priors

    def show_labels(self):
        #if len(self.states.keys()) == 0:
        try:
            print(self.states.keys())
        except(AttributeError):
            print("State list has not been populated yet! Try learn_emissions()")

    def show_tokens(self, label):
        print(self.emissions[label])

    def learn_model(self, train_data, n=1):
        """
        Learn emission and transition probabilities from training data.

        Training data should be a list of training sequences (e.g. sentences),
        each one a list of tuples (token, label). Create a State object for
        each label in the data and calculate emission and transition
        probabilities for each State. Transition probabilities are calculated
        taking n previous states into account. Easy to learn both in a single
        pass over the dataset.
        """
        # we take [[(token,label),...],...] to read nltk brown corpus easily

        # for emission probabilities per label
        tokens_per_label = defaultdict(lambda: defaultdict(int))
        # for label sequences
        label_ngram = defaultdict(lambda: defaultdict(int))
        label_prev = defaultdict(int)
        # for token sequences -- don't care about these cos they factor out
        #token_ngram = defaultdict(lambda: defaultdict(int))
        #token_prev = defaultdict(int)

        for train_seq in train_data:
            # add start/end symbols (need n times for higher-order ngrams)
            seq = [('<s>', '<s>')]*n + train_seq + [('</s>', '</s>')]*n

            # want to look behind at each step:
            #   p(w_i|w_i-1) = c(w_i-1,w_i) / c(w_i-1)

            for i, (token, label) in enumerate(seq[n:]):
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
                tokens[token] = count / total_tokens
                #tokens[token] = log(count / total_tokens)

        # calculate mle ngram probs from sequence counts
        for label, ngram_hists in label_ngram.items():
            for context, count in ngram_hists.items():
                context_total = float(label_prev[context])
                label_ngram[label][context] = count / context_total
                #label_ngram[label][context] = log(count / context_total)

        # actually we don't care about token_ngram: this factors out when calculating
        # most likely tag sequence of many over a single string of tokens
        #for token, ngram_hists in token_ngram.items():
        #    for context, count in ngram_hists.items():
        #        context_total = float(token_prev[context])
        #        token_ngram[token][context] = count / context_total
        #        #token_ngram[token][context] = log(count / context_total)

        # create State instance for each label in dataset and assign probabilities
        for label in tokens_per_label:
            self.states[label] = State(label=label, emissions=tokens_per_label[label], transitions=label_ngram[label])
        # p(t|l) = emissions is right. label_ngram gives p(l), token_ngram gives p(t)
        # neither of those ngrams really belongs in a State, they are priors on the whole model
        # -> possibly don't need a State class after all lol

    def decode(self, test_data):
        """Use Viterbi search to decode test sequences"""
        pass

    def learn_emissions(self, labelled_tokens):
        """
        Calculate p(token|label) from a list of tuples (token,label)

        Take labelled data and derive the list of state labels and emission
        probabilities for tokens from each state. Populate state list for HMM
        if not already defined.
        """
        tokens_per_label = defaultdict(lambda: defaultdict(int))
        for token, label in labelled_tokens:
            tokens_per_label[label][token] += 1
        # convert raw counts to probabilities
        for label, tokens in tokens_per_label.items():
            total_tokens = float(sum(tokens.values()))
            for token, count in tokens.items():
                tokens[token] = count / total_tokens
            # create or update State instance for each label in dataset
            if not self.states[label]:
                self.states[label] = State(label=label, emissions=tokens)
            else:
                self.states[label].emissions = tokens

    def learn_transitions(self, labelled_tokens):
        """
        Calculate p(l_i|l_i-1) from a list of tuples (token,label)
        """
        pass


class State(object):
    """Single state in an HMM with emission and transition probabilities"""

    def __init__(self, label=None, transitions=None, emissions=None):
        self.label = label
        self.transitions = transitions
        self.emissions = emissions

    def show_transitions(self):
        print(self.transitions)

    def show_emissions(self):
        print(self.emissions)
