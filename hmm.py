# Basic HMM framework
# States have probabilistic transition and emission tables
# Models have state maps with transitions inherited from states (fully-connected) ??

from collections import defaultdict

class State:
    def __init__(self, transitions={}, emissions={}):
        self.transitions = transitions
        self.emissions = emissions

    def show_transitions(self):
        print(self.transitions)

    def show_emissions(self):
        print(self.emissions)

    def show_labels(self):
        print(self.emissions.keys())

    def show_tokens(self, label):
        print(self.emissions[label])
    
    def learn_emissions(self, labelled_tokens):
        """
        Calculate p(token|label) from a list of tuples (token,label)
        """
        tokens_per_label = defaultdict(lambda: defaultdict(int))
        for token,label in labelled_tokens:
            tokens_per_label[label][token] += 1
        # convert raw counts to probabilities
        for label in tokens_per_label:
            total_tokens = float(sum(tokens_per_label[label].values()))
            for token,count in tokens_per_label[label].items():
                tokens_per_label[label][token] = count/total_tokens
        self.emissions = tokens_per_label

    def learn_transitions(self, labelled_tokens):
        """
        Calculate p(l_i|l_i-1) from a list of tuples (token,label)
        """
        pass



