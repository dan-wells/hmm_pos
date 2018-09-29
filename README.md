**Implementing basic HMM framework and learning a POS tagger from Brown corpus**

This thing can learn transition and emission probabilities from labelled data 
using MLE, then decode test sequences. Training data should be in the same
format as NLTK tagged_sents() corpora, i.e. a list of sequences, each sequence
a list of (token, label) items.

**Example usage:**

```python
# assuming you have the Brown corpus downloaded in NLTK
from nltk.corpus import brown
import hmm

# use 10k sentences for training
train_data = brown.tagged_sents(tagset='universal')[:10000]
pos_tagger = hmm.Hmm()
pos_tagger.learn_mle_model(train_data)

# tag a new sentence
tagged_sentence = pos_tagger.decode("This is a test sentence .")

#   [(u'DET', 'This'),
#    (u'VERB', 'is'),
#    (u'DET', 'a'),
#    (u'NOUN', 'test'),
#    (u'NOUN', 'sentence'),
#    (u'.', '.')]
```
