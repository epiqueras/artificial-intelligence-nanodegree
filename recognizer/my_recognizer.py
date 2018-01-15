import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    for idx in range(0, test_set.num_items):
        test_x, test_lengths = test_set.get_item_Xlengths(idx)
        probs = {}
        for word in test_set.wordlist:
            if word in models:
                try:
                    probs[word] = models[word].score(test_x, test_lengths)
                except Exception:
                    probs[word] = float('-inf')
            else:
                probs[word] = float('-inf')
        probabilities.append(probs)
        guesses.append(max(probs, key=lambda key: probs[key]))

    return (probabilities, guesses)
