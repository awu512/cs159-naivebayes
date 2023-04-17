import math
import sys
from collections import defaultdict

TOTAL = '<TOTAL>'

class TrainExample:
    def __init__(self, nb_class, sentence):
        self.nb_class = nb_class
        self.sentence = sentence

def read_train_data(path):
    ''' 
    Reads a file from path into a list of TrainExamples.
    Returns the TrainExamples and the priors for each seen class,
    '''
    with open(path, 'r') as f:
        lines = f.readlines()

    examples = []
    counts = defaultdict(int)
    total = 0
    for e in lines:
        nb_class, raw = e.split('\t')

        examples.append(TrainExample(nb_class, raw.split()))
        counts[nb_class] += 1
        total += 1

    priors = defaultdict(float)
    for cla, count in counts.items():
        priors[cla] = math.log10(count / total)

    return examples, priors

def read_test_data(path):
    ''' 
    Reads a file from path into a list of TrainExamples.
    If no labels are provided, the nb_class field will be None
    '''
    with open(path, 'r') as f:
        lines = f.readlines()

    examples = []
    for e in lines:
        spl = e.split('\t')

        if len(spl) > 1:
            examples.append(TrainExample(spl[0], spl[-1].split()))
        else:
            examples.append(TrainExample(None, spl[0].split()))

    return examples

def word_counts_by_class(train_data, priors):
    ''' Iterates a list of TrainExamples and counts word occurences by class '''
    wc = { c: defaultdict(int) for c in priors.keys() }

    for ex in train_data:
        for w in ex.sentence:
            wc[ex.nb_class][w] += 1
            wc[ex.nb_class][TOTAL] += 1

    return wc

def classify(ex, wc, priors, l):
    ''' Classifies a sentence '''
    vsize = len(wc.keys())

    argmax = (None, float('-inf'))

    for c in wc.keys():
        p = priors[c]

        for w in ex.sentence:
            feature_p = (wc[c][w] + l) / (wc[c][TOTAL] + (vsize * l))
            if feature_p > 0: # ???
                p += math.log10(feature_p)

        if p > argmax[1] or (p == argmax[1] and c == 'positive'):
            argmax = (c, p)

    return argmax

def classify_all(examples, wc, priors, l):
    ''' Classifies a list of sentences '''
    results = []

    for ex in examples:
        results.append(classify(ex, wc, priors, l))

    return results

def evaluate(examples, results):
    assert len(examples) == len(results), 'Mismatched lens for examples and results'
    right = 0
    for ex, res in zip(examples, results):
        if ex.nb_class == res[0]:
            right += 1

    return f'{right}/{len(examples)} | {round(100*(right/len(examples)),2)}%'

def output_results(results):
    for ex in results:
        print(f'{ex[0]}\t{ex[1]}')

def main(train_path, test_path, l):
    '''
    Takes a set of training data, a set of test sentences and a value for lambda (l).
    Trains a Multinomial Naive Bayes classifier and outputs the results on the test sentences.
    '''
    train_data, priors = read_train_data(train_path)
    test_data = read_test_data(test_path)

    wc = word_counts_by_class(train_data, priors)

    results = classify_all(test_data, wc, priors, l)

    output_results(results)

if __name__ == '__main__':
    assert len(sys.argv) == 4, f'expected 3 arguments, received {len(sys.argv)-1}'

    train_path, test_path, l = sys.argv[1:4]

    main(train_path, test_path, float(l))