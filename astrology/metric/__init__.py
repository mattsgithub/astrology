from astrology.util import mean, std


def validate(corpus, algorithm):
    training_examples = corpus.get_training_examples()
    test_examples = corpus.get_test_examples() 

    algorithm.reset()
    for example in training_examples:
        algorithm.observe(example.text, example.label)


    correct = 0
    for example in test_examples:
        pred = algorithm.predict(example.text)
        correct += 1 if pred == example.label else 0
    print correct / (len(test_examples)*1.0)


def cross_validate(corpus, algorithm, K=8):

    # Get training examples
    training_examples = corpus.get_all_examples()
    N = len(training_examples)
    
    # Split into k groups
    n = int(N/(K*1.0))

    groups = [(k*n, (k+1)*n) for k in xrange(0,K)]

    accs = []

    for g in groups:
        algorithm.reset()
        correct = 0

        i = g[0]
        j = g[1]

        train = training_examples[0:i] + training_examples[j:]
        test = training_examples[i:j]

        for example in train:
            algorithm.observe(example.text, example.label)
        
        for example in test:
            pred = algorithm.predict(example.text)
            correct += 1 if pred == example.label else 0
        acc = correct / (len(test)*1.0)
        accs.append(acc)

    print len(groups)
    print mean(accs)
    print std(accs)
