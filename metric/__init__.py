from astrology.util import mean, std

def cross_validate(corpus, algorithm, K=8):

    # Get training examples
    training_examples = corpus.get_training_examples()
    N = len(training_examples)
    
    # Split into k groups
    n = int(N/(K*1.0))

    groups = [(k*n, (k+1)*n) for k in xrange(0,K)]

    accs = []

    for g in groups:
        algorithm.reset()

        i = g[0]
        j = g[1]

        train = training_examples[0:i] + training_examples[j:]
        test = training_examples[i:j]

        for example in train:
            algorithm.observe(example.text, example.label)
        
        for example in test:
            pred = algorithm.predict(example.text)
            correct = 1 if pred == example.label else 0
        acc = correct / (len(test)*1.0)
        accs.append(acc)

    print mean(accs)
    print std(accs)
