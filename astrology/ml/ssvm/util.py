def print_metrics(error_matrix):
    
    # Number of examples that are wrong
    n_example_error = (error_matrix.sum(axis=1) > 0).astype(int).sum()
    example_error_rate = n_example_error / error_matrix.shape[0]

    n_label_errors = error_matrix.sum()
    n_attempts = error_matrix.size
    label_error_rate = n_label_errors / n_attempts
    # These metrics also treat each label as equal value. But do we want this?
    # 
    
    # Zero-One Loss (Very strict!)
    print('Example Error Rate: {} ({} / {})' % (example_error_rate, n_example_error, error_matrix.shape[0]))
    
    # Hamming Loss
    print('Label Error Rate: {} ({} / {})' % (label_error_rate, n_label_errors, n_attempts))


def get_batch_indices(n_examples,
                      batch_size=32):
                      
    indices = np.arange(n_examples)
    np.random.shuffle(indices)

    if batch_size < 1:
        raise ValueError("Batch size cannot be less than 1. "
                         "You provided a batch size of {}" % batch_size)
        
    if batch_size > n_examples:
        raise ValueError("Batch size cannot be greater than n_examples. "
                         "You gave batch_size={} "
                         "and n_examples={}" % (batch_size, n_examples))

    i = 0
    batches = []
    while True:

        indx = indices[i:i + batch_size]

        if len(indx) == 0:
            break

        batches.append(indx)

        i = i + batch_size
        
    return batches