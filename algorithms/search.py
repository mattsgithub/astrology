def binary_search(x, v, p=0, q=None):
    """Search a sorted list
    """
    if q is None:
        q = len(x) - 1

    if p <= q:
        i = (p + q) / 2
        if x[i] < v:
            return binary_search(x, v, i + 1, q)
            # Search right
        elif v < x[i]:
            # Search left
            return binary_search(x, v, p, i - 1)
        else:
            return True
    return False
