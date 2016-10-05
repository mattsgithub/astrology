"""Implementation of sorting algorithms
"""


def _compare(x1, x2, ascending):
    return x1 > x2 if ascending else x2 > x1


def merge_sort(x):
    """TODO"""
    pass


def bubble_sort(x, ascending=True):
    """O(N^2)
    """
    k = len(x)
    while k > 0:
        for i in xrange(1, k):
            if _compare(x[i-1], x[i], ascending):
                # swap
                val = x[i]
                x[i] = x[i-1]
                x[i-1] = val

        k -= 1
    return x


def insertion_sort(x, ascending=True):
    """O(N^2)
    """
    for i in xrange(1, len(x)):
        val = x[i]
        j = i - 1
        while j >= 0 and _compare(x[j], val, ascending):
            x[j + 1] = x[j]
            j -= 1
        x[j+1] = val
    return x


def selection_sort(x, ascending=True):
    """O(N^2)
    """

    for i in xrange(len(x)):
        index = i
        for j in xrange(i+1, len(x)):
            if _compare(x[index], x[j], ascending):
                index = j
        # Swap places
        val = x[index]
        x[index] = x[i]
        x[i] = val
    return x
