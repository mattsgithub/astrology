"""Implementation of sorting algorithms
"""


def _compare(x1, x2, ascending):
    return x1 > x2 if ascending else x2 > x1


def _merge(x, ascending, p, q, r):

    # Create copied list for left
    left = x[p:q + 1]

    # Create copied list for right
    right = x[q + 1:r + 1]

    # Add 'bottom cards'
    bottom_card = float("inf") if ascending else float("-inf")
    left += [bottom_card]
    right += [bottom_card]

    # Now, these two lists are
    # sorted piles. Keep pulling from either
    # until piles are empty. Replace current list
    # with most recently pulled card
    i = 0
    j = 0
    for k in range(p, r + 1):
        if _compare(right[j], left[i], ascending):
            x[k] = left[i]
            # Update count for left
            # since card was 'taken'
            i += 1
        else:
            x[k] = right[j]
            # Update count for right
            # since card was 'taken'
            j += 1


def merge_sort(x, ascending=True, p=0, r=None):
    if r is None:
        r = len(x) - 1

    # Keep splitting
    # as long as this
    # condition holds true
    if p < r:
        q = (p+r)/2
        merge_sort(x, ascending, p, q)
        merge_sort(x, ascending, q + 1, r)
        _merge(x, ascending, p, q, r)


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
