from astrology.algorithms import sort


def test_bubble_sort():
    x = [4, 5, 1, 9, 10, 3]
    x_sorted = sort.bubble_sort(x, False)
    print(x_sorted)


def test_selection_sort():
    x = [4, 5, 1, 9, 10, 3]
    x_sorted = sort.selection_sort(x, True)
    print(x_sorted)


def test_insertion_sort():
    x = [12, 5, 7, 9, 10, 1]
    x_sorted = sort.insertion_sort(x, True)
    print(x_sorted)

test_insertion_sort()
