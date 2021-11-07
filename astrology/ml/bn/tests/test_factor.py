import pytest

import numpy as np

from pgm.factor import Factor
from pgm.factor import log
from pgm.factor import exp
from pgm.factor import get_marg
from pgm.factor import get_mappings
from pgm.factor import is_equal
from pgm.factor import get_stride
from pgm.factor import get_product_from_list
from pgm.factor import get_product
from pgm.factor import assign_to_indx
from pgm.factor import indx_to_assign

'''
python -m pytest -vv -s test_factor.py | less
'''


def test_get_mappings_same_array_but_out_of_order():
    x = np.array([6, 4, 3])
    y = np.array([4, 3, 6])
    mapX, mapY = get_mappings(x, y)
    assert np.array_equal(x[mapX], y[mapY])


@pytest.fixture(scope='module')
def f1():
    '''
       f1
    [1   phi]
     --------
     0   0.11
     1   0.89
    '''
    return Factor(scope=np.array([1]),
                  card=np.array([2]),
                  val=np.array([0.11, 0.89]))


@pytest.fixture(scope='module')
def f2():
    '''
       f2
    [2  1  phi]
    ------------
    [0  0  0.59]
    [1  0  0.41]
    [0  1  0.22]
    [1  1  0.78]
    '''
    return Factor(scope=np.array([2, 1]),
                  card=np.array([2, 2]),
                  val=np.array([0.59, 0.41, 0.22, 0.78]))


@pytest.fixture(scope='module')
def f3():
    '''
       f3
    [3  2  phi]
    ------------
    [0  0  0.39]
    [1  0  0.61]
    [0  1  0.06]
    [1  1  0.94]
    '''
    return Factor(scope=np.array([3, 2]),
                  card=np.array([2, 2]),
                  val=np.array([0.39, 0.61, 0.06, 0.94]))


@pytest.fixture(scope='module')
def f4():
    '''
       f4
    [4  5  phi]
    ------------
    [0  0  0.39]
    [1  0  0.61]
    [0  1  0.06]
    [1  1  0.94]
    '''
    return Factor(scope=np.array([4, 5]),
                  card=np.array([2, 2]),
                  val=np.array([0.39, 0.61, 0.06, 0.94]))


def test_assign_to_indx():
    '''
       f
    [1  2  phi]
    ------------
    [0  0  0]
    [1  0  1]
    [0  1  2]
    [1  1  3]
    '''
    C = np.array([2, 2])
    A = np.array([[1, 1],
                  [1, 0],
                  [0, 1],
                  [0, 0]])

    I = assign_to_indx(A, C)
    assert np.array_equal(I, np.array([3, 1, 2, 0]))

    A = np.array([[1, 1]])
    I = assign_to_indx(A, C)
    assert np.array_equal(I, np.array([3]))

    A = np.array([[1, 1],
                  [0, 0],
                  [1, 1]])
    I = assign_to_indx(A, C)
    assert np.array_equal(I, np.array([3, 0, 3]))


def test_indx_to_assign():
    C = np.array([2, 2])
    A_gold = np.array([[1, 1],
                       [1, 0],
                       [0, 1],
                       [0, 0]])
    I = np.array([3, 1, 2, 0])
    A_test = indx_to_assign(I, C)
    assert np.allclose(A_test, A_gold)

    A_gold = np.array([[0, 0],
                       [0, 1]])

    I = np.array([0, 2])
    A_test = indx_to_assign(I, C)
    assert np.allclose(A_test, A_gold)


def test_self_factor_product(f1, f2):
    f = get_product(f1, f1)
    f = get_product(f, f1)
    f = get_product(f, f1)
    f = get_product(f, f1)
    assert np.allclose(f.scope, np.array([1]))
    assert np.allclose(f.card, np.array([2]))
    assert np.allclose(f.val, np.array([0.11 ** 5, 0.89 ** 5]))


def test_get_stride():
    f = get_stride(np.array([2, 2]))
    assert np.allclose(f, np.array([1, 2]))

    f = get_stride(np.array([2, 2, 2]))
    assert np.allclose(f, np.array([1, 2, 4]))

    f = get_stride(np.array([2, 4, 2]))
    assert np.allclose(f, np.array([1, 2, 8]))


def test_f3_f4_product(f3, f4):
    f = get_product(f3, f4)
    assert np.allclose(f.scope, np.array([2, 3, 4, 5]))
    assert np.allclose(f.card, np.array([2, 2, 2, 2]))


def test_f1_f2_product_log_space(f1, f2):
    f = get_product(log(f1), log(f2), use_log_space=True)
    f = exp(f)
    assert np.allclose(f.scope, np.array([1, 2]))
    assert np.allclose(f.card, np.array([2, 2]))
    assert np.allclose(f.val, np.array([0.11 * 0.59,
                                        0.89 * 0.22,
                                        0.11 * 0.41,
                                        0.89 * 0.78]))


def test_f1_f2_product(f1, f2):
    '''
       f2
    [2  1  phi]
    ------------
    [0  0  0.59]
    [1  0  0.41]
    [0  1  0.22]
    [1  1  0.78]

       f1
    [1   phi]
    ---------
    [0   0.11]
    [1   0.89]


    f1 * f2
    [1  2  phi]
    -----------
    [0  0  0.11 * 0.59]
    [1  0  0.89 * 0.22]
    [0  1  0.11 * 0.41]
    [1  1  0.89 * 0.78]
    '''

    f = get_product(f1, f2)
    assert np.allclose(f.scope, np.array([1, 2]))
    assert np.allclose(f.card, np.array([2, 2]))
    assert np.allclose(f.val, np.array([0.11 * 0.59,
                                        0.89 * 0.22,
                                        0.11 * 0.41,
                                        0.89 * 0.78]))


def test_f1_f3_product(f1, f3):
    '''
       f1
    [1   phi]
    ---------
    [0   0.11]
    [1   0.89]

       f3
    [3  2  phi]
    ------------
    [0  0  0.39]
    [1  0  0.61]
    [0  1  0.06]
    [1  1  0.94]

    f1 * f3
    [1  2  3  phi]
    --------------
    [0  0  0  0.11 * 0.39]
    [1  0  0  0.89 * 0.39]
    [0  1  0  0.11 * 0.06]
    [1  1  0  0.89 * 0.06]
    [1  0  1  0.11 * 0.61]
    [1  0  1  0.89 * 0.61]
    [1  1  1  0.11 * 0.94]
    [1  1  1  0.89 * 0.94]
    '''

    f = get_product(f1, f3)
    assert np.allclose(f.scope, np.array([1, 2, 3]))
    assert np.allclose(f.card, np.array([2, 2, 2]))
    assert np.allclose(f.val, np.array([0.11 * 0.39,
                                        0.89 * 0.39,
                                        0.11 * 0.06,
                                        0.89 * 0.06,
                                        0.11 * 0.61,
                                        0.89 * 0.61,
                                        0.11 * 0.94,
                                        0.89 * 0.94]))


def test_get_product_from_list(f1, f2):
    f = get_product_from_list([f1, f2])
    assert np.allclose(f.scope, np.array([1, 2]))
    assert np.allclose(f.card, np.array([2, 2]))
    assert np.allclose(f.val, np.array([0.11 * 0.59,
                                        0.89 * 0.22,
                                        0.11 * 0.41,
                                        0.89 * 0.78]))


def test_get_product_from_list_log_space(f1, f2):
    f = get_product_from_list([log(f1), log(f2)], use_log_space=True)
    f = exp(f)
    assert np.allclose(f.scope, np.array([1, 2]))
    assert np.allclose(f.card, np.array([2, 2]))
    assert np.allclose(f.val, np.array([0.11 * 0.59,
                                        0.89 * 0.22,
                                        0.11 * 0.41,
                                        0.89 * 0.78]))


def test_f1_f2_product_commuative(f1, f2):
    '''
       f2
    [2  1  phi]
    ------------
    [0  0  0.59]
    [1  0  0.41]
    [0  1  0.22]
    [1  1  0.78]

       f1
    [1   phi]
    ---------
    [0   0.11]
    [1   0.89]


    f1 * f2
    [1  2  phi]
    -----------
    [0  0  0.11 * 0.59]
    [1  0  0.89 * 0.22]
    [0  1  0.11 * 0.41]
    [1  1  0.89 * 0.78]
    '''

    f = get_product(f2, f1)
    assert np.allclose(f.scope, np.array([1, 2]))
    assert np.allclose(f.card, np.array([2, 2]))
    assert np.allclose(f.val, np.array([0.11 * 0.59,
                                        0.89 * 0.22,
                                        0.11 * 0.41,
                                        0.89 * 0.78]))


def test_f2_f3_factor_product(f2, f3):
    '''
       f2
    [2  1  phi]
    ------------
    [0  0  0.59]
    [1  0  0.41]
    [0  1  0.22]
    [1  1  0.78]

       f3
    [3  2  phi]
    ------------
    [0  0  0.39]
    [1  0  0.61]
    [0  1  0.06]
    [1  1  0.94]

    f2 * f3
    [1  2  3  phi]
    --------------
    [0  0  0  0.59 * 0.39]
    [1  0  0  0.22 * 0.39]
    [0  1  0  0.41 * 0.06]
    [1  1  0  0.78 * 0.06]
    [0  0  1  0.59 * 0.61]
    [1  0  1  0.22 * 0.61]
    [0  1  1  0.41 * 0.94]
    [1  1  1  0.78 * 0.94]
    '''

    f = get_product(f2, f3)
    assert np.allclose(f.scope, np.array([1, 2, 3]))
    assert np.allclose(f.card, np.array([2, 2, 2]))
    assert np.allclose(f.val, np.array([0.59 * 0.39,
                                        0.22 * 0.39,
                                        0.41 * 0.06,
                                        0.78 * 0.06,
                                        0.59 * 0.61,
                                        0.22 * 0.61,
                                        0.41 * 0.94,
                                        0.78 * 0.94]))


def test_f2_marg(f2):
    '''
       f2
    [2  1  phi]
    ------------
    [0  0  0.59]
    [1  0  0.41]
    [0  1  0.22]
    [1  1  0.78]

       f
    [2  phi]
    --------
    [0  0.59 + 0.22]
    [1  0.41 + 0.78]
    '''
    f = get_marg(f2, 1)
    assert np.allclose(f.scope, np.array([2]))
    assert np.allclose(f.card, np.array([2]))
    assert np.allclose(f.val, np.array([0.59 + 0.22,
                                        0.41 + 0.78]))


def test_another_marg():
    f1 = Factor(scope=np.array([0]),
                card=np.array([2], dtype=np.int32),
                val=np.array([0.57, 0.43]))

    f2 = Factor(scope=np.array([1, 0]),
                card=np.array([2, 2], dtype=np.int32),
                val=np.array([0.4, 0.6, 0.13, 0.87]))

    f = get_product(f1, f2)
    f_marg = get_marg(f, 1)

    assert np.array_equal(f_marg.scope, np.array([0]))
    assert np.array_equal(f_marg.card, np.array([2]))
    assert np.allclose(f_marg.val, np.array([0.57, 0.43]))


def test_another_marg_log_space():
    f1 = Factor(scope=np.array([0]),
                card=np.array([2], dtype=np.int32),
                val=np.array([0.57, 0.43]))

    f2 = Factor(scope=np.array([1, 0]),
                card=np.array([2, 2], dtype=np.int32),
                val=np.array([0.4, 0.6, 0.13, 0.87]))

    f1 = log(f1)
    f2 = log(f2)

    f = get_product(f1, f2, use_log_space=True)
    f_marg = get_marg(f, 1, use_log_space=True)

    f_marg = exp(f_marg)

    assert np.array_equal(f_marg.scope, np.array([0]))
    assert np.array_equal(f_marg.card, np.array([2]))
    assert np.allclose(f_marg.val, np.array([0.57, 0.43]))


def test_is_equal_f1_f1(f1):
    assert is_equal(f1, f1)


def test_is_equal_f2_f2(f2):
    assert is_equal(f2, f2)


def test_is_equal_scope_out_of_order(f2):
    '''
        f2
    [2  1  phi]
    ------------
    [0  0  0.59]
    [1  0  0.41]
    [0  1  0.22]
    [1  1  0.78]

       f2_
    [1  2  phi]
    ------------
    [0  0  0.59]
    [1  0  0.22]
    [0  1  0.41]
    [1  1  0.78]
    '''
    f2_ = Factor(scope=np.array([1, 2]),
                 card=np.array([2, 2]),
                 val=np.array([0.59, 0.22, 0.41, 0.78]))

    assert is_equal(f2, f2_)
    assert is_equal(f2_, f2)
