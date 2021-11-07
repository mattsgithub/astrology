from collections import namedtuple
import numpy as np


Factor = namedtuple('Factor', 'scope card val')
Scalar = namedtuple('Scalar', 'val')


def renormalize(f):
    val = f.val / f.val.sum()
    factor = Factor(scope=f.scope,
                    card=f.card,
                    val=val)
    return factor


def log(f):
    return Factor(scope=f.scope,
                  card=f.card,
                  val=np.log(f.val))


def exp(f):
    # Normalize
    val = np.exp(f.val)
    return Factor(scope=f.scope,
                  card=f.card,
                  val=val)


def get_matching_indices(f1,
                         f2,
                         map1,
                         map2):
    u = np.union1d(f1.scope, f2.scope)
    card = np.empty((u.shape[0],), dtype=int)
    N = np.prod(card)
    assign = indx_to_assign(np.arange(N), card)
    I1 = assign_to_indx(assign[:, map1], f1.card)
    I2 = assign_to_indx(assign[:, map2], f2.card)
    return I1, I2


def get_mappings(x, y):

    # Find union of both arrays
    u = np.union1d(x, y)

    # Get indices of elements
    # for union array (in sorted order)
    sorted_indx = u.argsort()

    u_sorted = u[sorted_indx]
    mapX = sorted_indx[np.searchsorted(u_sorted, x)]
    mapY = sorted_indx[np.searchsorted(u_sorted, y)]
    return mapX, mapY


def is_equal(f1, f2):

    if np.setdiff1d(f1.scope, f2.scope).shape[0] > 0:
        return False

    map1, map2 = get_mappings(f1.scope,
                              f2.scope)
    # Are cards equal?
    if not np.array_equal(f1.card[map1], f2.card[map2]):
        return False

    I1, I2 = get_matching_indices(f1,
                                  f2,
                                  map1,
                                  map2)
    # Are values equal?
    if not np.allclose(f1.val[I1], f2.val[I2]):
        return False

    return True


def assign_to_indx(A, C):
    """Given assignment(s) and
       card of factor, returns
       index/indices
     """

    # Returns M x 1
    # return A.dot(s.reshape((len(s), 1)))
    # Returns 1 dimesional array
    s = get_stride(C)
    return A.dot(s.reshape((len(s), 1))).reshape((A.shape[0],))


def get_stride(card):
    '''Returns 1-dimesional numpy array
       First element, by definition, is always
       1. Because it's the fastest moving
       column
    '''
    return np.cumprod(np.concatenate((np.array([1]), card[:-1])))


def indx_to_assign(i, c):
    """Calculates assignment vectors for
    a given index vector, i

    i: 1d np.array
    c: 1d np.array

    Idea is to take vector i and create matrix:

    i0 i0 i0
    i1 i1 i1

    Cast stride vector as column vector:
    s0
    s1
    s2


    I / s gives...

    i0/s0  i0/s1  i0/s2
    i1/s0  i1/s1  i1/s2

    No need to take floor, because we
    are dividing integers...

    Finally, take mod car

    x = floor(index / phi.stride(i))
    assignment[i] = x  mod card[i]
    """
    s = get_stride(c)
    i_ = i.reshape((len(i), 1))
    I = np.repeat(i_, len(s), axis=1)
    return np.mod(I / s, c)


def get_marg(A, v, use_log_space=False):
    if len(A.scope) == 1 and A.scope[0] == v:
        return Scalar(val=1.)

    # Convert out of log space
    if use_log_space:
        A = exp(A)

    # Get indices for every variable
    # except one being summed out
    map_ = np.where(A.scope != v)

    # Scope of returning factor
    scope = A.scope[map_]

    # Card of returning factor
    card = A.card[map_]

    # Init val of returning factor
    N = np.prod(card)
    val = np.zeros((N,))

    # Indices for all vals in A
    I = np.arange(0, len(A.val))

    # Get complete assignment matrix for A
    A_ = indx_to_assign(I, A.card)

    indx = assign_to_indx(A_[:, map_], card)

    # TODO: Redo without for loop?
    # For same index values, sum, and place
    # at index for returning factor
    for i, j in enumerate(indx):
        # Corresponding
        v = A.val[i]
        val[j] += v

    f = Factor(scope,
               card,
               val)
    
    # Convert back to log space
    if use_log_space:
        f = log(f)

    return f


def get_product_from_list(f, use_log_space=False):
    if len(f) == 0:
        return Scalar(val=1.)

    func = lambda x, y: get_product(x, y, use_log_space=use_log_space)
    return reduce(func, f)


def get_product(A, B, use_log_space=False):
    if isinstance(A, Scalar):
        return B
    if isinstance(B, Scalar):
        return A
    if A is None:
        return B
    elif B is None:
        return A
    scope = np.union1d(A.scope, B.scope)

    mapA, mapB = get_mappings(A.scope,
                              B.scope)

    card = np.empty((scope.shape[0],), dtype=int)

    np.put(card, mapA, A.card)
    np.put(card, mapB, B.card)

    N = np.prod(card)
    val = np.empty((N,))
    assign = indx_to_assign(np.arange(N), card)

    indx_A = assign_to_indx(assign[:, mapA], A.card)
    indx_B = assign_to_indx(assign[:, mapB], B.card)

    if use_log_space:
        #val = np.logaddexp(A.val[indx_A], B.val[indx_B])
        val = A.val[indx_A] + B.val[indx_B]
    else:
        val = A.val[indx_A] * B.val[indx_B]

    return Factor(scope,
                  card,
                  val)
