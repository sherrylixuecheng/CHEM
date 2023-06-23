import numpy as np


class UnderDetermined(Exception):
    pass


class OverDetermined(Exception):
    pass


def get_inv_list(N):
    x = np.arange(N)
    return np.argmax((x[:, None] * x) % N == 1, axis=1)


def Gauss_elimination(A, b, N, verbose=False):
    # convert A by row transformation to
    # I C
    # O O
    A = A.copy()
    b = b.copy()
    inv_list = get_inv_list(N)

    def div(a, b):
        return (a * inv_list[b]) % N

    def update(A, b, k, pivot, slic):
        f = div(A[slic, pivot], A[k, pivot])
        A[slic] = (A[slic] - A[k] * f[:, None]) % N
        b[slic] = (b[slic] - b[k] * f[:, None]) % N

    n, m = A.shape
    if verbose:
        print(np.concatenate((A, b), axis=1))
        print()

    pivots = []
    pivot = 0
    for k in range(n):
        pivot = np.argmax(A[k] > 0)
        slic = slice(k+1, n)
        update(A, b, k, pivot, slic)
        if verbose:
            print(np.concatenate((A, b), axis=1))
            print()
        pivots.append(pivot)
        pivot += 1

    for k in range(n - 1, -1, -1):
        pivot = pivots[k]
        slic = slice(0, k)
        update(A, b, k, pivot, slic)
        inv = inv_list[A[k, pivot]]
        if inv != 0:
            A[k] = (A[k] * inv) % N
            b[k] = (b[k] * inv) % N
        if verbose:
            print(np.concatenate((A, b), axis=1))
            print()
    return A, b


def solve_modN_matrix(A, b, N, verbose=False):
    n, m = A.shape
    assert b.shape[0] == n
    k = b.shape[1]
    A2, b2 = Gauss_elimination(A, b, N, verbose=verbose)
    conditions = np.sum(A2, axis=1)
    if np.any(b2[conditions == 0] != 0):
        raise OverDetermined
    elif np.any(conditions != 1):
        raise UnderDetermined
    else:
        pivots = np.argmax(A2, axis=1)
        x = np.zeros((n, k), dtype=int)
        x[pivots] = b2.copy()
        return x


def solve_modN(A, b, N, verbose=False):
    return solve_modN_matrix(A, b[:, None], N)[:, 0]


def inverse_modN(A, N, verbose=False):
    n = len(A)
    assert A.shape == (n, n)
    I = np.eye(n, dtype=int)
    return solve_modN_matrix(A, I, N, verbose=verbose)


def Gauss_elimination_Aonly(A, N, verbose=False):
    b = np.zeros((len(A), 0), dtype=int)
    A2, b2 = Gauss_elimination(A, b, N)
    return A2


if __name__ == '__main__':
    np.random.seed(0)
    n = 5
    N = 3
    results = []
    for i in range(100):
        A = np.random.randint(0, N, size=(n, n))
        try:
            Ainv = inverse_modN(A, N)
            equal = np.all(Ainv.dot(A) % N == np.eye(n))
            assert equal
            results.append(0)
        except OverDetermined:
            results.append(1)
        except UnderDetermined:
            results.append(2)
    results = np.array(results)
    assert np.all(results == np.load("./singular.npy"))
