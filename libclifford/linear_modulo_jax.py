import numpy as np
import jax.numpy as jnp


class UnderDetermined(Exception):
    pass


class OverDetermined(Exception):
    pass


def get_inv_list(N):
    x = jnp.arange(N)
    return jnp.argmax((x[:, None] * x) % N == 1, axis=1)


def Gauss_elimination(A: jnp.ndarray, b: jnp.ndarray, N: int):
    # convert A by row transformation to
    # I C
    # O O
    inv_list = get_inv_list(N)

    def div(a, b):
        return (a * inv_list[b]) % N

    def update(A, b, k, pivot, slic):
        f = div(A[slic, pivot], A[k, pivot])
        new_A_slic = (A[slic] - A[k] * f[:, None]) % N
        new_b_slic = (b[slic] - b[k] * f[:, None]) % N
        A = A.at[slic].set(new_A_slic)
        b = b.at[slic].set(new_b_slic)
        return A, b

    n, m = A.shape

    pivots = []
    pivot = 0
    for k in range(n):
        pivot = jnp.argmax(A[k] > 0)
        slic = slice(k+1, n)
        A, b = update(A, b, k, pivot, slic)
        pivots.append(pivot)
        pivot += 1

    for k in range(n - 1, -1, -1):
        pivot = pivots[k]
        slic = slice(0, k)
        A, b = update(A, b, k, pivot, slic)

        inv = inv_list[A[k, pivot]]
        new_A_k = (A[k] * inv) % N
        new_b_k = (b[k] * inv) % N
        new_A = A.at[k].set(new_A_k)
        new_b = b.at[k].set(new_b_k)
        A = new_A * (inv != 0) + A * (inv == 0)
        b = new_b * (inv != 0) + b * (inv == 0)
    return A, b


def solve_modN_matrix(A: jnp.ndarray, b: jnp.ndarray, N: int):
    n, m = A.shape
    assert b.shape[0] == n
    k = b.shape[1]
    A2, b2 = Gauss_elimination(A, b, N)
    conditions = jnp.sum(A2, axis=1)
    is_over = jnp.any(jnp.any(b2 != 0) & (conditions == 0))
    is_under = jnp.any(conditions != 1)
    pivots = jnp.argmax(A2, axis=1)
    x = jnp.zeros((n, k), dtype=int)
    x = x.at[pivots].set(b2)
    return is_over, is_under, x


def _solve_modN_matrix(A, b, N):
    is_over, is_under, x = solve_modN_matrix(A, b, N)
    if is_over:
        raise OverDetermined
    elif is_under:
        raise UnderDetermined
    else:
        return x


def _solve_modN(A, b, N):
    return _solve_modN_matrix(A, b[:, None], N)[:, 0]


def inverse_modN(A, N):
    n = len(A)
    assert A.shape == (n, n)
    I = jnp.eye(n, dtype=int)
    return solve_modN_matrix(A, I, N)


def _inverse_modN(A, N):
    n = len(A)
    assert A.shape == (n, n)
    I = np.eye(n, dtype=int)
    return _solve_modN_matrix(A, I, N)


def _Gauss_elimination_Aonly(A, N):
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
            Ainv = _inverse_modN(A, N)
            equal = np.all(Ainv.dot(A) % N == np.eye(n))
            assert equal
            results.append(0)
        except OverDetermined:
            results.append(1)
        except UnderDetermined:
            results.append(2)
    results = np.array(results)
    assert np.all(results == np.load("./singular.npy"))
