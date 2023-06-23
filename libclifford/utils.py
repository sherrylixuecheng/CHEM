import numpy as np
import time


def random_unitary(n):
    return np.linalg.qr(np.random.random((2**n, 2**n)) + np.random.random((2**n, 2**n)) * 1j)[0].reshape((2, ) * n * 2)


def count_gates(gates, gate):
    return len([None for item in gates if item[0] == gate])


def get_max_n(gates):
    indices = np.concatenate([index for gate, index in gates])
    return np.max(indices) + 1


def change_matrix(A, row_index, col_index, B):
    tmp = A[row_index].copy()
    tmp[:, col_index] = B.copy()
    A[row_index] = tmp


def indices_ndim(values, n):
    N = len(values) ** n
    return np.array(np.meshgrid(*([values] * n))).reshape((n, N)).T


class Timer():
    def __init__(self, verbose, **print_options):
        self.begin = time.time()
        self.verbose = verbose
        self.print_options = print_options

    def time(self, name=None):
        t = time.time() - self.begin
        if self.verbose:
            if name is None:
                print(f'time: {t}', **self.print_options)
            else:
                print(f'{name} done. time: {t}', **self.print_options)
        self.begin = time.time()
