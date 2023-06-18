import numpy as np
import qiskit.quantum_info as qskQ
from libclifford import hamiltonian as Ihamiltonian
from libclifford import clifford as Iclifford
from libclifford.jaxclifford import PauliArray, CliffordArray
from libclifford import linear_modulo, linear_modulo_jax
import jax
import jax.numpy as jnp
from functools import partial

n = 0
ry_names = ['i', 'sy', 'y', 'sydg']
ry_ids = np.array([Iclifford.clifford1.index(Iclifford.clifford_map[name]) for name in ry_names])
clifford1 = CliffordArray.from_qiskit(Iclifford.clifford1)
cnot = CliffordArray.from_qiskit(Iclifford.gate2clifford('cnot', [0, 1], 2))


def simplify_pauli(p: qskQ.Pauli):
    p = p.copy()
    p.z = False
    return p


def simplify_hamiltonian(hamiltonian: Ihamiltonian.Hamiltonian, remove_I=True):
    keys = [simplify_pauli(item) for item in list(hamiltonian.keys())]
    values = np.array(list(hamiltonian.values()))
    data = {}
    for key, value in zip(keys, values):
        assert key.phase in [0, 2]
        if key.phase == 0:
            positive_key = key
            positive_value = value
        else:
            positive_key = -key
            positive_value = -value
        if positive_key not in data:
            data[positive_key] = 0
        data[positive_key] += positive_value
    for key in list(data.keys()):
        if abs(data[key]) < 1e-10:
            del data[key]
    if remove_I:
        identity = Iclifford.identity_pauli(n)
        del data[identity]
    return Ihamiltonian.Hamiltonian(data)


@jax.jit
def diagonal_clifford_jax(x, phasex=None):
    n = len(x)
    is_over, is_under, z = linear_modulo_jax.inverse_modN(x.astype(int), 2)
    z = z.astype(bool)
    symplectic_matrix = jnp.zeros((2 * n, 2 * n), dtype=bool)
    symplectic_matrix = symplectic_matrix.at[:n, :n].set(x.T)
    symplectic_matrix = symplectic_matrix.at[n:, n:].set(z)
    phase = jnp.zeros((2 * n), dtype=int)
    if phasex is not None:
        phase[:n] = phasex
    return is_over, is_under, CliffordArray(symplectic_matrix, phase)


def diagonal_clifford(x, phasex=None):
    n = len(x)
    try:
        z = linear_modulo.inverse_modN(x.astype(int), 2).astype(bool)
    except linear_modulo.OverDetermined:
        return None
    symplectic_matrix = np.zeros((2 * n, 2 * n), dtype=bool)
    symplectic_matrix[:n, :n] = x.T
    symplectic_matrix[n:, n:] = z
    phase = np.zeros((2 * n), dtype=int)
    if phasex is not None:
        phase[:n] = phasex
    return CliffordArray(symplectic_matrix, phase)


def _random_diagonal_clifford(random_phase=False):
    x = np.random.random((n, n)) > 0.5
    if random_phase:
        phasex = np.random.randint(0, 4, (n, ))
    else:
        phasex = None
    return diagonal_clifford(x, phasex)


def random_diagonal_clifford(random_phase=False):
    while True:
        transformation = _random_diagonal_clifford(random_phase=random_phase)
        if transformation is not None:
            return transformation


@jax.jit
def create_new_layer(clifford: CliffordArray, id_layer: jnp.ndarray):
    new_clifford = clifford.copy()
    for site, id in enumerate(id_layer):
        new_clifford = clifford1[id].dot_subsystem(new_clifford, jnp.array([site]))
    for site in range(n - 1):
        new_clifford = cnot.dot_subsystem(new_clifford, jnp.array([site, site+1]))
    new_Ps_transform = new_clifford.transform_XYZ()
    return new_clifford, new_Ps_transform


class HEAState:
    def __init__(self, ids, use_Ry=False):
        if use_Ry:
            ids = ry_ids[ids]
        self.ids = ids
        self.Ps_transform_all = []
        self.clifford = CliffordArray.identity(n)
        for id_layer in ids:
            self.clifford, Ps_transform = create_new_layer(self.clifford, id_layer)
            self.Ps_transform_all.append(Ps_transform)
        self.Ps_transform_all = PauliArray.concatenate(tuple(self.Ps_transform_all))
        self.Ps_transform_all = self.Ps_transform_all.reshape((-1, ))

    @staticmethod
    def find_match(a, b):
        argsort = jnp.argsort(a)
        indices = jnp.searchsorted(a[argsort], b)
        indices -= (indices >= len(a))
        original_indices = argsort[indices]
        return original_indices, a[original_indices] == b

    @staticmethod
    def find_unique(x):
        #ids_P_x = P_x.astype(int).dot(2**jnp.arange(n))
        x_unique, x_indices = jnp.unique(x, size=len(x), fill_value=-1, return_index=True)
        #ids_P_x_indices = ids_P_x_indices.at[ids_P_x_unique == -1].set(-1)
        is_unique = jnp.zeros(x.shape, dtype=bool).at[x_indices].set(True)
        return is_unique

    @classmethod
    @partial(jax.jit, static_argnums=(0, ))
    def find_best_transformation(cls, Ps: PauliArray, h_pauli: PauliArray, h_values: jnp.ndarray):

        # c1.dot(c2).x = c2.x.dot(c1.x)
        # c1.dot(c2).z = c2.z.dot(c1.z)
        # p.evolve(c, frame='s').x = c.x.T.dot(p.x)
        # p.evolve(c, frame='s').z = c.z.T.dot(p.z)

        odd_Y = Ps._phase % 2 == 1
        P_x = Ps.x * odd_Y[:, None]
        h_x = h_pauli.x

        A_h_x, b_h_x = linear_modulo_jax.Gauss_elimination(h_x.T.astype(int), jnp.eye(n, dtype=int), 2)
        h_nonzero = jnp.sum(A_h_x, axis=1) > 0
        h_indices = jnp.argsort(jnp.logical_not(h_nonzero))

        A_P_x, b_P_x = linear_modulo_jax.Gauss_elimination(P_x.T.astype(int), jnp.eye(n, dtype=int), 2)
        P_nonzero = jnp.sum(A_P_x, axis=1) > 0
        P_indices = jnp.argsort(jnp.logical_not(P_nonzero))

        b_h_x_final = b_h_x[h_indices]
        b_P_x_final = b_P_x[P_indices]
        is_over1, is_under1, b_P_x_final_inv = linear_modulo_jax.inverse_modN(b_P_x_final, 2)
        U = b_P_x_final_inv.dot(b_h_x_final) % 2
        is_over2, is_under2, diagonal_clifford = diagonal_clifford_jax(U)
        U_cliff = diagonal_clifford.adjoint()
        # U = U_cliff.stab_z.astype(int)

        h_x_transform = (h_x.astype(int).dot(U.T) % 2).astype(bool)
        weights_ids = 2**jnp.arange(n)
        ids_h_x_transform = h_x_transform.astype(int).dot(weights_ids)
        ids_P_x = P_x.astype(int).dot(weights_ids)

        indices, match = cls.find_match(ids_h_x_transform, ids_P_x)
        P_sign = 1 - (Ps._phase % 4) // 2 * 2  # phase 1: 1, phase 3: -1
        grads = h_values[indices] * match * P_sign
        is_unique = cls.find_unique(ids_P_x)
        score = jnp.linalg.norm(grads * is_unique)

        # sanity check
        h_pivots = jnp.argmax(A_h_x[h_indices], axis=1)
        P_pivots = jnp.argmax(A_P_x[P_indices], axis=1)
        overlap = (jnp.sort(P_nonzero) & jnp.sort(h_nonzero))[::-1]
        equal = (ids_P_x[P_pivots] == ids_h_x_transform[h_pivots])
        success1 = jnp.all(equal | jnp.logical_not(overlap))
        success2 = jnp.logical_not(is_over1 | is_over2 | is_under1 | is_under2)
        success = success1 & success2

        return U_cliff, grads, score, success

    @staticmethod
    @jax.jit
    def get_original_order(grads: jnp.ndarray, Ps_order: jnp.ndarray):
        original_grads = jnp.zeros_like(grads).at[Ps_order].set(grads)
        return original_grads

    @staticmethod
    @jax.jit
    def get_ordered_Ps(Ps: PauliArray, Ps_order: jnp.ndarray):
        # need jit here
        return Ps[Ps_order]

    def find_best_transformation_reorder(self, h_pauli, h_values, Ps_order=None):
        if Ps_order is None:
            Ps_order = slice(None)
        ordered_Ps = self.get_ordered_Ps(self.Ps_transform_all, Ps_order)
        U_cliff, grads, score, success = self.find_best_transformation(ordered_Ps, h_pauli, h_values)
        assert success
        original_grads = self.get_original_order(grads, Ps_order)
        return U_cliff, original_grads, score
