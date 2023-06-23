import typing as tp
import qiskit.quantum_info as qskQ
import jax
import jax.numpy as jnp
import numpy as np
from functools import reduce

slice_all = slice(None, None, None)
CliffordTypes = tp.Union[qskQ.Clifford, "CliffordArray"]
PauliTypes = tp.Union[qskQ.Pauli, qskQ.PauliList, "PauliArray"]


def asarray(array):
    if not isinstance(array, jnp.ndarray):
        return jnp.array(array)
    else:
        return array


class CliffordArray:
    _COMPOSE_PHASE_LOOKUP = None

    def __init__(self, symplectic_matrix: jnp.ndarray, phase: jnp.ndarray):
        self.symplectic_matrix = asarray(symplectic_matrix)
        self.phase = asarray(phase)
        self.num_qubits = self.phase.shape[-1] // 2
        self.shape = self.phase.shape[0:-1]
        assert self.symplectic_matrix.shape == self.shape + (self.num_qubits * 2, self.num_qubits * 2)
        self._compose_lookup()

    @property
    def x(self):
        return self.symplectic_matrix[..., :self.num_qubits]

    @x.setter
    def x(self, value):
        self.symplectic_matrix = self.symplectic_matrix.at[..., :self.num_qubits].set(value)

    @property
    def z(self):
        return self.symplectic_matrix[..., self.num_qubits:]

    @z.setter
    def z(self, value):
        self.symplectic_matrix = self.symplectic_matrix.at[..., self.num_qubits:].set(value)

    @property
    def destab_z(self):
        return self.symplectic_matrix[..., :self.num_qubits, self.num_qubits:]

    @destab_z.setter
    def destab_z(self, value):
        self.symplectic_matrix = self.symplectic_matrix.at[..., :self.num_qubits, self.num_qubits:].set(value)

    @property
    def stab_z(self):
        return self.symplectic_matrix[..., self.num_qubits:, self.num_qubits:]

    @stab_z.setter
    def stab_z(self, value):
        self.symplectic_matrix = self.symplectic_matrix.at[..., self.num_qubits:, self.num_qubits:].set(value)

    @property
    def destab_x(self):
        return self.symplectic_matrix[..., :self.num_qubits, :self.num_qubits]

    @destab_x.setter
    def destab_x(self, value):
        self.symplectic_matrix = self.symplectic_matrix.at[..., :self.num_qubits, :self.num_qubits].set(value)

    @property
    def stab_x(self):
        return self.symplectic_matrix[..., self.num_qubits:, :self.num_qubits]

    @stab_x.setter
    def stab_x(self, value):
        self.symplectic_matrix = self.symplectic_matrix.at[..., self.num_qubits:, :self.num_qubits].set(value)

    @property
    def destab_phase(self):
        return self.phase[..., :self.num_qubits]

    @destab_phase.setter
    def destab_phase(self, value):
        self.phase = self.phase.at[..., :self.num_qubits].set(value)

    @property
    def stab_phase(self):
        return self.phase[..., self.num_qubits:]

    @stab_phase.setter
    def stab_phase(self, value):
        self.phase = self.phase.at[..., self.num_qubits:].set(value)

    @classmethod
    def identity(cls, num_qubits: int, shape=()) -> "CliffordArray":
        phase_shape = shape + (num_qubits * 2, )
        matrix_shape = shape + (num_qubits * 2, num_qubits * 2)
        symplectic_matrix = jnp.zeros(matrix_shape, dtype=bool)
        phase = jnp.zeros(phase_shape, dtype=bool)
        i1, i2 = jnp.diag_indices(num_qubits * 2)
        symplectic_matrix = symplectic_matrix.at[..., i1, i2].set(True)
        return cls(symplectic_matrix, phase)

    def _change_subspace(self, clifford: "CliffordArray", indices):
        m = len(indices)
        assert m == clifford.num_qubits
        indices = asarray(indices)
        indices = jnp.concatenate([indices, indices + self.num_qubits])
        a, b = jnp.meshgrid(indices, indices)
        self.symplectic_matrix = self.symplectic_matrix.at[b, a].set(clifford.symplectic_matrix.copy())
        self.phase = self.phase.at[indices].set(clifford.phase.copy())

    @classmethod
    def from_subspace(cls, clifford: "CliffordArray", indices, n) -> "CliffordArray":
        result = cls.identity(n)
        result._change_subspace(clifford, indices)
        return result

    @classmethod
    def from_qiskit(cls, clifford: tp.Union[CliffordTypes, tp.Sequence[CliffordTypes]]) -> "CliffordArray":
        if isinstance(clifford, (qskQ.Clifford, cls)):
            return cls(clifford.symplectic_matrix, clifford.phase)
        else:
            objs = [cls.from_qiskit(item) for item in clifford]
            symplectic_matrix = jnp.array([item.symplectic_matrix for item in objs])
            phase = jnp.array([item.phase for item in objs])
            return cls(symplectic_matrix, phase)

    def to_qiskit(self):
        assert self.shape == ()
        array = np.array(self.symplectic_matrix)
        phase = np.array(self.phase)
        return qskQ.Clifford(qskQ.StabilizerTable(array, phase=phase))

    def to_array(self):
        return jnp.concatenate([self.symplectic_matrix, self.phase[..., None]], axis=-1)

    @classmethod
    def from_array(cls, array):
        num_qubits = array.shape[-1] // 2
        assert array.shape[-1] == num_qubits * 2 + 1
        assert array.shape[-2] == num_qubits * 2
        symplectic_matrix = array[..., 0:num_qubits * 2]
        phase = array[..., -1]
        return cls(symplectic_matrix, phase)

    def copy(self):
        return self.__class__(self.symplectic_matrix.copy(), self.phase.copy())

    @staticmethod
    def get_index(index):
        if not isinstance(index, tuple):
            index = (index, )
        index1 = index + (slice_all, )
        index2 = index + (slice_all, slice_all)
        return index, index1, index2

    def __getitem__(self, index):
        index, index1, index2 = self.get_index(index)
        return self.__class__(self.symplectic_matrix[index2], self.phase[index1])

    def __setitem__(self, index, value: "CliffordArray"):
        index, index1, index2 = self.get_index(index)
        self.symplectic_matrix = self.symplectic_matrix.at[index2].set(value.symplectic_matrix)
        self.phase = self.phase.at[index1].set(value.phase)

    def __len__(self):
        return self.shape[0]

    def __eq__(self, other: "CliffordArray"):
        cond1 = self.shape == other.shape
        cond2 = jnp.all(self.symplectic_matrix == other.symplectic_matrix)
        cond3 = jnp.all(self.phase == other.phase)
        return cond1 and cond2 and cond3

    def __repr__(self):
        """Display representation."""
        return self.__str__()

    def __str__(self):
        """Print representation."""
        ps = self.to_pauli()
        destabilizers = ps[:self.num_qubits]
        stabilizers = ps[self.num_qubits:]
        data = {'destabilizers': destabilizers,
                'stabilizers': stabilizers,
                }
        return str(data)

    def _tree_flatten(self):
        dynamic = (self.symplectic_matrix, self.phase)
        static = {}
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        symplectic_matrix, phase = dynamic
        return cls(symplectic_matrix, phase)

    @classmethod
    def _compose_lookup(cls):
        if cls._COMPOSE_PHASE_LOOKUP is None:
            # A lookup table for calculating phases.  The indices are
            #     current_x, current_z, running_x_count, running_z_count
            # where all counts taken modulo 2.
            lookup = jnp.zeros((2, 2, 2, 2), dtype=int)
            for index in [(0, 1, 1, 0), (1, 0, 1, 1), (1, 1, 0, 1)]:
                lookup = lookup.at[index].set(-1)
            for index in [(0, 1, 1, 1), (1, 0, 0, 1), (1, 1, 1, 0)]:
                lookup = lookup.at[index].set(1)
            cls._COMPOSE_PHASE_LOOKUP = lookup

    def transform_XYZ(self) -> "PauliArray":
        inv = self.adjoint()
        Zs = PauliArray(inv.stab_x, inv.stab_z, inv.stab_phase * 2, realphase=True)
        Xs = PauliArray(inv.destab_x, inv.destab_z, inv.destab_phase * 2, realphase=True)
        Ys = Xs.dot(Zs) * 1j
        return PauliArray.from_qiskit([Xs, Ys, Zs])

    def to_pauli(self) -> "PauliArray":
        return PauliArray(self.x, self.z, self.phase * 2, realphase=True)

    @classmethod
    def from_pauli(cls, ps: "PauliArray") -> "CliffordArray":
        symplectic_matrix = jnp.concatenate([ps.x, ps.z], axis=-1)
        return cls(symplectic_matrix, ps.phase // 2)

    def dot(first: "CliffordArray", second: "CliffordArray") -> "CliffordArray":
        return CliffordArray.from_pauli(second.to_pauli().evolve(first[..., None], frame='s'))

    def dot_subsystem(first: "CliffordArray", second: "CliffordArray", qargs1: jnp.ndarray) -> "CliffordArray":
        return CliffordArray.from_pauli(second.to_pauli().evolve_subsystem(first[..., None], qargs1, frame='s'))

    def adjoint(self) -> "CliffordArray":
        result = self.copy()
        tmp = result.destab_x.copy()
        result.destab_x = result.stab_z.swapaxes(-1, -2)
        result.destab_z = result.destab_z.swapaxes(-1, -2)
        result.stab_x = result.stab_x.swapaxes(-1, -2)
        result.stab_z = tmp.swapaxes(-1, -2)
        result.phase ^= self.dot(result).phase
        return result


class PauliArray:
    phase_factor = {
        1: 0,
        -1j: 1,
        -1: 2,
        1j: 3,
    }

    def __init__(self, x: jnp.ndarray, z: jnp.ndarray, _phase: jnp.ndarray, realphase=False):
        self.x = asarray(x)
        self.z = asarray(z)
        if realphase:
            self.phase = asarray(_phase)
        else:
            self._phase = asarray(_phase)
        self.shape = self._phase.shape
        self.num_qubits = self.x.shape[-1]
        assert self.x.shape == self.shape + (self.num_qubits, )
        assert self.z.shape == self.shape + (self.num_qubits, )

    @classmethod
    def identity(cls, num_qubits: int, shape=()) -> "PauliArray":
        array_shape = shape + (num_qubits, )
        x = jnp.zeros(array_shape, dtype=bool)
        z = jnp.zeros(array_shape, dtype=bool)
        _phase = jnp.zeros(shape, dtype=int)
        return cls(x, z, _phase)

    @classmethod
    def from_part(cls, indices: jnp.ndarray, ps: "PauliArray") -> "PauliArray":
        x = indices[..., None] * ps.x
        z = indices[..., None] * ps.z
        _phase = indices * ps._phase
        return cls(x, z, _phase)

    @classmethod
    def from_qiskit(cls, p: tp.Union[PauliTypes, tp.Sequence[PauliTypes]]) -> "PauliArray":
        if isinstance(p, (qskQ.Pauli, qskQ.PauliList, cls)):
            return cls(p.x, p.z, p._phase)
        else:
            objs = [cls.from_qiskit(item) for item in p]
            x = jnp.array([item.x for item in objs])
            z = jnp.array([item.z for item in objs])
            _phase = jnp.array([item._phase for item in objs])
            return cls(x, z, _phase)

    def to_qiskit(self):
        data = (self.z, self.x, self.phase)
        if len(self.shape) == 0:
            return qskQ.Pauli(data)
        elif len(self.shape) == 1:
            return qskQ.PauliList.from_symplectic(*data)
        else:
            raise ValueError("Dimension higher than one")

    def to_array(self):
        return jnp.concatenate([self.x, self.z, self._phase[..., None]], axis=-1)

    @classmethod
    def from_array(cls, array):
        num_qubits = array.shape[-1] // 2
        assert array.shape[-1] == num_qubits * 2 + 1
        x = array[..., 0:num_qubits]
        z = array[..., num_qubits: num_qubits * 2]
        _phase = array[..., -1]
        return cls(x, z, _phase)

    @staticmethod
    def count_y(x, z):
        return (x & z).sum(axis=-1)

    @classmethod
    def get_real_phase(cls, x, z, _phase):
        return jnp.mod(_phase - cls.count_y(x, z), 4)

    @classmethod
    def get_inner_phase(cls, x, z, phase):
        return jnp.mod(phase + cls.count_y(x, z), 4)

    @property
    def phase(self):
        """Return the phase ejnponent of the PauliList."""
        # Convert internal ZX-phase convention to group phase convention
        return self.get_real_phase(self.x, self.z, self._phase)

    @phase.setter
    def phase(self, value):
        # Convert group phase convetion to internal ZX-phase convention
        self._phase = self.get_inner_phase(self.x, self.z, value)

    @staticmethod
    def get_index(index):
        if not isinstance(index, tuple):
            index = (index, )
        index1 = index + (slice_all, )
        return index, index1

    def __getitem__(self, index):
        index, index1 = self.get_index(index)
        return self.__class__(self.x[index1], self.z[index1], self._phase[index])

    def __setitem__(self, index, value: "PauliArray"):
        index, index1 = self.get_index(index)
        self.x = self.x.at[index1].set(value.x)
        self.z = self.z.at[index1].set(value.z)
        self._phase = self._phase.at[index].set(value._phase)

    def __len__(self):
        return self.shape[0]

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index == len(self):
            raise StopIteration
        else:
            result = self[self.index]
            self.index += 1
            return result

    def __mul__(self, factor):
        assert factor in self.phase_factor
        _phase = (self._phase + self.phase_factor[factor] % 4)
        return self.__class__(self.x, self.z, _phase)

    def _tree_flatten(self):
        dynamic = (self.x, self.z, self._phase)
        static = {}
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        x, z, _phase = dynamic
        return cls(x, z, _phase)

    def get_subsystem(self, index, keep_phase=True) -> "PauliArray":
        if keep_phase:
            return self.__class__(self.x[..., index], self.z[..., index], self._phase)
        else:
            return self.__class__(self.x[..., index], self.z[..., index], jnp.zeros_like(self._phase))

    def set_subsystem(self, index, sub: "PauliArray", old_phase=False, new_phase=True):
        self.x = self.x.at[..., index].set(sub.x)
        self.z = self.z.at[..., index].set(sub.z)
        if not old_phase:
            self._phase = jnp.zeros_like(self._phase)
        if new_phase:
            self._phase = (self._phase + sub._phase) % 4

    def reshape(self, shape) -> "PauliArray":
        extended_shape = shape + (self.num_qubits, )
        return PauliArray(self.x.reshape(extended_shape),
                          self.z.reshape(extended_shape),
                          self._phase.reshape(shape))

    @classmethod
    def concatenate(cls, arrays: tp.List["PauliArray"], axis=0) -> "PauliArray":
        if axis < 0:
            axis1 = axis - 1
        else:
            axis1 = axis
        x = jnp.concatenate([item.x for item in arrays], axis=axis1)
        z = jnp.concatenate([item.z for item in arrays], axis=axis1)
        _phase = jnp.concatenate([item._phase for item in arrays], axis=axis)
        return cls(x, z, _phase)

    @property
    def label(self):
        x = np.array(self.x).astype(int)
        z = np.array(self.z).astype(int)
        phase = np.array(self.phase)
        signs = np.array(np.array(['+', '-i', '-', 'i'])[phase])[..., None]
        chars = np.array(np.array(['I', 'Z', 'X', 'Y'])[x * 2 + z])[..., ::-1]
        chars_all = np.concatenate([signs, chars], axis=-1)
        return reduce(np.char.add, list(np.moveaxis(chars_all, -1, 0)))

    def __repr__(self):
        """Display representation."""
        return self.__str__()

    def __str__(self):
        """Print representation."""
        return str(self.label)

    def __eq__(self, other: "PauliArray"):
        cond1 = (self.shape == other.shape)
        cond2 = jnp.all(self.x == other.x)
        cond3 = jnp.all(self.z == other.z)
        cond4 = jnp.all(self.phase == other.phase)
        return cond1 and cond2 and cond3 and cond4

    def copy(self):
        return self.__class__(self.x.copy(), self.z.copy(), self._phase.copy())

    def compose(self, other, front=False):
        x1, z1 = self.x, self.z
        x2, z2 = other.x, other.z

        # Get phase shift
        if front:
            _phase = self._phase + other._phase + 2 * self.count_y(x1, z2)
        else:
            _phase = self._phase + other._phase + 2 * self.count_y(x2, z1)

        x = jnp.logical_xor(x1, x2)
        z = jnp.logical_xor(z1, z2)
        return PauliArray(x, z, _phase % 4)

    def dot(self, other) -> "PauliArray":
        return self.compose(other, front=True)

    def evolve(self, clifford: CliffordArray, frame='h') -> "PauliArray":
        final_shape = jnp.broadcast_shapes(self.shape, clifford.shape)
        _phase = jnp.broadcast_to(self._phase, final_shape)
        result = self.identity(self.num_qubits, final_shape)
        result._phase = _phase
        if frame == 'h':
            clifford = clifford.adjoint()
        idx = jnp.concatenate((self.x, self.z), axis=-1)
        ps_clifford = clifford.to_pauli()
        for i in range(self.num_qubits * 2):
            idx_ = idx[..., i]
            row = ps_clifford[..., i]
            row_all = self.from_part(idx_, row)
            result = result.compose(row_all)
        return result

    def evolve_subsystem(self, clifford: CliffordArray, qargs: jnp.ndarray, frame='h') -> "PauliArray":
        result = self
        sub_self = result.get_subsystem(qargs, keep_phase=True)
        sub_self_evolve = sub_self.evolve(clifford, frame=frame)
        result.set_subsystem(qargs, sub_self_evolve, old_phase=False, new_phase=True)
        return result


jax.tree_util.register_pytree_node(CliffordArray,
                                   CliffordArray._tree_flatten,
                                   CliffordArray._tree_unflatten)
jax.tree_util.register_pytree_node(PauliArray,
                                   PauliArray._tree_flatten,
                                   PauliArray._tree_unflatten)
