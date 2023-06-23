import typing as tp
import numpy as np
import qiskit.quantum_info as qskQ


class Hamiltonian:
    def __init__(self, paulis: qskQ.PauliList, values: np.ndarray):
        self.n = paulis.num_qubits
        self.paulis = paulis
        self.values = values

    @staticmethod
    def term2pauli(term, n):
        string = ['I'] * n
        for i, char in term:
            string[i] = char
        return qskQ.Pauli(''.join(string)[::-1])

    @staticmethod
    def pauli2term(pauli):
        assert pauli.phase == 0
        string = pauli.to_label()[::-1]
        term = tuple([(i, char) for i, char in enumerate(list(string)) if char != 'I'])
        return term

    @classmethod
    def from_openfermion(cls, H, n=None) -> "Hamiltonian":
        terms = H.terms
        if n is None:
            indices = np.array(sum([[key[0] for key in term] for term in terms], []), dtype=int)
            n = np.max(indices) + 1
        paulis = []
        values = []
        for d in terms:
            pauli = cls.term2pauli(d, n)
            value = terms[d]
            paulis.append(pauli)
            values.append(value)
        paulis = qskQ.PauliList(paulis)
        values = np.array(values)
        return cls(paulis, values)

    def to_openfermion(self):
        from openfermion.ops import QubitOperator
        terms = {self.pauli2term(pauli): value for pauli, value in zip(self.paulis, self.values)}
        return QubitOperator.accumulate([QubitOperator(k, terms[k]) for k in terms])

    def transform_clifford(self, cliff: qskQ.Clifford) -> "Hamiltonian":
        new_paulis = self.paulis.evolve(cliff)
        new_values = self.values.copy()
        negative = new_paulis.phase == 2
        new_paulis[negative] = new_paulis[negative] * -1
        new_values[negative] = new_values[negative] * -1
        return self.__class__(new_paulis, new_values)

    def to_dict(self):
        data = {
            'x': self.paulis.x,
            'z': self.paulis.z,
            'phase': self.paulis.phase,
            'values': self.values
        }
        return data

    def save(self, path):
        np.save(path, self.to_dict())

    @classmethod
    def from_dict(cls, data):
        z = data['z']
        x = data['x']
        phase = data['phase']
        values = data['values']
        paulis = qskQ.PauliList.from_symplectic(z, x, phase)
        return cls(paulis, values)

    @classmethod
    def load(cls, path) -> "Hamiltonian":
        return cls.from_dict(np.load(path, allow_pickle=True).item())
