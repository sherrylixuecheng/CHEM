import numpy as np
import typing as tp
from functools import reduce
import qiskit as qsk
import qiskit.quantum_info as qskQ
from . import linear_modulo
from . import utils

'''
some name conventions:
term 'clifford': qskQ.Clifford object
term 'gate': a 2-tuple of (gate_name, indices), where gate_name is either 'm' (measurement) or one element of `gate_list` defined below
term 'Cgate': a 2-tuple of (U, indices), where U is either 'm' (measurement) or a qskQ.Clifford object
temm 'gates/Cgates': list of 'gate/Cgate'
'''


def apply_circuit_qiskit(c, gates):
    current_classical = 0
    for gate, indices in gates:
        if gate == 'm':
            c.measure(indices, [current_classical])
            current_classical += 1
        else:
            getattr(c, gate)(*indices)


def construct_circuit_qiskit(gates, n=None):
    if n is None:
        n = utils.get_max_n(gates)
    n_measure = utils.count_gates(gates, 'm')
    c = qsk.QuantumCircuit(qsk.QuantumRegister(n), qsk.ClassicalRegister(n_measure))
    apply_circuit_qiskit(c, gates)
    return c


def gate2clifford(gate, indices, n):
    c = qsk.QuantumCircuit(qsk.QuantumRegister(n))
    apply_circuit_qiskit(c, [(gate, indices)])
    return qskQ.Clifford.from_circuit(c)


single_gates = ['i', 'h', 's', 'x', 'y', 'z', 'sdg', 'sx', 'sxdg', 'sy', 'sydg']
double_gates = ['cx', 'swap']
gate_list = single_gates + double_gates
conjugate_map = {
    'i': 'i',
    'h': 'h',
    's': 'sdg',
    'x': 'x',
    'y': 'y',
    'z': 'z',
    'sdg': 's',
    'sx': 'sxdg',
    'sxdg': 'sx',
    'sy': 'sydg',
    'sydg': 'sy',
    'cx': 'cx',
    'swap': 'swap',
}
pauli_measure_map = {
    'x': ['h'],
    'y': ['sdg', 'h'],
    'z': [],
}
clifford_map = {}
for gate in single_gates:
    if gate not in ['sy', 'sxdg', 'sydg']:
        clifford_map[gate] = gate2clifford(gate, [0], 1)
for gate in double_gates:
    clifford_map[gate] = gate2clifford(gate, [0, 1], 2)
clifford_map['sy'] = reduce(qskQ.Clifford.dot, [clifford_map[name] for name in ['s', 'sx', 'sdg']])
clifford_map['sxdg'] = reduce(qskQ.Clifford.dot, [clifford_map[name] for name in ['x', 'sx']])
clifford_map['sydg'] = reduce(qskQ.Clifford.dot, [clifford_map[name] for name in ['y', 'sy']])


def identity_clifford(n) -> qskQ.Clifford:
    c = qsk.QuantumCircuit(qsk.QuantumRegister(n))
    return qskQ.Clifford.from_circuit(c)


def expand_clifford(n, U: qskQ.Clifford, indices):
    m = len(indices)
    assert m == U.num_qubits
    indices = np.array(indices)
    indices = np.concatenate([indices, indices + n])
    result = identity_clifford(n)
    utils.change_matrix(result.table.array, indices, indices, U.table.array)
    result.table.phase[indices] = U.table.phase.copy()
    return result


def gates2clifford(gates, n):
    U = identity_clifford(n)
    for name, indices in gates:
        U = expand_clifford(n, clifford_map[name], indices).dot(U)
    return U


def gates2cliffords(gates, n):
    Us = []
    U = identity_clifford(n)
    for name, indices in gates:
        U = expand_clifford(n, clifford_map[name], indices).dot(U)
        Us.append(U)
    return Us


def circuit2gates(c):
    convert = lambda name: 'm' if name == 'measure' else name
    return [(convert(gate[0].name), [i.index for i in gate[1]]) for gate in c.data]


def clifford2gates(U):
    return circuit2gates(qskQ.decompose_clifford(U))


def Cgate2gates(U, indices):
    indices = np.array(indices)
    gates = clifford2gates(U)
    return [(gatename, indices[gate_indices].tolist()) for gatename, gate_indices in gates]


def Cgates2gates(Cgates):
    gates = []
    for U, indices in Cgates:
        if U == 'm':
            gates.append((U, indices))
        else:
            assert isinstance(U, qskQ.Clifford)
            gates.extend(Cgate2gates(U, indices))
    return gates


def gates2Cgates(gates):
    Cgates = []
    for name, indices in gates:
        if name == 'm':
            Cgates.append((name, indices))
        else:
            assert name in gate_list
            clifford = clifford_map[name]
            Cgates.append((clifford, indices))
    return Cgates


def decompose_paulimeasure(P: qskQ.Pauli):
    # decompose a multi-qubit pauli measurement to simple gates
    # example: XIYIZ --> ZIZIZ --> IIZII
    operations = []
    n = len(P)
    string = str(P)
    if string[0] not in ['+', '-']:
        string = '+' + string
    sign = string[0]
    string = list(string[1:][::-1])
    for i in range(n):
        char = string[i]
        if char == 'I':
            continue
        elif char in ['X', 'Y', 'Z']:
            gates = pauli_measure_map[char.lower()]
            operations.extend([(gate, [i]) for gate in gates])
            string[i] = 'Z'
        else:
            assert False
    Z_indices = np.where([char == 'Z' for char in string])[0]
    nZ = len(Z_indices)
    begin = Z_indices[0]
    end = Z_indices[-1]
    assert nZ > 0
    for i in range(nZ - 1):
        a, b = Z_indices[i], Z_indices[i + 1]
        for j in range(a, b - 1):
            operations.append(('cx', [j + 1, j]))
    for i in range(begin, end):
        operations.append(('cx', [i, i + 1]))
    if sign == '-':
        operations.append(('x', [end]))
    operations_conjugate = [(conjugate_map[gate], indices) for gate, indices in operations[::-1]]
    return operations + [('m', [end])] + operations_conjugate


def single_qubit_paulistring(n, site, char, sign='+'):
    assert char in ['X', 'Y', 'Z']
    string = ['I'] * n
    string[site] = char
    string = [sign] + string[::-1]
    string = ''.join(string)
    return qskQ.Pauli(string)


def identity_pauli(n):
    if n == 0:
        return qskQ.Pauli(([], []))
    return qskQ.Pauli(''.join(['I'] * n))


def all_paulis(n, sign=True):
    p = np.array(['I', 'X', 'Y', 'Z'])
    if sign:
        ps = np.array(np.meshgrid(['+', '-'], *([p] * n))).reshape((n+1, -1))
    else:
        ps = np.array(np.meshgrid(*([p] * n))).reshape((n, -1))
    Ps = reduce(np.char.add, ps)
    return Ps


def decompose_pauli_product(Ps: qskQ.PauliList, P: qskQ.Pauli):
    # Given commuting Pauli operators Ps, try to decompose P as the product of some of them up to a global sign
    get_array = lambda p: np.concatenate([p.x, p.z], axis=-1).astype(int)
    A = get_array(Ps).T
    b = get_array(P)
    I = identity_pauli(P.num_qubits)
    if len(Ps) == 0:
        zero = np.array([], dtype=bool)
        if P == I:
            return zero, 1
        elif P == -I:
            return zero, -1
        else:
            return None
    try:
        x = linear_modulo.solve_modN(A, b, 2)
    except linear_modulo.OverDetermined:
        return None
    except linear_modulo.UnderDetermined as e:
        raise e
    Ps1 = [p for order, p in zip(x.tolist(), Ps) if order == 1]
    product = reduce(lambda x, y: x.dot(y), Ps1, I)
    if product == P:
        sign = 1
    elif product == -P:
        sign = -1
    else:
        assert False
    return x.astype(bool), sign


def compress_clifford(gates):
    n = utils.get_max_n(gates)
    m_position = [index for index, gate in enumerate(gates) if gate[0] == 'm']
    start = 0
    new_gates = []
    for index_m in m_position:
        clifford = gates2clifford(gates[start:index_m], n)
        new_gates.extend(circuit2gates(qskQ.decompose_clifford(clifford)))
        new_gates.append(gates[index_m])
        start = index_m + 1
    clifford = gates2clifford(gates[start:], n)
    new_gates.extend(circuit2gates(qskQ.decompose_clifford(clifford)))
    return new_gates


def labels2pauli(labels):
    n = len(labels) // 2
    d = {
        'stabilizer': labels[n:],
        'destabilizer': labels[:n]
    }
    return qskQ.Clifford.from_dict(d)


XYZ = np.array(['X', 'Y', 'Z'])
XYZ2 = np.array(np.meshgrid(XYZ, XYZ)).reshape((2, -1)).T
XYZ2 = XYZ2[XYZ2[:, 0] != XYZ2[:, 1]]
sign = np.array(['+', '-'])
sign2 = np.array(np.meshgrid(sign, sign)).reshape((2, -1)).T
labels = np.char.add(sign2[:, None, :], XYZ2[None, :, :]).reshape((-1, 2))
clifford1 = [labels2pauli(l) for l in labels.tolist()]
clifford1_gates = [[item[0] for item in circuit2gates(qskQ.decompose_clifford(c))] for c in clifford1]
