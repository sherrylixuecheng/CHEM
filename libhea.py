import numpy as np
import tensorcircuit as tc
from libclifford import clifford as Iclifford
from tencirchem import HEA
n = 0


def get_HF_gs(nelec, nqubits):
    assert nelec % 2 == 0
    assert nqubits % 2 == 0
    ne = nelec // 2
    nq = nqubits // 2
    original_gs = np.zeros(nq, dtype=int)
    original_gs[-ne:] = 1
    state1 = np.concatenate([original_gs, original_gs])
    state2 = np.cumsum(state1) % 2
    state3 = np.concatenate([state2[0: nq - 1], state2[nq: nq * 2 - 1]])
    return state3


def get_circ_cliff1(ids, reverse=False):
    assert len(ids) == n
    gates = sum([[(g, [i]) for g in Iclifford.clifford1_gates[id]] for i, id in enumerate(ids)], [])
    if reverse:
        gates = [(Iclifford.conjugate_map[g], i) for g, i in gates[::-1]]
    circ = tc.Circuit.from_qiskit(Iclifford.construct_circuit_qiskit(gates, n=n))
    circ.barrier_instruction(list(range(n)))
    return circ


def get_circ_XYZ(params):
    assert params.shape == (3, n)
    circ = tc.Circuit(n)
    for i in range(n):
        circ.rx(i, theta=params[0, i])
        circ.ry(i, theta=params[1, i])
        circ.rz(i, theta=params[2, i])
    circ.barrier_instruction(list(range(n)))
    return circ


def get_circ_cnot(reverse=False):
    circ = tc.Circuit(n)
    qubits = range(n - 1)
    if reverse:
        qubits = qubits[::-1]
    for i in qubits:
        circ.cnot(i, (i + 1))
    circ.barrier_instruction(list(range(n)))
    return circ


def get_circuit(params, ids, init_circ=None, add_reverse=True) -> tc.Circuit:
    t = len(ids)
    assert ids.shape == (t, n)
    params = params.reshape((t, 3, n))
    circ = tc.Circuit(n)
    for i in range(t):
        circ.append(get_circ_cliff1(ids[i]))
        circ.append(get_circ_cnot())
        circ.append(get_circ_XYZ(params[i]))
    if add_reverse:
        for i in range(t)[::-1]:
            circ.append(get_circ_cnot(reverse=True))
            circ.append(get_circ_cliff1(ids[i], reverse=True))
    if init_circ is not None:
        circ.append(tc.Circuit.from_qir(init_circ.to_qir(), init_circ.circuit_param))
    return circ


def get_HEA(ids, hop, init_circ=None, add_reverse=True, engine="tensornetwork"):
    get_circ = lambda params: get_circuit(params, ids, init_circ=init_circ, add_reverse=add_reverse)
    init_guess = tc.backend.convert_to_tensor(np.zeros(ids.shape + (3, )).flatten())
    hea = HEA(hop, get_circ, init_guess, engine=engine)
    return hea
