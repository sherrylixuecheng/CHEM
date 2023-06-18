import jax
from tencirchem import HEA, UCCSD, set_backend
from tencirchem.molecule import h2o, h8, h6, h4, h2, lih
import tensorcircuit as tc
import sys
import os
import numpy as np
import qiskit.quantum_info as qskQ
from pyscf import M
from libclifford import clifford as Iclifford
from libclifford import hamiltonian as Ihamiltonian
from functools import reduce

K = set_backend("jax")
gate = 0  # No gate noise
molecule = sys.argv[1]
nlayers = int(sys.argv[2])
trials = int(sys.argv[3])
mode = sys.argv[4]
shots = 0
methods = 'L-BFGS-B'

if molecule == 'h2o_4e4o':
    uccsd = UCCSD(h2o(), active_space=(4, 4))
    uccsd.print_energy()
elif molecule == 'h4':
    if mode == 'full':
        mol = h4()
        uccsd = UCCSD(mol)
        uccsd.print_energy()
    else:
        uccsd = UCCSD(h4(), active_space=(2, 3))
        uccsd.print_energy()
elif molecule == 'lih':
    if mode == 'full':
        uccsd = UCCSD(lih())
        uccsd.print_energy()
    else:
        hf = lih().HF()
        hf.kernel()
        hf.mo_coeff = hf.mo_coeff[:, [0, 1, 2, 5, 3, 4]]
        uccsd = UCCSD(lih(), active_space=(2, 3), mo_coeff=hf.mo_coeff)
        uccsd.print_energy()
elif molecule == 'beh2':
    mol = M(atom=[["H", -1, 0, 0], ["H", 1, 0, 0], ["Be", 0, 0, 0]], charge=0, symmetry=True)
    if mode == 'full':
        uccsd = UCCSD(mol)
        uccsd.print_energy()
    else:
        hf = mol.HF()
        hf.kernel()
        hf.mo_coeff = hf.mo_coeff[:, [0, 1, 2, 5, 6, 3, 4]]
        uccsd = UCCSD(mol, active_space=(4, 4), mo_coeff=hf.mo_coeff)
        uccsd.print_energy()

uccsd.print_summary()


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


n = uccsd.n_qubits - 2
n_elec = uccsd.n_elec
hop = HEA.ry(uccsd.int1e, uccsd.int2e, uccsd.n_elec, uccsd.e_core, n_layers=nlayers, engine="tensornetwork").h_qubit_op
hamiltonian = Ihamiltonian.Hamiltonian.from_openfermion(hop)
gs = get_HF_gs(n_elec, uccsd.n_qubits)
gate_gs = [('x', [i]) for i in np.where(gs)[0]]
init_circ = tc.Circuit(n)
for name, indices in gate_gs:
    getattr(init_circ, name)(*indices)


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


def get_circuit(params, ids, add_init=True, add_reverse=True) -> tc.Circuit:
    t = len(ids)
    assert ids.shape == (t, n)
    params = params.reshape((t, 3, n))
    if add_init:
        circ = tc.Circuit.from_qir(init_circ.to_qir(), init_circ.circuit_param)
    else:
        circ = tc.Circuit(n)
    for i in range(t):
        circ.append(get_circ_cliff1(ids[i]))
        circ.append(get_circ_cnot())
        circ.append(get_circ_XYZ(params[i]))
    if add_reverse:
        for i in range(t)[::-1]:
            circ.append(get_circ_cnot(reverse=True))
            circ.append(get_circ_cliff1(ids[i], reverse=True))
    return circ


def get_HEA(ids, hop, add_init=True, add_reverse=True):
    init_guess = tc.backend.convert_to_tensor(np.zeros(ids.shape + (3, )).flatten())
    get_circ = lambda params: get_circuit(params, ids, add_init=add_init, add_reverse=add_reverse)
    hea = HEA(hop, get_circ, init_guess)
    return hea


def get_clifford_cliff1(ids):
    return qskQ.Clifford.from_circuit(get_circ_cliff1(ids).to_qiskit())


clifford_cnot_layer = qskQ.Clifford.from_circuit(get_circ_cnot().to_qiskit())


class HEAClifford:
    def __init__(self, cnot_layer: qskQ.Clifford, hamiltonian: Ihamiltonian.Hamiltonian, gs, ids=None):
        self.cnot_layer = cnot_layer
        self.hamiltonian = hamiltonian
        self.gs = gs
        self.n = self.cnot_layer.num_qubits
        self.clifford = Iclifford.identity_clifford(self.n)
        self.ids = np.zeros((0, self.n), dtype=int)
        self.Ps_transform = []
        self.grads = np.array([], dtype=float)
        self.score = 0.0
        if ids is not None:
            for id_layer in ids:
                self.add_new_layer(id_layer, True)

    def new_score(self, new_clifford, apply=False):
        new_Ps_transform = Ihamiltonian.transform_XYZ(new_clifford)
        new_grads = Ihamiltonian.paulis_gradient(qskQ.PauliList(new_Ps_transform), self.hamiltonian, self.gs)
        Ps_transform_all = self.Ps_transform + new_Ps_transform
        grads_all = np.concatenate([self.grads, new_grads])
        overlap_all = Ihamiltonian.paulis_overlap(qskQ.PauliList(Ps_transform_all), gs)
        grads_orth = Ihamiltonian.orthogonalize(grads_all, overlap_all)
        score = np.linalg.norm(grads_orth)
        if apply:
            self.clifford = new_clifford
            self.Ps_transform = Ps_transform_all
            self.grads = grads_all
            self.score = score
            return None
        else:
            return score

    def add_new_layer(self, id_layer, apply=False):
        cliff1s = [Iclifford.expand_clifford(self.n, Iclifford.clifford1[id], [site]) for site, id in enumerate(id_layer)]
        cliff1 = reduce(qskQ.Clifford.dot, cliff1s)
        new_clifford = self.cnot_layer.dot(cliff1).dot(self.clifford)
        if apply:
            self.ids = np.concatenate([self.ids, id_layer[None, :]])
        return self.new_score(new_clifford, apply=apply)


# Optimize Clifford score
from copy import copy, deepcopy
np.random.seed(trials)
heacliff = HEAClifford(clifford_cnot_layer, hamiltonian, gs)
for i in range(nlayers):
    maxid = np.random.randint(0,24, size=(n,))
    scoremax = heacliff.add_new_layer(maxid, apply=False)
    print(i, scoremax)
    for j in range(n):
        temp = copy(maxid)
        scoretemp =[]
        for k in range(24):
            temp[j] = k
            scoretemp.append(heacliff.add_new_layer(temp, apply=False))
        shuffleorder = np.arange(24)
        np.random.shuffle(shuffleorder)
        # print(scoretemp)
        # print(shuffleorder)
        scoretemp = np.array(scoretemp)[shuffleorder]
#        print(scoretemp)
        loc = np.argsort(scoretemp)[-1]
 #       print(loc)
        maxid[j] = shuffleorder[loc]
        print(heacliff.add_new_layer(maxid, apply=False),max(scoretemp))
            # if scoretemp >scoremax:
            #     print('yes',scoretemp, scoremax, maxid, temp)
            #     maxid = copy(temp)
            #     scoremax = scoretemp
            # if scoretemp == scoremax:
            #     randint = np.random.randint(0,1)
#        print(maxid)
    heacliff.add_new_layer(maxid, apply=False)
    score1 = heacliff.score
    heacliff.add_new_layer(maxid, apply=True)
    score2 = heacliff.score
    print(score1, score2)

# Real optimization
ids = heacliff.ids # save
score = heacliff.score
print(ids, score)
clifford_inv = heacliff.clifford.adjoint()
hamiltonian_transform = hamiltonian.transform_clifford(clifford_inv)
hop_transform = hamiltonian_transform.to_openfermion()
hea_transform = get_HEA(ids, hop_transform, add_reverse=False)
energy_opt = hea_transform.kernel()
print(energy_opt)
if mode == 'full':
    np.save('results/clifford_init/{}_full/{}/{}_{}'.format(molecule, methods, nlayers, trials), {'x_clifford': ids, 'score_clifford':score, 'grad': heacliff.grads, 'x': hea_transform.params, 'y': float(hea_transform.energy())})
else:
    np.save('results/clifford_init/{}/{}/{}_{}'.format(molecule, methods, nlayers, trials), {'x_clifford': ids, 'score_clifford':score, 'grad': heacliff.grads, 'x': hea_transform.params, 'y': float(hea_transform.energy())})


sanity_check = False
if sanity_check:
    energy_clifford = Ihamiltonian.get_energy_clifford(hamiltonian, gs)
    grad_clifford = heacliff.grads.reshape((nlayers, 3, n))
    print('hf err', uccsd.e_hf - energy_clifford)

    hea_full = get_HEA(ids, hop, add_init=True)
    with jax.disable_jit():
        energy_ref, grad_ref = hea_full.energy_and_grad(hea_full.init_guess)
    grad_ref = grad_ref.reshape((-1, 3, n))
    err = np.max(np.abs(grad_ref - grad_clifford))
    print('energy err', energy_ref - energy_clifford)
    print('grad err', err)

    with jax.disable_jit():
        energy_transform, grad_transform = hea_transform.energy_and_grad(hea_transform.init_guess)
    grad_transform = grad_transform.reshape((-1, 3, n))
    err = np.max(np.abs(grad_transform - grad_clifford))
    print('energy err', energy_transform - energy_clifford)
    print('grad err', err)
