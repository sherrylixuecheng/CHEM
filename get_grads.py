import jax
#jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
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
from libclifford.jaxclifford import PauliArray, CliffordArray
from libclifford import linear_modulo, linear_modulo_jax
import libhea
import libheaclifford
import time

K = set_backend("jax")
gate = 0  # No gate noise

molecule = sys.argv[1]
nlayers = int(sys.argv[2])
trials = int(sys.argv[3])
mode = sys.argv[4]
shots = 0
methods = 'L-BFGS-B'
np.random.seed(trials)

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
        hf = mol.HF()
        hf.kernel()
        hf.mo_coeff = hf.mo_coeff[:, [0, 1, 2, 5, 3, 4]]
        uccsd = UCCSD(mol, active_space=(2, 3), mo_coeff=hf.mo_coeff)
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

#uccsd.print_summary()


add_init = False
n = uccsd.n_qubits - 2
n_elec = uccsd.n_elec
libhea.n = n
libheaclifford.n = n
hop = HEA.ry(uccsd.int1e, uccsd.int2e, uccsd.n_elec, uccsd.e_core, n_layers=nlayers, engine="tensornetwork").h_qubit_op
hamiltonian = Ihamiltonian.Hamiltonian.from_openfermion(hop)
gs = libhea.get_HF_gs(n_elec, uccsd.n_qubits)
gate_gs = [('x', [i]) for i in np.where(gs)[0]]
init_circ = tc.Circuit(n)
for name, indices in gate_gs:
    getattr(init_circ, name)(*indices)

if not add_init:
    clifford_init = qskQ.Clifford.from_circuit(init_circ.to_qiskit())
    hamiltonian = hamiltonian.transform_clifford(clifford_init)
    gs = np.zeros_like(gs)


final_hamiltonian = libheaclifford.simplify_hamiltonian(hamiltonian)
h_pauli = PauliArray.from_qiskit(qskQ.PauliList(list(final_hamiltonian.keys())))
h_values = jnp.array(list(final_hamiltonian.values()))
argsort = jnp.argsort(jnp.abs(h_values))[::-1]
h_pauli = h_pauli[argsort]
h_values = h_values[argsort]
max_score = jnp.linalg.norm(h_values[jnp.any(h_pauli.x, axis=1)]).item()
print('max score:', max_score)
gs = jnp.array(gs)


class SA:
    def __init__(self, func, temperatures, use_Ry=False):
        self.func = func
        self.temperatures = temperatures
        self.niter = len(self.temperatures)
        self.use_Ry = use_Ry
        if use_Ry:
            self.id_max = 4
        else:
            self.id_max = 24
        self.ids = np.random.randint(0, self.id_max, size=(nlayers, n))
        self.nparams = nlayers * n * 3
        self.Ps_order = np.random.permutation(self.nparams)
        self.score = self.func(self.ids, self.Ps_order)
        self.iter = 0
        self.update_best()

    def update_best(self):
        self.best_ids = self.ids
        self.best_Ps_order = self.Ps_order
        self.best_score = self.score

    def get_step_ids(self):
        # replace a random element to a random new value
        qubit_pos = np.random.randint(0, n)
        qubit_layer = np.random.randint(0, nlayers)
        old_value = self.ids[qubit_layer, qubit_pos]
        new_value = np.random.randint(0, self.id_max)
        while new_value == old_value:
            new_value = np.random.randint(0, self.id_max)
        new_ids = self.ids.copy()
        new_ids[qubit_layer, qubit_pos] = new_value
        return new_ids

    def get_step_Ps_order(self):
        a, b = np.random.choice(np.arange(self.nparams), size=(2, ))
        new_Ps_order = self.Ps_order.copy()
        new_Ps_order[[a, b]] = new_Ps_order[[b, a]]
        return new_Ps_order

    def accept(self, score_diff):
        T = self.temperatures[self.iter]
        return np.random.random() > 1 / (np.exp(score_diff / T) + 1)

    def step(self):
        new_ids = self.get_step_ids()
        new_Ps_order = self.get_step_Ps_order()
        new_score = self.func(new_ids, new_Ps_order)
        score_diff = new_score - self.score
        if self.accept(score_diff):
            self.ids = new_ids
            self.Ps_order = new_Ps_order
            self.score = new_score
            if self.score > self.best_score:
                self.update_best()
        self.iter += 1

    def kernel(self, verbose=0):
        for i in range(self.niter):
            self.step()
            if verbose >= 1:
                print(self.temperatures[i], self.score)
            if verbose >= 2:
                print(self.ids)


# Optimize Clifford score
use_Ry = False
id_max = 4 if use_Ry else 24
niter_SA = n * nlayers * 50
temperatures = np.geomspace(5e-2, 2e-3, num=niter_SA, endpoint=False)


def get_heastate(ids):
    return libheaclifford.HEAState(ids, use_Ry=use_Ry)


def get_score(ids, Ps_order):
    heastate = get_heastate(ids)
    transformation, grads, score = heastate.find_best_transformation_reorder(h_pauli, h_values, Ps_order)
    return score.item()


# compile region, only to track the compilation time
data = np.load('results/clifford_autotransform/{}_full/{}.npy'.format(molecule, nlayers), allow_pickle=True).item()
ids = data['ids']
Ps_order = data['Ps_order']
gradient = []

for i in range(len(ids)):
    heastate = get_heastate(ids[i])
    transformation, grads, score = heastate.find_best_transformation_reorder(h_pauli, h_values, Ps_order[i])
    gradient.append(grads)
np.save('results/clifford_autotransform/{}_full/{}_grads.npy'.format(molecule, nlayers), gradient)
