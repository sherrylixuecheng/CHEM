from scipy.optimize import minimize
from tencirchem import HEA, UCCSD, set_backend
from tencirchem.molecule import h2o,h8, h6, h4, h2, lih
import sys, os
import numpy as np
from pyscf import M

K = set_backend("jax")
gate = 0 # No gate noise

molecule = sys.argv[1]
methods = sys.argv[2]
total_cycle = 1000

if molecule == 'h2o_4e4o':
    uccsd = UCCSD(h2o(), active_space=(4, 4))
    uccsd.print_energy()
elif molecule == 'h4':
    uccsd = UCCSD(h4(), active_space=(2, 3))
    uccsd.print_energy()
    mol = h4()
    # uccsd = UCCSD(mol)
    # uccsd.print_energy()
elif molecule == 'lih':
    mol = lih()
    uccsd = UCCSD(mol)
    uccsd.print_energy()
    hf = mol.HF()
    hf.kernel()
    hf.mo_coeff = hf.mo_coeff[:, [0, 1, 2, 5, 3, 4]]
    uccsd = UCCSD(mol, active_space=(2, 3), mo_coeff=hf.mo_coeff)
    uccsd.print_energy()
elif molecule == 'beh2':
    mol = M(atom=[["H", -1, 0, 0], ["H", 1, 0, 0], ["Be", 0, 0, 0]], charge=0, symmetry=True)
    uccsd = UCCSD(mol)
    uccsd.print_energy()
    # uccsd.kernel()
    # print(uccsd.e_hf, uccsd.e_mp2, uccsd.e_ccsd, uccsd.e_fci, uccsd.energy())
    hf = mol.HF()
    hf.kernel()
    hf.mo_coeff = hf.mo_coeff[:, [0, 1, 2, 5, 6, 3, 4]]
    uccsd = UCCSD(mol, active_space=(4, 4), mo_coeff=hf.mo_coeff)
    uccsd.print_energy()
    # uccsd.kernel()
    # print(uccsd.e_hf, uccsd.e_mp2, uccsd.e_ccsd, uccsd.e_fci, uccsd.energy())
uccsd.print_summary()
#print(uccsd.e_hf, uccsd.e_mp2, uccsd.e_ccsd, uccsd.e_fci, uccsd.energy())

nlayers = int(sys.argv[3])  #1,2,3,4,5
shots = int(sys.argv[4])
trials = int(sys.argv[5])  #0-19
np.random.seed(trials)


hea_nonoise = HEA.ry(uccsd.int1e, uccsd.int2e, uccsd.n_elec, uccsd.e_core, n_layers=nlayers, engine="tensornetwork")
hea_nonoise.grad = "autodiff"
# init_X = np.load('results/nonoise/h4/L-BFGS-B/full_5_0_560.npy', allow_pickle=True).item()['x'].ravel()
# a = hea_nonoise.statevector(init_X)
# print(a.shape)
# print(hea_nonoise)
# np.save('wavefunction_lih_2.npy', a)
# ratios = 2*init_X/np.pi
# print(ratios)
# ratios[0:] = np.round_(2*init_X[0:]/np.pi)
# np.save('wavefunction_h4_full_5_local560.npy', a)
# sys.exit()
# print(ratios)
# f_scipy = hea_nonoise.get_opt_function()
# print(f_scipy(np.zeros(hea_nonoise.init_guess.shape)+np.pi*2))
# hea_nonoise.init_guess = ratios*np.pi/2
# hea_nonoise.kernel()
# hea_nonoise.print_summary()
# print(2*hea_nonoise.params/np.pi)
#sys.exit()

print(hea_nonoise.init_guess.shape)
#init_X = np.load('results/nonoise/h4/L-BFGS-B/full_1_0_239.npy', allow_pickle=True).item()['x'].ravel()
init_X = np.random.uniform(low=0, high=2*np.pi, size=hea_nonoise.init_guess.shape)
#init_Xadd = np.zeros((6,))
#hea_nonoise.init_guess = np.hstack([init_X,init_Xadd]).ravel()
hea_nonoise.init_guess = init_X
print(hea_nonoise.init_guess)
np.save('results/X/{}/x0_{}_{}.npy'.format(molecule, nlayers, trials), hea_nonoise.init_guess)

# f2 = hea_nonoise.get_opt_function()
# res = minimize(f2, init_X, jac=True, method="L-BFGS-B")
hea_nonoise.kernel()
hea_nonoise.print_summary()
print(hea_nonoise.params)
np.save('results/nonoise/{}/{}/{}_{}_{}'.format(molecule, methods, nlayers, 0, trials), {'x': hea_nonoise.params, 'y': float(hea_nonoise.energy())})
sys.exit()


if gate == 0:
    hea = HEA.ry(uccsd.int1e,
                 uccsd.int2e,
                 uccsd.n_elec,
                 uccsd.e_core,
                 n_layers=nlayers,
                 engine="tensornetwork-shot")
else:
    print("Gate+shot noise")
    hea = HEA.ry(uccsd.int1e,
                 uccsd.int2e,
                 uccsd.n_elec,
                 uccsd.e_core,
                 n_layers=nlayers,
                 engine="tensornetwork-noise&shot")
    hea.shots = shots

path = 'results/X/{}/x0_{}_{}.npy'.format(molecule, nlayers, trials)

if os.path.isfile(path) == False:
    init_X = np.random.uniform(low=-np.pi / 8,
                               high=np.pi / 8,
                               size=hea.init_guess.shape)
    np.save('results/X/{}/x0_{}_{}.npy'.format(molecule, nlayers, trials),
            init_X)
    hea.init_guess = init_X
else:
    print('load files')
    hea.init_guess = np.load('results/X/{}/x0_{}_{}.npy'.format(
        molecule, nlayers, trials))
    init_X = hea.init_guess

if methods == "COBYLA" or methods == "SPSA":
    f_scipy = hea.get_opt_function(grad="free")
    print('Done function')
else:
    f_scipy = hea.get_opt_function()

y = []
def new_fscipy(*args):
    global y
    new = f_scipy(*args)
    print(new)
    y.append(new)
    return new


def new_fscipy_nm(*args):
    global y
    new = f_scipy(*args)
    print(new)
    y.append(new)
    return float(new[0])


if methods == 'L-BFGS-B':
    r = minimize(new_fscipy,
                 init_X.ravel(),
                 method=methods,
                 jac=True,
                 options={'maxfun': total_cycle})
else:
    if methods == 'COBYLA':
        r = minimize(new_fscipy,
                     init_X.ravel(),
                     method=methods,
                     options={
                         'maxiter': total_cycle,
                         'tol': 0.0001
                     })
    elif methods == 'SPSA':
        bounds = []
        for i in range(len(init_X.ravel())):
            bounds.append([0, 2 * np.pi])
        bounds = np.array(bounds)
        print(bounds)
        from noisyopt import minimizeSPSA
        r = minimizeSPSA(new_fscipy_nm,
                         bounds=bounds,
                         x0=init_X.ravel(),
                         a=0.01,
                         c=0.01,
                         niter=total_cycle,
                         paired=False)
#print(r)
if gate == 0:
    np.save(
        'results/shot_noise/{}/{}/{}_{}_{}'.format(molecule, methods,
                                                      nlayers, shots,
                                                      trials), y)
else:
    np.save(
        'results/gate+shot_noise/{}/{}/{}_{}_{}'.format(
            molecule, methods, nlayers, shots, trials), y)
