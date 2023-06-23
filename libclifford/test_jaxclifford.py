import qiskit.quantum_info as qskQ
import jax.numpy as jnp
import time
from libclifford.jaxclifford import PauliArray, CliffordArray

n = 3
ps = qskQ.PauliList([qskQ.random_pauli(n, group_phase=True) for i in range(20)])
cliff = qskQ.random_clifford(n)
cliff_array = CliffordArray.from_qiskit(cliff)
pauli_cliff = qskQ.PauliList.from_symplectic(cliff.z, cliff.x, cliff.phase * 2)
_phase = PauliArray.get_inner_phase(cliff.x, cliff.z, cliff.phase * 2)
pauli_cliff_array = PauliArray(cliff.x, cliff.z, _phase)

ps_array = PauliArray.from_qiskit(ps)
ps_evolve = ps.evolve(cliff)
ps_array_evolve = ps_array.reshape((4, 5)).evolve(cliff_array).reshape((-1, ))
assert PauliArray.from_qiskit(ps_evolve) == ps_array_evolve
ps_product = ps_array[:, None].dot(ps_array[None, :])
for i in range(20):
    assert ps_product[:, i] == PauliArray.from_qiskit(ps.dot(ps[i]))

cliff1 = qskQ.random_clifford(n)
cliff2 = qskQ.random_clifford(n)
cliff1_array = CliffordArray.from_qiskit(cliff1)
cliff2_array = CliffordArray.from_qiskit(cliff2)
cliff = cliff1.dot(cliff2)
cliff_array = cliff1_array.dot(cliff2_array)
assert jnp.all(cliff.symplectic_matrix == cliff_array.symplectic_matrix)
assert jnp.all(cliff.phase == cliff_array.phase)
assert jnp.all(cliff.x == cliff_array.x)
assert jnp.all(cliff.z == cliff_array.z)
assert jnp.all(cliff.stab_x == cliff_array.stab_x)
assert jnp.all(cliff.stab_z == cliff_array.stab_z)
assert jnp.all(cliff.destab_x == cliff_array.destab_x)
assert jnp.all(cliff.destab_z == cliff_array.destab_z)
assert CliffordArray.from_pauli(cliff_array.to_pauli()) == cliff_array
assert CliffordArray.dot(cliff1_array, cliff2_array) == cliff1_array.dot(cliff2_array)
XYZ = cliff_array.transform_XYZ().evolve(cliff_array.adjoint())
assert jnp.all(XYZ[0].x == jnp.eye(n))
assert jnp.all(XYZ[0].z == jnp.zeros((n, n), dtype=int))

n_big = 5
subsystem = jnp.arange(n)
ps_big = qskQ.PauliList([qskQ.random_pauli(n_big, group_phase=True) for i in range(10)])
ps_big_array = PauliArray.from_qiskit(ps_big)
cliff_big_array = CliffordArray.from_subspace(cliff_array, jnp.arange(n), n_big)
assert ps_big_array.evolve(cliff_big_array).get_subsystem(subsystem) == ps_big_array.get_subsystem(subsystem).evolve(cliff_array)
sub = ps_big_array.get_subsystem(subsystem).evolve(cliff_array)
assert ps_big_array.evolve_subsystem(cliff_array, qargs=subsystem).to_qiskit() == ps_big.evolve(cliff, qargs=subsystem.tolist())
cliff_big_array2 = CliffordArray.from_qiskit(qskQ.random_clifford(n_big))
assert cliff_big_array.dot(cliff_big_array2) == cliff_array.dot_subsystem(cliff_big_array2, jnp.arange(n))

cliff_list = [qskQ.random_clifford(n) for i in range(20)]
cliff_list_array = CliffordArray.from_qiskit(cliff_list)

result_array1 = cliff_list_array.dot(cliff_list_array[0:1])
result_array2 = cliff_list_array.dot(cliff_list_array)
result_array3 = cliff_list_array[0:1].dot(cliff_list_array)
result_list1 = CliffordArray.from_qiskit([item.dot(cliff_list[0]) for item in cliff_list])
result_list2 = CliffordArray.from_qiskit([item.dot(item) for item in cliff_list])
result_list3 = CliffordArray.from_qiskit([cliff_list[0].dot(item) for item in cliff_list])
assert result_array1 == result_list1
assert result_array2 == result_list2
assert result_array3 == result_list3
