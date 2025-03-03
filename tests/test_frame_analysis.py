import pytest
import numpy as np
from frame_solver import Node, Element, Load, BoundaryCondition, FrameSolver

@pytest.fixture
def simple_frame():
    nodes = [
        Node(0, 0, 0, 0),
        Node(1, 0, 0, 3),
        Node(2, 4, 0, 3)
    ]
    elements = [
        Element(0, nodes[0], nodes[1], 200e9, 0.3, 0.01, 1e-4, 1e-4, 2e-4, 2e-4, np.array([0, 1, 0])),
        Element(1, nodes[1], nodes[2], 200e9, 0.3, 0.01, 1e-4, 1e-4, 2e-4, 2e-4, np.array([0, 0, 1]))
    ]
    loads = [
        Load(nodes[2], fz=-10000)
    ]
    bcs = [
        BoundaryCondition(nodes[0], True, True, True, True, True, True),
        BoundaryCondition(nodes[1], False, True, False, False, False, False)
    ]
    return nodes, elements, loads, bcs

def test_node_creation():
    node = Node(0, 1, 2, 3)
    assert node.id == 0
    assert node.x == 1
    assert node.y == 2
    assert node.z == 3
    np.testing.assert_array_equal(node.coordinates, np.array([1, 2, 3]))

def test_element_creation():
    node1 = Node(0, 0, 0, 0)
    node2 = Node(1, 0, 0, 3)
    element = Element(0, node1, node2, 200e9, 0.3, 0.01, 1e-4, 1e-4, 2e-4, 2e-4, np.array([0, 1, 0]))
    assert element.id == 0
    assert element.node1 == node1
    assert element.node2 == node2
    assert element.E == 200e9
    assert element.nu == 0.3
    assert element.A == 0.01
    assert element.Iz == 1e-4
    assert element.Iy == 1e-4
    assert element.J == 2e-4
    assert element.I_rho == 2e-4
    np.testing.assert_array_equal(element.local_z, np.array([0, 1, 0]))

def test_load_creation():
    node = Node(0, 0, 0, 0)
    load = Load(node, fx=1000, fy=2000, fz=3000, mx=100, my=200, mz=300)
    assert load.node == node
    assert load.fx == 1000
    assert load.fy == 2000
    assert load.fz == 3000
    assert load.mx == 100
    assert load.my == 200
    assert load.mz == 300

def test_boundary_condition_creation():
    node = Node(0, 0, 0, 0)
    bc = BoundaryCondition(node, ux=True, uy=True, uz=True, rx=False, ry=False, rz=False)
    assert bc.node == node
    assert bc.ux == True
    assert bc.uy == True
    assert bc.uz == True
    assert bc.rx == False
    assert bc.ry == False
    assert bc.rz == False

def test_frame_solver_initialization(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    assert len(solver.nodes) == 3
    assert len(solver.elements) == 2
    assert len(solver.loads) == 1
    assert len(solver.bcs) == 2
    assert solver.ndof == 18

def test_frame_solver_solution(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    displacements, reactions, critical_load_factor, buckling_mode = solver.solve()
    
    assert len(displacements) == 18
    assert len(reactions) == 18
    assert isinstance(critical_load_factor, float)
    assert len(buckling_mode) == 18
    
    # Check that constrained DOFs have zero displacement
    assert np.allclose(displacements[0:6], 0)  # Node 0 is fully constrained
    assert np.isclose(displacements[7], 0)  # Node 1 is constrained in y-direction
    
    # Check that the vertical displacement at the tip is negative
    assert displacements[14] < 0
    
    # Check that the reaction forces at the support sum up to the applied load
    assert np.isclose(np.sum(reactions[2::6]), 10000)

def test_assemble_global_stiffness_matrix(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    K = solver._assemble_global_stiffness_matrix()
    assert K.shape == (18, 18)
    assert np.allclose(K, K.T)  # Global stiffness matrix should be symmetric

def test_compute_element_stiffness_matrix(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    k = solver._compute_element_stiffness_matrix(elements[0])
    assert k.shape == (12, 12)
    assert np.allclose(k, k.T)  # Element stiffness matrix should be symmetric

def test_compute_transformation_matrix(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    T = solver._compute_transformation_matrix(elements[0])
    assert T.shape == (12, 12)
    assert np.allclose(T @ T.T, np.eye(12))  # Transformation matrix should be orthogonal

def test_get_element_dofs(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    dofs = solver._get_element_dofs(elements[0])
    assert len(dofs) == 12
    assert dofs == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

def test_assemble_load_vector(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    F = solver._assemble_load_vector()
    assert len(F) == 18
    assert F[14] == -10000  # Vertical load at node 2

def test_apply_boundary_conditions(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    K = solver._assemble_global_stiffness_matrix()
    F = solver._assemble_load_vector()
    K_mod, F_mod = solver._apply_boundary_conditions(K, F)
    assert K_mod.shape[0] < K.shape[0]
    assert len(F_mod) < len(F)

def test_recover_full_displacement_vector(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    U_mod = np.ones(11)  # 11 unconstrained DOFs
    U = solver._recover_full_displacement_vector(U_mod)
    assert len(U) == 18
    assert np.allclose(U[0:6], 0)  # First node fully constrained
    assert np.isclose(U[7], 0)  # Second node constrained in y-direction

def test_get_constrained_dofs(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    constrained_dofs = solver._get_constrained_dofs()
    assert len(constrained_dofs) == 7
    assert set(constrained_dofs) == {0, 1, 2, 3, 4, 5, 7}

def test_update_geometric_stiffness(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    U = np.random.rand(18)
    solver._update_geometric_stiffness(U)
    assert elements[0].Fx2 != 0
    assert elements[1].Fx2 != 0

def test_rotation_matrix_3D(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    element = elements[0]
    x1, y1, z1 = element.node1.coordinates
    x2, y2, z2 = element.node2.coordinates
    gamma = solver.rotation_matrix_3D(x1, y1, z1, x2, y2, z2, element.local_z)
    assert gamma.shape == (3, 3)
    assert np.allclose(gamma @ gamma.T, np.eye(3))  # Rotation matrix should be orthogonal

def test_transformation_matrix_3D(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    gamma = np.eye(3)
    T = solver.transformation_matrix_3D(gamma)
    assert T.shape == (12, 12)
    assert np.allclose(T @ T.T, np.eye(12))  # Transformation matrix should be orthogonal

def test_local_elastic_stiffness_matrix_3D_beam(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    element = elements[0]
    L = np.linalg.norm(element.node2.coordinates - element.node1.coordinates)
    k_e = solver.local_elastic_stiffness_matrix_3D_beam(
        element.E, element.nu, element.A, L, element.Iy, element.Iz, element.J
    )
    assert k_e.shape == (12, 12)
    assert np.allclose(k_e, k_e.T)  # Stiffness matrix should be symmetric

def test_local_geometric_stiffness_matrix_3D_beam_without_interaction_terms(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    element = elements[0]
    L = np.linalg.norm(element.node2.coordinates - element.node1.coordinates)
    k_g = solver.local_geometric_stiffness_matrix_3D_beam_without_interaction_terms(
        L, element.A, element.I_rho, 1000  # Assuming Fx2 = 1000 for this test
    )
    assert k_g.shape == (12, 12)
    assert np.allclose(k_g, k_g.T)  # Geometric stiffness matrix should be symmetric

def test_assemble_global_geometric_stiffness_matrix(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    K_g = solver._assemble_global_geometric_stiffness_matrix()
    assert K_g.shape == (18, 18)
    assert np.allclose(K_g, K_g.T)  # Global geometric stiffness matrix should be symmetric

def test_compute_member_forces(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    U = np.random.rand(18)
    forces = solver.compute_member_forces(elements[0], U)
    assert len(forces) == 12

def test_plot_member_forces(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    forces = np.random.rand(12)
    solver.plot_member_forces(elements[0], forces)
    # This test just checks if the method runs without errors

def test_plot_deformed_shape(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    U = np.random.rand(18)
    solver.plot_deformed_shape(U)
    # This test just checks if the method runs without errors

if __name__ == "__main__":
    pytest.main()