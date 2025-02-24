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
    displacements, reactions = solver.solve()
    
    assert len(displacements) == 18
    assert len(reactions) == 18
    
    # Check that constrained DOFs have zero displacement
    assert np.allclose(displacements[0:6], 0)  # Node 0 is fully constrained
    assert np.isclose(displacements[7], 0)  # Node 1 is constrained in y-direction
    
    # Check that the vertical displacement at the tip is negative
    assert displacements[14] < 0
    
    # Check that the reaction forces at the support sum up to the applied load
    assert np.isclose(np.sum(reactions[2::6]), 10000)

def test_local_stiffness_matrix(simple_frame):
    nodes, elements, _, _ = simple_frame
    solver = FrameSolver(nodes, elements, [], [])
    element = elements[0]
    L = np.linalg.norm(element.node2.coordinates - element.node1.coordinates)
    k_e = solver.local_elastic_stiffness_matrix_3D_beam(
        element.E, element.nu, element.A, L, element.Iy, element.Iz, element.J
    )
    assert k_e.shape == (12, 12)
    assert np.allclose(k_e, k_e.T)  # Stiffness matrix should be symmetric

def test_geometric_stiffness_matrix(simple_frame):
    nodes, elements, _, _ = simple_frame
    solver = FrameSolver(nodes, elements, [], [])
    element = elements[0]
    L = np.linalg.norm(element.node2.coordinates - element.node1.coordinates)
    k_g = solver.local_geometric_stiffness_matrix_3D_beam_without_interaction_terms(
        L, element.A, element.I_rho, 1000  # Assuming Fx2 = 1000 for this test
    )
    assert k_g.shape == (12, 12)
    assert np.allclose(k_g, k_g.T)  # Geometric stiffness matrix should be symmetric

def test_rotation_matrix(simple_frame):
    nodes, elements, _, _ = simple_frame
    solver = FrameSolver(nodes, elements, [], [])
    element = elements[0]
    x1, y1, z1 = element.node1.coordinates
    x2, y2, z2 = element.node2.coordinates
    gamma = solver.rotation_matrix_3D(x1, y1, z1, x2, y2, z2, element.local_z)
    assert gamma.shape == (3, 3)
    assert np.allclose(gamma @ gamma.T, np.eye(3))  # Rotation matrix should be orthogonal

def test_transformation_matrix(simple_frame):
    nodes, elements, _, _ = simple_frame
    solver = FrameSolver(nodes, elements, [], [])
    gamma = np.eye(3)
    T = solver.transformation_matrix_3D(gamma)
    assert T.shape == (12, 12)
    assert np.allclose(T @ T.T, np.eye(12))  # Transformation matrix should be orthogonal

if __name__ == "__main__":
    pytest.main()