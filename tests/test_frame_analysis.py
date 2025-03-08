import pytest
import numpy as np
from frame_solver import Node, Element, Load, BoundaryCondition, FrameSolver
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from unittest.mock import patch
@pytest.fixture
def simple_frame():
    nodes = [
        Node(0, 0, 0, 0),
        Node(1, 0, 0, 3),
        Node(2, 3, 0, 3)
    ]
    elements = [
        Element(0, nodes[0], nodes[1], 200e9, 0.3, 0.01, 1e-4, 1e-4, 2e-4, 2e-4, np.array([0, 1, 0])),
        Element(1, nodes[1], nodes[2], 200e9, 0.3, 0.01, 1e-4, 1e-4, 2e-4, 2e-4, np.array([0, 1, 0]))
    ]
    loads = [Load(nodes[2], fx=-1000)]
    bcs = [
        BoundaryCondition(nodes[0], ux=True, uy=True, uz=True, rx=True, ry=True, rz=True),
        BoundaryCondition(nodes[1], uy=True)
    ]
    return nodes, elements, loads, bcs

def test_node_properties():
    node = Node(1, 2, 3, 4)
    assert node.id == 1
    assert node.x == 2
    assert node.y == 3
    assert node.z == 4
    np.testing.assert_array_equal(node.coordinates, np.array([2, 3, 4]))

def test_element_properties():
    node1 = Node(0, 0, 0, 0)
    node2 = Node(1, 1, 1, 1)
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

def test_load_properties():
    node = Node(0, 0, 0, 0)
    load = Load(node, fx=1, fy=2, fz=3, mx=4, my=5, mz=6)
    assert load.node == node
    assert load.fx == 1
    assert load.fy == 2
    assert load.fz == 3
    assert load.mx == 4
    assert load.my == 5
    assert load.mz == 6

def test_boundary_condition_properties():
    node = Node(0, 0, 0, 0)
    bc = BoundaryCondition(node, ux=True, uy=False, uz=True, rx=False, ry=True, rz=False)
    assert bc.node == node
    assert bc.ux == True
    assert bc.uy == False
    assert bc.uz == True
    assert bc.rx == False
    assert bc.ry == True
    assert bc.rz == False

def test_frame_solver_initialization(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    assert solver.nodes == nodes
    assert solver.elements == elements
    assert solver.loads == loads
    assert solver.bcs == bcs
    assert solver.ndof == len(nodes) * 6
    assert solver.kg_included == False

def test_frame_solver_solve(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    U, R, critical_load_factor, buckling_mode = solver.solve()
    assert U.shape == (len(nodes) * 6,)
    assert R.shape == (len(nodes) * 6,)
    assert isinstance(critical_load_factor, float)
    assert buckling_mode.shape == (len(nodes) * 6,)


def test_frame_solver_compute_element_stiffness_matrix(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    k = solver._compute_element_stiffness_matrix(elements[0])
    assert k.shape == (12, 12)
    assert np.allclose(k, k.T)  # Check symmetry

def test_frame_solver_compute_transformation_matrix(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    T = solver._compute_transformation_matrix(elements[0])
    assert T.shape == (12, 12)
    assert np.allclose(T @ T.T, np.eye(12))  # Check orthogonality

def test_frame_solver_get_element_dofs(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    dofs = solver._get_element_dofs(elements[0])
    assert len(dofs) == 12
    assert all(0 <= dof < solver.ndof for dof in dofs)

def test_frame_solver_assemble_load_vector(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    F = solver._assemble_load_vector()
    assert F.shape == (len(nodes) * 6,)
    assert np.sum(F) == -1000  # Sum of loads should equal the applied load

def test_frame_solver_apply_boundary_conditions(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    K = solver._assemble_global_stiffness_matrix()
    F = solver._assemble_load_vector()
    K_mod, F_mod = solver._apply_boundary_conditions(K, F)
    assert K_mod.shape[0] < K.shape[0]
    assert F_mod.shape[0] < F.shape[0]

def test_frame_solver_recover_full_displacement_vector(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    U_mod = np.ones(len(nodes) * 6 - len(solver._get_constrained_dofs()))
    U = solver._recover_full_displacement_vector(U_mod)
    assert U.shape == (len(nodes) * 6,)
    assert np.all(U[solver._get_constrained_dofs()] == 0)

def test_frame_solver_update_geometric_stiffness(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    U = np.random.rand(len(nodes) * 6) * 0.01
    initial_Fx2 = [element.Fx2 for element in elements]
    solver._update_geometric_stiffness(U)
    final_Fx2 = [element.Fx2 for element in elements]
    assert any(initial != final for initial, final in zip(initial_Fx2, final_Fx2))

def test_frame_solver_rotation_matrix_3D(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    gamma = solver.rotation_matrix_3D(0, 0, 0, 1, 1, 1)
    assert gamma.shape == (3, 3)
    assert np.allclose(gamma @ gamma.T, np.eye(3))  # Check orthogonality

def test_frame_solver_transformation_matrix_3D(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    gamma = np.eye(3)
    T = solver.transformation_matrix_3D(gamma)
    assert T.shape == (12, 12)
    assert np.allclose(T @ T.T, np.eye(12))  # Check orthogonality

def test_frame_solver_local_elastic_stiffness_matrix_3D_beam(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    k_e = solver.local_elastic_stiffness_matrix_3D_beam(200e9, 0.3, 0.01, 3, 1e-4, 1e-4, 2e-4)
    assert k_e.shape == (12, 12)
    assert np.allclose(k_e, k_e.T)  # Check symmetry


def test_frame_solver_local_geometric_stiffness_matrix_3D_beam(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    k_g = solver.local_geometric_stiffness_matrix_3D_beam(3, 0.01, 2e-4, 1000, 100, 200, 300, 400, 500)
    assert k_g.shape == (12, 12)
    assert np.allclose(k_g, k_g.T)  # Check symmetry

def test_frame_solver_assemble_global_geometric_stiffness_matrix(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    U = np.ones(len(nodes) * 6)
    K_geo = solver._assemble_global_geometric_stiffness_matrix(U)
    assert K_geo.shape == (len(nodes) * 6, len(nodes) * 6)
    assert np.allclose(K_geo, K_geo.T)  # Check symmetry

def test_frame_solver_get_free_dofs(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    free_dofs = solver._get_free_dofs()
    assert len(free_dofs) < len(nodes) * 6
    assert all(0 <= dof < solver.ndof for dof in free_dofs)

def test_frame_solver_compute_member_forces(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    U = np.ones(len(nodes) * 6)
    forces = solver.compute_member_forces(elements[0], U)
    assert forces.shape == (12,)

def test_critical_load_factor_cantilever_beam():
    L = 3.0
    E = 200e9
    I = 1e-6
    A = 1e-4

    nodes = [Node(0, 0, 0, 0), Node(1, 0, 0, L)]
    elements = [Element(0, nodes[0], nodes[1], E, 0.3, A, I, I, 2*I, 2*I, np.array([0, 1, 0]))]
    loads = [Load(nodes[1], fy=-1000)]
    bcs = [BoundaryCondition(nodes[0], ux=True, uy=True, uz=True, rx=True, ry=True, rz=True)]
    
    solver = FrameSolver(nodes, elements, loads, bcs)
    U, R, critical_load_factor, buckling_mode = solver.solve()
    
    P_cr_theoretical = (np.pi**2 * E * I) / (4 * L**2)
    expected_critical_load_factor = P_cr_theoretical / 1000
    
    np.testing.assert_allclose(critical_load_factor, expected_critical_load_factor, rtol=1e-2)

def test_solve_with_geometric_stiffness(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    solver.kg_included = True
    U, R, critical_load_factor, buckling_mode = solver.solve()
    assert U.shape == (len(nodes) * 6,)
    assert R.shape == (len(nodes) * 6,)
    assert isinstance(critical_load_factor, float)
    assert buckling_mode.shape == (len(nodes) * 6,)

def test_solve_critical_buckling_load(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    U = np.random.rand(len(nodes) * 6) * 0.01
    critical_load_factor, buckling_mode = solver.solve_critical_buckling_load(U)
    assert isinstance(critical_load_factor, float)
    assert buckling_mode.shape == (len(nodes) * 6,)

def test_critical_load_factor_cantilever_beam():
    """Solve and analyze a simple 3D beam problem."""
    nodes = [
        Node(0, 0, 0, 0),
        Node(1, 30, 40, 0),
    ]

    r, E, nu = 1, 1000, 0.3
    A = np.pi * r**2.0
    Iy = Iz = np.pi * (r**4.0) / 4.0
    I_rho = np.pi * (r**4) / 2
    J = np.pi * r**4.0 / 2.0

    elements = [Element(0, nodes[0], nodes[1], E, nu, A, Iz, Iy, J, I_rho, None)]

    loads = [Load(nodes[1], fx=-3/5, fy=-4/5, fz=0, mx=0, my=0, mz=0)]

    bcs = [
        BoundaryCondition(nodes[0], True, True, True, True, True, True),
        BoundaryCondition(nodes[1], False, False, False, False, False, False),
    ]

    solver = FrameSolver(nodes, elements, loads, bcs)
    displacements, reactions, critical_load_factor, buckling_mode = solver.solve()

    
    # Calculate the expected critical load factor
    # For a cantilever beam, the theoretical critical load is:
    # P_cr = (pi^2 * E * I) / (4 * L^2)
    expected_critical_load_factor = 0.7751
    
    
    # Check if the computed critical load factor matches the expected value
    np.testing.assert_allclose(critical_load_factor, expected_critical_load_factor, rtol=1e-2)
    
    print(f"Computed critical load factor: {critical_load_factor}")
    print(f"Expected critical load factor: {expected_critical_load_factor}")
# Note: The plot_member_forces and plot_deformed_shape methods are not tested here
# as they involve matplotlib visualization. These would typically be tested
# separately or with mocking of matplotlib functions.
def test_frame_solver_local_geometric_stiffness_matrix_3D(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    k_g = solver.local_geometric_stiffness_matrix_3D_beam(3, 0.01, 2e-4, 1000,1,1,1,1,1)
    assert k_g.shape == (12, 12)
    assert np.allclose(k_g, k_g.T)  # Check symmetry

def test_plot_deformed_shape():
    # Create a dummy FrameSolver instance with minimal required attributes
    L = 3.0
    E = 200e9
    I = 1e-6
    A = 1e-4

    nodes = [Node(0, 0, 0, 0), Node(1, 0, 0, L)]
    elements = [Element(0, nodes[0], nodes[1], E, 0.3, A, I, I, 2*I, 2*I, np.array([0, 1, 0]))]
    loads = [Load(nodes[1], fy=-1000)]
    bcs = [BoundaryCondition(nodes[0], ux=True, uy=True, uz=True, rx=True, ry=True, rz=True)]
    
    solver = FrameSolver(nodes, elements, loads, bcs)
    U, R, critical_load_factor, buckling_mode = solver.solve()
    # Call the plot_deformed_shape method
    fig, ax = solver.plot_deformed_shape(U, buckling_mode, show=False)
    assert fig is not None, "Figure was not created"
    assert isinstance(fig, plt.Figure), "Returned object is not a matplotlib Figure"
    assert ax is not None, "Axes was not created"
    assert isinstance(ax, Axes3D), "Returned object is not a matplotlib Axes"


    print("Plot creation test passed successfully!")
if __name__ == "__main__":
    pytest.main()