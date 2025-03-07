import pytest
import numpy as np
from frame_solver import Node, Element, Load, BoundaryCondition, FrameSolver

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

def test_node():
    node = Node(0, 1, 2, 3)
    assert node.id == 0
    assert node.x == 1
    assert node.y == 2
    assert node.z == 3
    np.testing.assert_array_equal(node.coordinates, np.array([1, 2, 3]))

def test_element():
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

def test_load():
    node = Node(0, 0, 0, 0)
    load = Load(node, fx=1000, fy=2000, fz=3000, mx=4000, my=5000, mz=6000)
    assert load.node == node
    assert load.fx == 1000
    assert load.fy == 2000
    assert load.fz == 3000
    assert load.mx == 4000
    assert load.my == 5000
    assert load.mz == 6000

def test_boundary_condition():
    node = Node(0, 0, 0, 0)
    bc = BoundaryCondition(node, ux=True, uy=True, uz=False, rx=True, ry=False, rz=True)
    assert bc.node == node
    assert bc.ux == True
    assert bc.uy == True
    assert bc.uz == False
    assert bc.rx == True
    assert bc.ry == False
    assert bc.rz == True

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

def test_frame_solver_assemble_global_stiffness_matrix(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    K = solver._assemble_global_stiffness_matrix()
    assert K.shape == (len(nodes) * 6, len(nodes) * 6)

def test_frame_solver_compute_element_stiffness_matrix(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    k = solver._compute_element_stiffness_matrix(elements[0])
    assert k.shape == (12, 12)

def test_frame_solver_compute_transformation_matrix(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    T = solver._compute_transformation_matrix(elements[0])
    assert T.shape == (12, 12)

def test_frame_solver_get_element_dofs(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    dofs = solver._get_element_dofs(elements[0])
    assert len(dofs) == 12

def test_frame_solver_assemble_load_vector(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    F = solver._assemble_load_vector()
    assert F.shape == (len(nodes) * 6,)

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

def test_frame_solver_get_constrained_dofs(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    constrained_dofs = solver._get_constrained_dofs()
    assert len(constrained_dofs) > 0

def test_frame_solver_update_geometric_stiffness(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    
    # Store initial Fx2 values
    initial_Fx2 = [element.Fx2 for element in elements]
    
    # Create a more realistic displacement vector
    U = np.random.rand(len(nodes) * 6) * 0.01  # Small random displacements
    
    # Update geometric stiffness
    solver._update_geometric_stiffness(U)
    
    # Check if the method ran without errors
    assert True
    
    # Optionally, check if any Fx2 values changed
    final_Fx2 = [element.Fx2 for element in elements]
    assert any(initial != final for initial, final in zip(initial_Fx2, final_Fx2)), \
        "No changes in Fx2 values after updating geometric stiffness"

def test_frame_solver_rotation_matrix_3D(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    gamma = solver.rotation_matrix_3D(0, 0, 0, 1, 1, 1)
    assert gamma.shape == (3, 3)

def test_frame_solver_transformation_matrix_3D(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    gamma = np.eye(3)
    T = solver.transformation_matrix_3D(gamma)
    assert T.shape == (12, 12)

def test_frame_solver_local_elastic_stiffness_matrix_3D_beam(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    k_e = solver.local_elastic_stiffness_matrix_3D_beam(200e9, 0.3, 0.01, 3, 1e-4, 1e-4, 2e-4)
    assert k_e.shape == (12, 12)

def test_frame_solver_local_geometric_stiffness_matrix_3D_beam_without_interaction_terms(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    k_g = solver.local_geometric_stiffness_matrix_3D_beam_without_interaction_terms(3, 0.01, 2e-4, 1000)
    assert k_g.shape == (12, 12)

def test_frame_solver_local_geometric_stiffness_matrix_3D_beam(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    k_g = solver.local_geometric_stiffness_matrix_3D_beam(3, 0.01, 2e-4, 1000, 100, 200, 300, 400, 500)
    assert k_g.shape == (12, 12)

def test_frame_solver_assemble_global_geometric_stiffness_matrix(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    U = np.ones(len(nodes) * 6)
    K_geo = solver._assemble_global_geometric_stiffness_matrix(U)
    assert K_geo.shape == (len(nodes) * 6, len(nodes) * 6)

def test_frame_solver_get_free_dofs(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    free_dofs = solver._get_free_dofs()
    assert len(free_dofs) < len(nodes) * 6

def test_frame_solver_compute_member_forces(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    U = np.ones(len(nodes) * 6)
    forces = solver.compute_member_forces(elements[0], U)
    assert forces.shape == (12,)

def test_frame_solver_solve_critical_buckling_load(simple_frame):
    nodes, elements, loads, bcs = simple_frame
    solver = FrameSolver(nodes, elements, loads, bcs)
    U = np.ones(len(nodes) * 6)
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

if __name__ == "__main__":
    pytest.main()