# examples/frame_analysis_example.py
import numpy as np
from frame_solver import Node, Element, Load, BoundaryCondition, FrameSolver
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_vector(start, end, num_elements=7):
    """Create a linearly spaced vector between start and end points."""
    vector = np.linspace(start, end, num_elements)
    return vector.reshape(-1, 3)

def print_results(nodes, displacements, reactions):
    """Print displacements, rotations, and reactions for each node."""
    print("Displacements and Rotations:")
    for i, node in enumerate(nodes):
        start_index = i * 6
        print(f"Node {node.id} (x={node.x:.2f}, y={node.y:.2f}, z={node.z:.2f}):")
        disp_vector = displacements[start_index:start_index + 6]
        print(f"  [ux, uy, uz, rx, ry, rz] = {np.array2string(disp_vector, precision=6, separator=', ')}")

    print("\nReaction Forces and Moments:")
    for i, node in enumerate(nodes):
        start_index = i * 6
        reaction_vector = reactions[start_index:start_index + 6]
        if np.any(np.abs(reaction_vector) > 1e-6):
            print(f"Node {node.id} (x={node.x:.2f}, y={node.y:.2f}, z={node.z:.2f}):")
            print(f"  [Fx, Fy, Fz, Mx, My, Mz] = {np.array2string(reaction_vector, precision=6, separator=', ')}")
def Problem1_Sec():
    """Solve and analyze a 3D frame problem with 7 nodes."""
    print("Problem 1")
    start = [0, 0, 0]
    end = [18, 56, 44]
    result = create_vector(start, end)
    nodes = [Node(i, *coord) for i, coord in enumerate(result)]

    r, E, nu = 1, 10000, 0.3
    A = np.pi * r**2.0
    Iy = Iz = np.pi * (r**4.0) / 4.0
    I_rho = np.pi * r**4 / 2
    J = np.pi * r**4.0 / 2.0

    elements = [Element(i, nodes[i], nodes[i+1], E, nu, A, Iz, Iy, J, I_rho, None) for i in range(6)]
    loads = [Load(nodes[6], fx=0.05, fy=-0.1, fz=0.23, mx=0.1, my=-0.025, mz=-0.08)]
    bcs = [BoundaryCondition(nodes[0], True, True, True, True, True, True)]
    bcs.extend([BoundaryCondition(node, False, False, False, False, False, False) for node in nodes[1:]])

    solver = FrameSolver(nodes, elements, loads, bcs)
    displacements, reactions, critical_load_factor, buckling_mode = solver.solve()

    print_results(nodes, displacements, reactions)
    solver.plot_deformed_shape(displacements, buckling_mode, scale=10)
    print(f"\nElastic Critical Load Factor: {critical_load_factor}")
    print(f"\nBuckling Mode: {buckling_mode}")

def Problem2_Sec():
    """Solve and analyze a 3D frame problem with axial load."""
    print("Problem 2")
    start = np.array([0, 0, 0])
    end = np.array([18, 56, 44])
    result = create_vector(start, end)
    nodes = [Node(i, *coord) for i, coord in enumerate(result)]

    r, E, nu = 1, 10000, 0.3
    A = np.pi * r**2.0
    Iy = Iz = np.pi * (r**4.0) / 4.0
    I_rho = np.pi * r**4 / 2
    J = np.pi * r**4.0 / 2.0

    elements = [Element(i, nodes[i], nodes[i+1], E, nu, A, Iz, Iy, J, I_rho, None) for i in range(6)]

    P = 1
    L = np.linalg.norm(end-start)
    F = -1.0 * P * (end-start) / L
    loads = [Load(nodes[6], fx=F[0], fy=F[1], fz=F[2], mx=0, my=0, mz=0)]

    bcs = [BoundaryCondition(nodes[0], True, True, True, True, True, True)]
    bcs.extend([BoundaryCondition(node, False, False, False, False, False, False) for node in nodes[1:]])

    solver = FrameSolver(nodes, elements, loads, bcs)
    displacements, reactions, critical_load_factor, buckling_mode = solver.solve()

    solver.plot_deformed_shape(displacements, buckling_mode, scale=1, show_deformed=False)
    print(f"\nElastic Critical Load Factor: {critical_load_factor}")
    print(f"\nBuckling Mode: {buckling_mode}")

def Problem3_Sec():
    """Solve and analyze a 3D frame problem with a more complex structure."""
    print("Problem 3")
    L1, L2, L3, L4 = 15.0, 30.0, 14.0, 16.0
    nodes = [
        Node(0, 0, 0, 0), Node(1, L1, 0, 0), Node(2, L1, L2, 0), Node(3, 0, L2, 0),
        Node(4, 0, 0, L3), Node(5, L1, 0, L3), Node(6, L1, L2, L3), Node(7, 0, L2, L3),
        Node(8, 0, 0, L3+L4), Node(9, L1, 0, L3+L4), Node(10, L1, L2, L3+L4), Node(11, 0, L2, L3+L4)
    ]

    r_a, E_a, nu_a = 1.0, 10000.0, 0.3
    A_a = np.pi * r_a**2.0
    Iy_a = Iz_a = np.pi * (r_a**4.0) / 4.0
    I_rho_a = J_a = np.pi * r_a**4.0 / 2.0

    b, h = 0.5, 1.0
    E_b, nu_b = 50000.0, 0.3
    A_b = b * h
    Iy_b = h * b**3.0 / 12.0
    Iz_b = b * h**3.0 / 12.0
    I_rho_b = b * h / 12.0 * (b**2.0 + h**2.0)
    J_b = 0.028610026041666667

    elements = [
        Element(i, nodes[i], nodes[i+4], E_a, nu_a, A_a, Iz_a, Iy_a, J_a, I_rho_a, None) for i in range(4)
    ] + [
        Element(i+4, nodes[i+4], nodes[i+8], E_a, nu_a, A_a, Iz_a, Iy_a, J_a, I_rho_a, None) for i in range(4)
    ] + [
        Element(i+8, nodes[i+4], nodes[(i+1)%4+4], E_b, nu_b, A_b, Iz_b, Iy_b, J_b, I_rho_b, np.array([0,0,1])) for i in range(4)
    ] + [
        Element(i+12, nodes[i+8], nodes[(i+1)%4+8], E_b, nu_b, A_b, Iz_b, Iy_b, J_b, I_rho_b, np.array([0,0,1])) for i in range(4)
    ]

    loads = [Load(nodes[i], fx=0, fy=0, fz=-1, mx=0, my=0, mz=0) for i in range(8, 12)]

    bcs = [BoundaryCondition(nodes[i], True, True, True, True, True, True) for i in range(4)] + \
          [BoundaryCondition(nodes[i], False, False, False, False, False, False) for i in range(4, 12)]

    solver = FrameSolver(nodes, elements, loads, bcs)
    displacements, reactions, critical_load_factor, buckling_mode = solver.solve()

    print(f"\nElastic Critical Load Factor: {critical_load_factor}")
    print(f"\nBuckling Mode: {buckling_mode}")
    solver.plot_deformed_shape(displacements, buckling_mode, scale=0.1)
def Problem1():
    """Solve and analyze a 3D frame problem with 7 nodes."""
    print("Problem 1")
    start = [0, 0, 0]
    end = [25, 50, 37]
    result = create_vector(start, end)
    nodes = [Node(i, *coord) for i, coord in enumerate(result)]

    r, E, nu = 1, 10000, 0.3
    A = np.pi * r**2.0
    Iy = Iz = np.pi * (r**4.0) / 4.0
    I_rho = np.pi * r**4 / 2
    J = np.pi * r**4.0 / 2.0

    elements = [Element(i, nodes[i], nodes[i+1], E, nu, A, Iz, Iy, J, I_rho, None) for i in range(6)]
    loads = [Load(nodes[6], fx=0.05, fy=-0.1, fz=0.23, mx=0.1, my=-0.025, mz=-0.08)]
    bcs = [BoundaryCondition(nodes[0], True, True, True, True, True, True)]
    bcs.extend([BoundaryCondition(node, False, False, False, False, False, False) for node in nodes[1:]])

    solver = FrameSolver(nodes, elements, loads, bcs)
    displacements, reactions, critical_load_factor, buckling_mode = solver.solve()

    print_results(nodes, displacements, reactions)
    solver.plot_deformed_shape(displacements, buckling_mode, scale=10)
    print(f"\nElastic Critical Load Factor: {critical_load_factor}")
    print(f"\nBuckling Mode: {buckling_mode}")

def Problem2():
    """Solve and analyze a 3D frame problem with axial load."""
    print("Problem 2")
    start = np.array([0, 0, 0])
    end = np.array([25, 50, 37])
    result = create_vector(start, end)
    nodes = [Node(i, *coord) for i, coord in enumerate(result)]

    r, E, nu = 1, 10000, 0.3
    A = np.pi * r**2.0
    Iy = Iz = np.pi * (r**4.0) / 4.0
    I_rho = np.pi * r**4 / 2
    J = np.pi * r**4.0 / 2.0

    elements = [Element(i, nodes[i], nodes[i+1], E, nu, A, Iz, Iy, J, I_rho, None) for i in range(6)]

    P = 1
    L = np.linalg.norm(end-start)
    F = -1.0 * P * (end-start) / L
    loads = [Load(nodes[6], fx=F[0], fy=F[1], fz=F[2], mx=0, my=0, mz=0)]

    bcs = [BoundaryCondition(nodes[0], True, True, True, True, True, True)]
    bcs.extend([BoundaryCondition(node, False, False, False, False, False, False) for node in nodes[1:]])

    solver = FrameSolver(nodes, elements, loads, bcs)
    displacements, reactions, critical_load_factor, buckling_mode = solver.solve()

    solver.plot_deformed_shape(displacements, buckling_mode, scale=1, show_deformed=False)
    print(f"\nElastic Critical Load Factor: {critical_load_factor}")
    print(f"\nBuckling Mode: {buckling_mode}")

def Problem3():
    """Solve and analyze a 3D frame problem with a more complex structure."""
    print("Problem 3")
    L1, L2, L3, L4 = 11.0, 23.0, 15.0, 13.0
    nodes = [
        Node(0, 0, 0, 0), Node(1, L1, 0, 0), Node(2, L1, L2, 0), Node(3, 0, L2, 0),
        Node(4, 0, 0, L3), Node(5, L1, 0, L3), Node(6, L1, L2, L3), Node(7, 0, L2, L3),
        Node(8, 0, 0, L3+L4), Node(9, L1, 0, L3+L4), Node(10, L1, L2, L3+L4), Node(11, 0, L2, L3+L4)
    ]

    r_a, E_a, nu_a = 1.0, 10000.0, 0.3
    A_a = np.pi * r_a**2.0
    Iy_a = Iz_a = np.pi * (r_a**4.0) / 4.0
    I_rho_a = J_a = np.pi * r_a**4.0 / 2.0

    b, h = 0.5, 1.0
    E_b, nu_b = 50000.0, 0.3
    A_b = b * h
    Iy_b = h * b**3.0 / 12.0
    Iz_b = b * h**3.0 / 12.0
    I_rho_b = b * h / 12.0 * (b**2.0 + h**2.0)
    J_b = 0.028610026041666667

    elements = [
        Element(i, nodes[i], nodes[i+4], E_a, nu_a, A_a, Iz_a, Iy_a, J_a, I_rho_a, None) for i in range(4)
    ] + [
        Element(i+4, nodes[i+4], nodes[i+8], E_a, nu_a, A_a, Iz_a, Iy_a, J_a, I_rho_a, None) for i in range(4)
    ] + [
        Element(i+8, nodes[i+4], nodes[(i+1)%4+4], E_b, nu_b, A_b, Iz_b, Iy_b, J_b, I_rho_b, np.array([0,0,1])) for i in range(4)
    ] + [
        Element(i+12, nodes[i+8], nodes[(i+1)%4+8], E_b, nu_b, A_b, Iz_b, Iy_b, J_b, I_rho_b, np.array([0,0,1])) for i in range(4)
    ]

    loads = [Load(nodes[i], fx=0, fy=0, fz=-1, mx=0, my=0, mz=0) for i in range(8, 12)]

    bcs = [BoundaryCondition(nodes[i], True, True, True, True, True, True) for i in range(4)] + \
          [BoundaryCondition(nodes[i], False, False, False, False, False, False) for i in range(4, 12)]

    solver = FrameSolver(nodes, elements, loads, bcs)
    displacements, reactions, critical_load_factor, buckling_mode = solver.solve()

    print(f"\nElastic Critical Load Factor: {critical_load_factor}")
    print(f"\nBuckling Mode: {buckling_mode}")
    solver.plot_deformed_shape(displacements, buckling_mode, scale=0.1)

def Problem4():
    """Solve and analyze a 3D L-shaped frame."""
    nodes = [
        Node(0, 0, 0, 10),
        Node(1, 15, 0, 10),
        Node(2, 15, 0, 0)
    ]

    b, h = 0.5, 1.0
    E, nu = 1000, 0.3
    A = b * h
    Iy = h * (b**3) / 12
    Iz = b * (h**3) / 12
    I_rho = b * h * ((b**2) + (h**2)) / 12
    J = 0.02861

    elements = [
        Element(0, nodes[0], nodes[1], E, nu, A, Iz, Iy, J, I_rho, np.array([0, 0, 1])),
        Element(1, nodes[1], nodes[2], E, nu, A, Iz, Iy, J, I_rho, np.array([1, 0, 0]))
    ]

    loads = [Load(nodes[1], fx=-0.05, fy=0.075, fz=0.1, mx=-0.05, my=0.1, mz=-0.25)]

    bcs = [
        BoundaryCondition(nodes[0], True, True, True, True, True, True),
        BoundaryCondition(nodes[1], False, False, False, False, False, False),
        BoundaryCondition(nodes[2], True, True, True, False, False, False)
    ]

    solver = FrameSolver(nodes, elements, loads, bcs)
    displacements, reactions, critical_load_factor, buckling_mode = solver.solve()

    print_results(nodes, displacements, reactions)
    forces1 = solver.compute_member_forces(elements[0], displacements)
    solver.plot_deformed_shape(displacements, buckling_mode, scale=10)

    print(f"\nElastic Critical Load Factor: {critical_load_factor}")
    print(f"\nBuckling Mode: {buckling_mode}")

def Problem5():
    """Solve and analyze a 3D frame with multiple elements and mixed boundary conditions."""
    nodes = [
        Node(0, 0, 0, 0), Node(1, -5, 1, 10), Node(2, -1, 5, 13),
        Node(3, -3, 7, 11), Node(4, 6, 9, 5)
    ]

    r, E, nu = 1, 500, 0.3
    A = np.pi * (r**2)
    Iy = Iz = np.pi * (r**4) / 4
    I_rho = J = np.pi * (r**4) / 2

    elements = [
        Element(i, nodes[i], nodes[i+1], E, nu, A, Iz, Iy, J, I_rho, None) for i in range(3)
    ] + [Element(3, nodes[4], nodes[2], E, nu, A, Iz, Iy, J, I_rho, None)]

    loads = [
        Load(nodes[1], fx=0.05, fy=0.05, fz=-0.1, mx=0, my=0, mz=0),
        Load(nodes[2], fx=0, fy=0, fz=0, mx=-0.1, my=-0.1, mz=0.3)
    ]

    bcs = [
        BoundaryCondition(nodes[0], False, False, True, False, False, False),
        BoundaryCondition(nodes[1], False, False, False, False, False, False),
        BoundaryCondition(nodes[2], False, False, False, False, False, False),
        BoundaryCondition(nodes[3], True, True, True, True, True, True),
        BoundaryCondition(nodes[4], True, True, True, False, False, False),
    ]

    solver = FrameSolver(nodes, elements, loads, bcs)
    displacements, reactions, critical_load_factor, buckling_mode = solver.solve()

    print_results(nodes, displacements, reactions)
    solver.plot_deformed_shape(displacements, buckling_mode, scale=10)

    print(f"\nElastic Critical Load Factor: {critical_load_factor}")
    print(f"\nBuckling Mode: {buckling_mode}")

def Problem6():
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

    print_results(nodes, displacements, reactions)
    forces1 = solver.compute_member_forces(elements[0], displacements)
    solver.plot_deformed_shape(displacements, buckling_mode, scale=1)

    print(f"\nElastic Critical Load Factor: {critical_load_factor}")
    print(f"\nBuckling Mode: {buckling_mode}")

if __name__ == "__main__":
    Problem1_Sec()
    Problem2_Sec()
    Problem3_Sec()
    Problem1()
    Problem2()
    Problem3()
    Problem4()
    Problem5()
    Problem6()