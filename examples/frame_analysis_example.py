# examples/frame_analysis_example.py
import numpy as np
from  frame_solver import Node, Element, Load, BoundaryCondition, FrameSolver
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def Problem1():
    # Define nodes
    # Format: Node(id, x, y, z)
    nodes = [
        Node(0, 0, 0, 10),    # Base of the structure
        Node(1, 15, 0, 10),    # Top of the vertical member
        Node(2, 15, 0, 0)     # End of the horizontal member
    ]

    # Define elements
    # Format: Element(id, node1, node2, E, nu, A, Iz, Iy, J, I_rho, local_z)
    b=0.5
    h=1.0
    E = 1000
    nu = 0.3
    A = b*h
    Iy = h*(b**3)/12
    Iz =b*(h**3)/12
    I_rho = b*h*((b**2)+(h**2))/12
    J = 0.02861
                    
    elements = [
        Element(0, nodes[0], nodes[1], E, nu, A, Iz, Iy, J, I_rho, np.array([0, 0, 1])),  # Vertical member
        Element(1, nodes[1], nodes[2], E, nu, A, Iz, Iy, J, I_rho, np.array([1, 0, 0]))   # Horizontal member
    ]

    # Define loads
    # Format: Load(node, fx, fy, fz, mx, my, mz)
    loads = [
        Load(nodes[1], fx = -0.05,fy=0.075,fz = 0.1,mx=-0.05,my=0.1,mz=-0.25 )  # 10 kN downward force at the end of the horizontal member
    ]

    # Define boundary conditions
    # Format: BoundaryCondition(node, ux, uy, uz, rx, ry, rz)
    # True means the DOF is constrained (fixed), False means it's free
    bcs = [
        BoundaryCondition(nodes[0], True, True, True, True, True, True), # Fixed node
        BoundaryCondition(nodes[1], False, False, False, False, False, False), # free node
        BoundaryCondition(nodes[2], True, True, True, False, False, False)  # pin node
    ]

    # Create the FrameSolver instance
    solver = FrameSolver(nodes, elements, loads, bcs)

    # Solve the frame
    displacements, reactions, critical_load_factor, buckling_mode = solver.solve()
    # Print results
    print("Displacements and Rotations:")
    for i, node in enumerate(nodes):
        start_index = i * 6
        print(f"Node {node.id} (x={node.x}, y={node.y}, z={node.z}):")
        print(f"  ux: {displacements[start_index]:.6f}")
        print(f"  uy: {displacements[start_index + 1]:.6f}")
        print(f"  uz: {displacements[start_index + 2]:.6f}")
        print(f"  rx: {displacements[start_index + 3]:.6f}")
        print(f"  ry: {displacements[start_index + 4]:.6f}")
        print(f"  rz: {displacements[start_index + 5]:.6f}")

    print("\nReaction Forces and Moments:")
    for i, node in enumerate(nodes):
        start_index = i * 6
        if any(abs(reactions[start_index:start_index + 6]) > 1e-6):
            print(f"Node {node.id} (x={node.x}, y={node.y}, z={node.z}):")
            print(f"  Fx: {reactions[start_index]:.6f}")
            print(f"  Fy: {reactions[start_index + 1]:.6f}")
            print(f"  Fz: {reactions[start_index + 2]:.6f}")
            print(f"  Mx: {reactions[start_index + 3]:.6f}")
            print(f"  My: {reactions[start_index + 4]:.6f}")
            print(f"  Mz: {reactions[start_index + 5]:.6f}")
    # Compute and plot member forces for element0
    forces1 = solver.compute_member_forces(elements[0], displacements)
    solver.plot_member_forces(elements[0], forces1)

    # Plot deformed shape
    solver.plot_deformed_shape(reactions, scale=100)

    print(f"\nElastic Critical Load Factor: {critical_load_factor}")
    print(f"\nBuckling Mode: {buckling_mode}")
def Problem2():
    # Define nodes
    # Format: Node(id, x, y, z)

    nodes = [
        Node(0, 0, 0, 0),    # Base of the structure
        Node(1, -5, 1, 10),    # Top of the vertical member
        Node(2, -1, 5, 13),     # End of the horizontal member
        Node(3,-3,7,11),
        Node(4,6,9,5)
    ]

    # Define elements
    # Format: Element(id, node1, node2, E, nu, A, Iz, Iy, J, I_rho, local_z)
    r=1
    E=500
    nu =0.3
    A = np.pi*(r**2)
    Iy = Iz = np.pi*(r**4)/4
    I_rho = J = np.pi*(r**4)/2
                    
    elements = [
        Element(0, nodes[0], nodes[1], E, nu, A, Iz, Iy, J, I_rho, None),  
        Element(1, nodes[1], nodes[2], E, nu, A, Iz, Iy, J, I_rho, None),
        Element(2, nodes[3], nodes[2], E, nu, A, Iz, Iy, J, I_rho, None),   
        Element(3, nodes[4], nodes[2], E, nu, A, Iz, Iy, J, I_rho, None),      
    ]

    # Define loads
    # Format: Load(node, fx, fy, fz, mx, my, mz)
    loads = [
        Load(nodes[1], fx = 0.05,fy=0.05,fz = -0.1,mx=0,my=0,mz=0 ),
        Load(nodes[2], fx = 0,fy=0,fz = 0,mx=-0.1,my=-0.1,mz=0.3 )   
    ]

    # Define boundary conditions
    # Format: BoundaryCondition(node, ux, uy, uz, rx, ry, rz)
    # True means the DOF is constrained (fixed), False means it's free
    bcs = [
        BoundaryCondition(nodes[0], False, False, True, False, False, False),
        BoundaryCondition(nodes[1], False, False, False, False, False, False), 
        BoundaryCondition(nodes[2], False, False, False, False, False, False),
        BoundaryCondition(nodes[3], True, True, True, True, True, True), 
        BoundaryCondition(nodes[4], True, True, True, False, False, False),
    ]

    # Create the FrameSolver instance
    solver = FrameSolver(nodes, elements, loads, bcs)

    # Solve the frame
    displacements, reactions, critical_load_factor, buckling_mode = solver.solve()
    # Print results
    print("Displacements and Rotations:")
    for i, node in enumerate(nodes):
        start_index = i * 6
        print(f"Node {node.id} (x={node.x}, y={node.y}, z={node.z}):")
        print(f"  ux: {displacements[start_index]:.6f}")
        print(f"  uy: {displacements[start_index + 1]:.6f}")
        print(f"  uz: {displacements[start_index + 2]:.6f}")
        print(f"  rx: {displacements[start_index + 3]:.6f}")
        print(f"  ry: {displacements[start_index + 4]:.6f}")
        print(f"  rz: {displacements[start_index + 5]:.6f}")

    print("\nReaction Forces and Moments:")
    for i, node in enumerate(nodes):
        start_index = i * 6
        if any(abs(reactions[start_index:start_index + 6]) > 1e-6):
            print(f"Node {node.id} (x={node.x}, y={node.y}, z={node.z}):")
            print(f"  Fx: {reactions[start_index]:.6f}")
            print(f"  Fy: {reactions[start_index + 1]:.6f}")
            print(f"  Fz: {reactions[start_index + 2]:.6f}")
            print(f"  Mx: {reactions[start_index + 3]:.6f}")
            print(f"  My: {reactions[start_index + 4]:.6f}")
            print(f"  Mz: {reactions[start_index + 5]:.6f}")
    # Compute and plot member forces for element0
    forces1 = solver.compute_member_forces(elements[0], displacements)
    solver.plot_member_forces(elements[0], forces1)

    # Plot deformed shape
    solver.plot_deformed_shape(reactions, scale=100)

    print(f"\nElastic Critical Load Factor: {critical_load_factor}")
    print(f"\nBuckling Mode: {buckling_mode}")

if __name__ == "__main__":
    Problem2()