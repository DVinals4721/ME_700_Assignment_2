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
    solver.plot_deformed_shape(displacements, buckling_mode, scale=10)  # Adjust scale as needed

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
    solver.plot_deformed_shape(displacements, buckling_mode, scale=10)  # Adjust scale as needed

    print(f"\nElastic Critical Load Factor: {critical_load_factor}")
    print(f"\nBuckling Mode: {buckling_mode}")

def Problem3():
    # Define nodes
    # Format: Node(id, x, y, z)

    nodes = [
        Node(0, 0, 0, 0),    # Base of the structure
        Node(1, 30, 40, 0),    # Top of the vertical member

    ]

    # Define elements
    # Format: Element(id, node1, node2, E, nu, A, Iz, Iy, J, I_rho, local_z)
    r=1
    E=1000
    nu =0.3
    A = np.pi*r**2.0
    Iy = Iz = np.pi*(r**4.0)/4.0
    I_rho = np.pi*(r**4)/2
    J = np.pi*r**4.0/2.0
                    
    elements = [
        Element(0, nodes[0], nodes[1], E, nu, A, Iz, Iy, J, I_rho, None),       
    ]

    # Define loads
    # Format: Load(node, fx, fy, fz, mx, my, mz)
    loads = [
        Load(nodes[1], fx = -3/5,fy=-4/5,fz = 0,mx=0,my=0,mz=0 ),  
    ]

    # Define boundary conditions
    # Format: BoundaryCondition(node, ux, uy, uz, rx, ry, rz)
    # True means the DOF is constrained (fixed), False means it's free
    bcs = [
        BoundaryCondition(nodes[0], True, True, True, True, True, True),
        BoundaryCondition(nodes[1], False, False, False, False, False, False), 
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
    solver.plot_deformed_shape(displacements, buckling_mode, scale=1)  # Adjust scale as needed

    print(f"\nElastic Critical Load Factor: {critical_load_factor}")
    print(f"\nBuckling Mode: {buckling_mode}")
def Problem4():
    # Define nodes
    # Format: Node(id, x, y, z)
    def create_vector(start, end, num_elements=7):
        # Create a linearly spaced array
        vector = np.linspace(start, end, num_elements)
        
        # Reshape the array to have 3 coordinates (x, y, z)
        vector = vector.reshape(-1, 3)
        
        return vector
    # Example usage
    start = [0, 0, 0]  # Starting point
    end = [25, 50, 37]  # Ending point

    result = create_vector(start, end)
    nodes = [Node(i, *coord) for i, coord in enumerate(result)]

    # Define elements
    # Format: Element(id, node1, node2, E, nu, A, Iz, Iy, J, I_rho, local_z)
    r=1
    E=10000
    nu =0.3
    A = np.pi*r**2.0
    Iy = Iz = np.pi*(r**4.0)/4.0
    I_rho = np.pi*(r**4)/2
    J = np.pi*r**4.0/2.0
                    
    elements = [
        Element(0, nodes[0], nodes[1], E, nu, A, Iz, Iy, J, I_rho, None),
        Element(1, nodes[1], nodes[2], E, nu, A, Iz, Iy, J, I_rho, None),   
        Element(2, nodes[2], nodes[3], E, nu, A, Iz, Iy, J, I_rho, None),   
        Element(3, nodes[3], nodes[4], E, nu, A, Iz, Iy, J, I_rho, None),   
        Element(4, nodes[4], nodes[5], E, nu, A, Iz, Iy, J, I_rho, None),
        Element(5, nodes[5], nodes[6], E, nu, A, Iz, Iy, J, I_rho, None),      
    ]

    # Define loads
    # Format: Load(node, fx, fy, fz, mx, my, mz)
    loads = [
        Load(nodes[6], fx = 0.05,fy=-0.01,fz = 0.23,mx=0.1,my=-0.025,mz=-0.08 ),  
    ]

    # Define boundary conditions
    # Format: BoundaryCondition(node, ux, uy, uz, rx, ry, rz)
    # True means the DOF is constrained (fixed), False means it's free
    bcs = [
        BoundaryCondition(nodes[0], True, True, True, True, True, True),
        BoundaryCondition(nodes[1], False, False, False, False, False, False),
        BoundaryCondition(nodes[2], False, False, False, False, False, False),
        BoundaryCondition(nodes[3], False, False, False, False, False, False),
        BoundaryCondition(nodes[4], False, False, False, False, False, False),
        BoundaryCondition(nodes[5], False, False, False, False, False, False),
        BoundaryCondition(nodes[6], False, False, False, False, False, False),  
    ]

    # Create the FrameSolver instance
    solver = FrameSolver(nodes, elements, loads, bcs)

    # Solve the frame
    displacements, reactions, critical_load_factor, buckling_mode = solver.solve()
    # Print results
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
    # Compute and plot member forces for element0
    forces1 = solver.compute_member_forces(elements[0], displacements)
    solver.plot_member_forces(elements[0], forces1)

    # Plot deformed shape
    solver.plot_deformed_shape(displacements, buckling_mode, scale=10)  # Adjust scale as needed

    print(f"\nElastic Critical Load Factor: {critical_load_factor}")
    print(f"\nBuckling Mode: {buckling_mode}")
def Problem5():
    # Define nodes
    # Format: Node(id, x, y, z)
    L1=11.0
    L2=23.0
    L3=15.0
    L4 =13.0

    nodes = [
        Node(0, 0, 0, 0),    
        Node(1, L1, 0, 0),
        Node(2,L1,L2,0),
        Node(3,0,L2,0),
        Node(4,0,0,L3),
        Node(5,L1,0,L3),
        Node(6,L1,L2,L3),
        Node(7,0,L2,L3),
        Node(8,0,0,L3+L4),
        Node(9,L1,0,L3+L4),
        Node(10,L1,L2,L3+L4),
        Node(11,0,L2,L3+L4)
    ]

    # Define elements
    # Format: Element(id, node1, node2, E, nu, A, Iz, Iy, J, I_rho, local_z)
    r_a=1.0
    E_a=10000.0
    nu_a =0.3
    A_a = np.pi*r_a**2.0
    Iy_a = Iz_a = np.pi*(r_a**4.0)/4.0
    I_rho_a = np.pi*(r_a**4)/2
    J_a = np.pi*r_a**4.0/2.0
    local_z_a = None

    b=0.5
    h=1.0
    E_b=50000.0
    nu_b =0.3
    A_b = b*h
    Iy_b = h*b**3.0 / 12.0
    Iz_b = b*h**3.0/12.0
    I_rho_b = b*h/12.0*(b**2.0+h**2.0)
    J_b = 0.028610026041666667
    local_z_b = np.asarray([0,0,1])
                    
    elements = [
        Element(0, nodes[0], nodes[4], E_a, nu_a, A_a, Iz_a, Iy_a, J_a, I_rho_a, local_z_a),    
        Element(1, nodes[1], nodes[5], E_a, nu_a, A_a, Iz_a, Iy_a, J_a, I_rho_a, local_z_a),
        Element(2, nodes[2], nodes[6], E_a, nu_a, A_a, Iz_a, Iy_a, J_a, I_rho_a, local_z_a),
        Element(3, nodes[3], nodes[7], E_a, nu_a, A_a, Iz_a, Iy_a, J_a, I_rho_a, local_z_a),
        Element(4, nodes[4], nodes[8], E_a, nu_a, A_a, Iz_a, Iy_a, J_a, I_rho_a, local_z_a),
        Element(5, nodes[5], nodes[9], E_a, nu_a, A_a, Iz_a, Iy_a, J_a, I_rho_a, local_z_a),
        Element(6, nodes[6], nodes[10], E_a, nu_a, A_a, Iz_a, Iy_a, J_a, I_rho_a, local_z_a),
        Element(7, nodes[7], nodes[11], E_a, nu_a, A_a, Iz_a, Iy_a, J_a, I_rho_a, local_z_a),
        Element(8, nodes[4], nodes[5], E_b, nu_b, A_b, Iz_b, Iy_b, J_b, I_rho_b, local_z_b),
        Element(9, nodes[5], nodes[6], E_b, nu_b, A_b, Iz_b, Iy_b, J_b, I_rho_b, local_z_b),
        Element(10, nodes[6], nodes[7], E_b, nu_b, A_b, Iz_b, Iy_b, J_b, I_rho_b, local_z_b),
        Element(11, nodes[7], nodes[4], E_b, nu_b, A_b, Iz_b, Iy_b, J_b, I_rho_b, local_z_b),
        Element(12, nodes[8], nodes[9], E_b, nu_b, A_b, Iz_b, Iy_b, J_b, I_rho_b, local_z_b),
        Element(13, nodes[9], nodes[10], E_b, nu_b, A_b, Iz_b, Iy_b, J_b, I_rho_b, local_z_b),
        Element(14, nodes[10], nodes[11], E_b, nu_b, A_b, Iz_b, Iy_b, J_b, I_rho_b, local_z_b),
        Element(15, nodes[11], nodes[8], E_b, nu_b, A_b, Iz_b, Iy_b, J_b, I_rho_b, local_z_b),
    ]
    #print(elements)
    # Define loads
    # Format: Load(node, fx, fy, fz, mx, my, mz)
    loads = [
        Load(nodes[8], fx = 0.0,fy=0.0,fz = -1.0,mx=0.0,my=0.0,mz=0.0 ),  
        Load(nodes[9], fx = 0.0,fy=0.0,fz = -1.0,mx=0.0,my=0.0,mz=0.0 ),
        Load(nodes[10], fx = 0.0,fy=0.0,fz = -1.0,mx=0.0,my=0.0,mz=0.0 ),
        Load(nodes[11], fx = 0.0,fy=0.0,fz = -1.0,mx=0.0,my=0.0,mz=0.0 ),
    ]

    # Define boundary conditions
    # Format: BoundaryCondition(node, ux, uy, uz, rx, ry, rz)
    # True means the DOF is constrained (fixed), False means it's free
    bcs = [
        BoundaryCondition(nodes[0], True, True, True, True, True, True),
        BoundaryCondition(nodes[1], True, True, True, True, True, True),
        BoundaryCondition(nodes[2], True, True, True, True, True, True),
        BoundaryCondition(nodes[3], True, True, True, True, True, True),
        BoundaryCondition(nodes[4], False, False, False, False, False, False),
        BoundaryCondition(nodes[5], False, False, False, False, False, False),
        BoundaryCondition(nodes[6], False, False, False, False, False, False),
        BoundaryCondition(nodes[7], False, False, False, False, False, False),
        BoundaryCondition(nodes[8], False, False, False, False, False, False),
        BoundaryCondition(nodes[9], False, False, False, False, False, False),  
        BoundaryCondition(nodes[10], False, False, False, False, False, False),
        BoundaryCondition(nodes[11], False, False, False, False, False, False),  
    ]

    # Create the FrameSolver instance
    solver = FrameSolver(nodes, elements, loads, bcs)

    # Solve the frame
    displacements, reactions, critical_load_factor, buckling_mode = solver.solve()
    #solver.plot_original_frame()
    # Plot deformed shape
    solver.plot_deformed_shape(displacements, buckling_mode, scale=0.1)  # Adjust scale as needed

    print(f"\nElastic Critical Load Factor: {critical_load_factor}")
    print(f"\nBuckling Mode: {buckling_mode}")
if __name__ == "__main__":
    Problem5()