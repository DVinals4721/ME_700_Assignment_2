# examples/frame_analysis_example.py
import numpy as np
from  frame_solver import Node, Element, Load, BoundaryCondition, FrameSolver
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def main():
    # Define nodes
    # Format: Node(id, x, y, z)
    try:
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
            Load(nodes[1], fx = 0.1,fy=0.05,fz = -0.07,mx=0.05,my=-0.1,mz=0.25 )  # 10 kN downward force at the end of the horizontal member
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
        displacements, reactions = solver.solve()

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

        # Visualize the frame
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot original frame
        for element in elements:
            x = [element.node1.x, element.node2.x]
            y = [element.node1.y, element.node2.y]
            z = [element.node1.z, element.node2.z]
            ax.plot(x, y, z, 'b-', linewidth=2)

        # Plot deformed frame (scaled for visibility)
        scale = 100  # Adjust this value to make deformations more visible
        for element in elements:
            x = [element.node1.x + displacements[element.node1.id * 6] * scale,
                element.node2.x + displacements[element.node2.id * 6] * scale]
            y = [element.node1.y + displacements[element.node1.id * 6 + 1] * scale,
                element.node2.y + displacements[element.node2.id * 6 + 1] * scale]
            z = [element.node1.z + displacements[element.node1.id * 6 + 2] * scale,
                element.node2.z + displacements[element.node2.id * 6 + 2] * scale]
            ax.plot(x, y, z, 'r--', linewidth=1)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Frame Deformation (scaled)')
        plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Debug information:")
        print(f"Number of nodes: {len(nodes)}")
        print(f"Number of elements: {len(elements)}")
        print(f"Number of loads: {len(loads)}")
        print(f"Number of boundary conditions: {len(bcs)}")
        for i, element in enumerate(elements):
            print(f"Element {i}:")
            print(f"  Node 1: {element.node1.id} ({element.node1.x}, {element.node1.y}, {element.node1.z})")
            print(f"  Node 2: {element.node2.id} ({element.node2.x}, {element.node2.y}, {element.node2.z})")
            print(f"  Properties: E={element.E}, nu={element.nu}, A={element.A}, Iz={element.Iz}, Iy={element.Iy}, J={element.J}, I_rho={element.I_rho}")
            print(f"  Local z: {element.local_z}")

if __name__ == "__main__":
    main()