import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy

@dataclass
class Node:
    id: int
    x: float
    y: float
    z: float

    @property
    def coordinates(self):
        return np.array([self.x, self.y, self.z])

@dataclass
class Element:
    id: int
    node1: Node
    node2: Node
    E: float  # Young's modulus
    nu: float  # Poisson's ratio
    A: float  # Cross-sectional area
    Iz: float  # Second moment of area about local z-axis
    Iy: float  # Second moment of area about local y-axis
    J: float  # Torsional constant
    I_rho: float  # Polar moment of inertia
    local_z: np.ndarray  # Local z-axis direction
    Fx2: float = 0  # Axial force (to be updated during analysis)

@dataclass
class Load:
    node: Node
    fx: float = 0
    fy: float = 0
    fz: float = 0
    mx: float = 0
    my: float = 0
    mz: float = 0

@dataclass
class BoundaryCondition:
    node: Node
    ux: bool = False
    uy: bool = False
    uz: bool = False
    rx: bool = False
    ry: bool = False
    rz: bool = False

class FrameSolver:
    def __init__(self, nodes: List[Node], elements: List[Element], loads: List[Load], bcs: List[BoundaryCondition]):
        self.nodes = nodes
        self.elements = elements
        self.loads = loads
        self.bcs = bcs
        self.ndof = len(nodes) * 6
        self.kg_included = False

    def solve(self):
        max_iterations = 1000
        tolerance = 1e-8
        
        for iteration in range(max_iterations):
            K = self._assemble_global_stiffness_matrix()
            F = self._assemble_load_vector()
    
            
            K_mod, F_mod = self._apply_boundary_conditions(K, F)
            
            if np.any(np.isnan(K_mod)) or np.any(np.isinf(K_mod)):
                raise ValueError("K_mod contains NaN or Inf values")
            if np.any(np.isnan(F_mod)) or np.any(np.isinf(F_mod)):
                raise ValueError("F_mod contains NaN or Inf values")
            
            try:
                U_mod = np.linalg.solve(K_mod, F_mod)
            except np.linalg.LinAlgError as e:
                raise
            
            U = self._recover_full_displacement_vector(U_mod)
            
            if iteration > 0 and np.linalg.norm(U - U_prev) < tolerance:
                break
            
            U_prev = U.copy()
            self._update_geometric_stiffness(U)
        
        R = K @ U - F
        critical_load_factor, buckling_mode = self.solve_critical_buckling_load(U)
        
        return U, R, critical_load_factor, buckling_mode
  
    
    def _assemble_global_stiffness_matrix(self) -> np.ndarray:
        K = np.zeros((self.ndof, self.ndof))
        for element in self.elements:
            k_local = self._compute_element_stiffness_matrix(element)
            T = self._compute_transformation_matrix(element)
            k_global = T.T @ k_local @ T
            dofs = self._get_element_dofs(element)
            K[np.ix_(dofs, dofs)] += k_global
        return K

    def _compute_element_stiffness_matrix(self, element: Element) -> np.ndarray:
        L = np.linalg.norm(element.node2.coordinates - element.node1.coordinates)
        k_e = self.local_elastic_stiffness_matrix_3D_beam(
            element.E, element.nu, element.A, L, element.Iy, element.Iz, element.J
        )
        return k_e
    def _compute_transformation_matrix(self, element: Element) -> np.ndarray:
        x1, y1, z1 = element.node1.coordinates
        x2, y2, z2 = element.node2.coordinates
        gamma = self.rotation_matrix_3D(x1, y1, z1, x2, y2, z2, element.local_z)
        return self.transformation_matrix_3D(gamma)
    def _get_element_dofs(self, element: Element) -> List[int]:
        dofs = []
        for node in [element.node1, element.node2]:
            start_dof = node.id * 6
            dofs.extend(range(start_dof, start_dof + 6))
        return dofs
    def _assemble_load_vector(self) -> np.ndarray:
        F = np.zeros(self.ndof)
        for load in self.loads:
            start_dof = load.node.id * 6
            F[start_dof:start_dof+6] += [load.fx, load.fy, load.fz, load.mx, load.my, load.mz]
        return F
    def _apply_boundary_conditions(self, K: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        free_dofs = list(range(self.ndof))
        for bc in self.bcs:
            start_dof = bc.node.id * 6
            if bc.ux: free_dofs.remove(start_dof)
            if bc.uy: free_dofs.remove(start_dof + 1)
            if bc.uz: free_dofs.remove(start_dof + 2)
            if bc.rx: free_dofs.remove(start_dof + 3)
            if bc.ry: free_dofs.remove(start_dof + 4)
            if bc.rz: free_dofs.remove(start_dof + 5)
        return K[np.ix_(free_dofs, free_dofs)], F[free_dofs]
    def _recover_full_displacement_vector(self, U_mod: np.ndarray) -> np.ndarray:
        U = np.zeros(self.ndof)
        free_dofs = [i for i in range(self.ndof) if i not in self._get_constrained_dofs()]
        U[free_dofs] = U_mod
        return U
    def _get_constrained_dofs(self) -> List[int]:
        constrained_dofs = []
        for bc in self.bcs:
            start_dof = bc.node.id * 6
            if bc.ux: constrained_dofs.append(start_dof)
            if bc.uy: constrained_dofs.append(start_dof + 1)
            if bc.uz: constrained_dofs.append(start_dof + 2)
            if bc.rx: constrained_dofs.append(start_dof + 3)
            if bc.ry: constrained_dofs.append(start_dof + 4)
            if bc.rz: constrained_dofs.append(start_dof + 5)
        return constrained_dofs
    def _update_geometric_stiffness(self, U):
        for element in self.elements:
            u1 = U[element.node1.id * 6 : element.node1.id * 6 + 6]
            u2 = U[element.node2.id * 6 : element.node2.id * 6 + 6]
            k_local = self._compute_element_stiffness_matrix(element)
            T = self._compute_transformation_matrix(element)
            
            # Combine displacements of both nodes
            u_element = np.concatenate((u1, u2))
            
            # Transform global displacements to local coordinates
            u_local = T.T @ u_element
            
            # Compute local forces
            f_local = k_local @ u_local
            
            # Update the axial force (Fx2)
            element.Fx2 = f_local[6]  # The axial force at the second node
    def rotation_matrix_3D(self, x1, y1, z1, x2, y2, z2, v_temp=None):
        L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        if L < 1e-10:  # Check for very short elements
            raise ValueError(f"Element length is too small: {L}")
        
        local_x = np.array([(x2 - x1) / L, (y2 - y1) / L, (z2 - z1) / L])
        
        # If v_temp is not provided, choose a default
        if v_temp is None:
            v_temp = np.array([0, 1.0, 0.0])
        
        # Check if local_x is parallel to v_temp
        if np.abs(np.dot(local_x, v_temp)) > 0.999:
            # If they're parallel, choose a different v_temp
            v_temp = np.array([1.0, 0.0, 0.0])
            
            # Check again, just in case
            if np.abs(np.dot(local_x, v_temp)) > 0.999:
                v_temp = np.array([0.0, 0.0, 1.0])
        
        local_y = np.cross(v_temp, local_x)
        local_y_norm = np.linalg.norm(local_y)
        
        if local_y_norm < 1e-10:
            raise ValueError(f"Unable to compute local y-axis. local_x: {local_x}, v_temp: {v_temp}")
        
        local_y /= local_y_norm
        local_z = np.cross(local_x, local_y)
        
        return np.vstack((local_x, local_y, local_z))
    def transformation_matrix_3D(self,gamma):
        Gamma = np.zeros((12, 12))
        for i in range(4):
            Gamma[i*3:(i+1)*3, i*3:(i+1)*3] = gamma
        return Gamma
    def local_elastic_stiffness_matrix_3D_beam(self,E: float, nu: float, A: float, L: float, Iy: float, Iz: float, J: float) -> np.ndarray:
        k_e = np.zeros((12, 12))
        # Axial terms
        axial_stiffness = E * A / L
        k_e[0, 0] = k_e[6, 6] = axial_stiffness
        k_e[0, 6] = k_e[6, 0] = -axial_stiffness
        # Torsion terms
        torsional_stiffness = E * J / (2.0 * (1 + nu) * L)
        k_e[3, 3] = k_e[9, 9] = torsional_stiffness
        k_e[3, 9] = k_e[9, 3] = -torsional_stiffness
        # Bending terms - z axis
        k_e[1, 1] = k_e[7, 7] = 12.0 * E * Iz / L**3
        k_e[1, 7] = k_e[7, 1] = -12.0 * E * Iz / L**3
        k_e[1, 5] = k_e[5, 1] = k_e[1, 11] = k_e[11, 1] = 6.0 * E * Iz / L**2
        k_e[5, 7] = k_e[7, 5] = k_e[7, 11] = k_e[11, 7] = -6.0 * E * Iz / L**2
        k_e[5, 5] = k_e[11, 11] = 4.0 * E * Iz / L
        k_e[5, 11] = k_e[11, 5] = 2.0 * E * Iz / L
        # Bending terms - y axis
        k_e[2, 2] = k_e[8, 8] = 12.0 * E * Iy / L**3
        k_e[2, 8] = k_e[8, 2] = -12.0 * E * Iy / L**3
        k_e[2, 4] = k_e[4, 2] = k_e[2, 10] = k_e[10, 2] = -6.0 * E * Iy / L**2
        k_e[4, 8] = k_e[8, 4] = k_e[8, 10] = k_e[10, 8] = 6.0 * E * Iy / L**2
        k_e[4, 4] = k_e[10, 10] = 4.0 * E * Iy / L
        k_e[4, 10] = k_e[10, 4] = 2.0 * E * Iy / L
        return k_e

    def local_geometric_stiffness_matrix_3D_beam(self,L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2):
        """
        local element geometric stiffness matrix
        source: p. 258 of McGuire's Matrix Structural Analysis 2nd Edition
        Given:
            material and geometric parameters:
                L, A, I_rho (polar moment of inertia)
            element forces and moments:
                Fx2, Mx2, My1, Mz1, My2, Mz2
        Context:
            load vector:
                [Fx1, Fy1, Fz1, Mx1, My1, Mz1, Fx2, Fy2, Fz2, Mx2, My2, Mz2]
            DOF vector:
                [u1, v1, w1, th_x1, th_y1, th_z1, u2, v2, w2, th_x2, th_y2, th_z2]
            Equation:
                [load vector] = [stiffness matrix] @ [DOF vector]
        Returns:
            12 x 12 geometric stiffness matrix k_g
        """
        k_g = np.zeros((12, 12))
        # upper triangle off diagonal terms
        k_g[0, 6] = -Fx2 / L
        k_g[1, 3] = My1 / L
        k_g[1, 4] = Mx2 / L
        k_g[1, 5] = Fx2 / 10.0
        k_g[1, 7] = -6.0 * Fx2 / (5.0 * L)
        k_g[1, 9] = My2 / L
        k_g[1, 10] = -Mx2 / L
        k_g[1, 11] = Fx2 / 10.0
        k_g[2, 3] = Mz1 / L
        k_g[2, 4] = -Fx2 / 10.0
        k_g[2, 5] = Mx2 / L
        k_g[2, 8] = -6.0 * Fx2 / (5.0 * L)
        k_g[2, 9] = Mz2 / L
        k_g[2, 10] = -Fx2 / 10.0
        k_g[2, 11] = -Mx2 / L
        k_g[3, 4] = -1.0 * (2.0 * Mz1 - Mz2) / 6.0
        k_g[3, 5] = (2.0 * My1 - My2) / 6.0
        k_g[3, 7] = -My1 / L
        k_g[3, 8] = -Mz1 / L
        k_g[3, 9] = -Fx2 * I_rho / (A * L)
        k_g[3, 10] = -1.0 * (Mz1 + Mz2) / 6.0
        k_g[3, 11] = (My1 + My2) / 6.0
        k_g[4, 7] = -Mx2 / L
        k_g[4, 8] = Fx2 / 10.0
        k_g[4, 9] = -1.0 * (Mz1 + Mz2) / 6.0
        k_g[4, 10] = -Fx2 * L / 30.0
        k_g[4, 11] = Mx2 / 2.0
        k_g[5, 7] = -Fx2 / 10.0
        k_g[5, 8] = -Mx2 / L
        k_g[5, 9] = (My1 + My2) / 6.0
        k_g[5, 10] = -Mx2 / 2.0
        k_g[5, 11] = -Fx2 * L / 30.0
        k_g[7, 9] = -My2 / L
        k_g[7, 10] = Mx2 / L
        k_g[7, 11] = -Fx2 / 10.0
        k_g[8, 9] = -Mz2 / L
        k_g[8, 10] = Fx2 / 10.0
        k_g[8, 11] = Mx2 / L
        k_g[9, 10] = (Mz1 - 2.0 * Mz2) / 6.0
        k_g[9, 11] = -1.0 * (My1 - 2.0 * My2) / 6.0
        # add in the symmetric lower triangle
        k_g = k_g + k_g.transpose()
        # add diagonal terms
        k_g[0, 0] = Fx2 / L
        k_g[1, 1] = 6.0 * Fx2 / (5.0 * L)
        k_g[2, 2] = 6.0 * Fx2 / (5.0 * L)
        k_g[3, 3] = Fx2 * I_rho / (A * L)
        k_g[4, 4] = 2.0 * Fx2 * L / 15.0
        k_g[5, 5] = 2.0 * Fx2 * L / 15.0
        k_g[6, 6] = Fx2 / L
        k_g[7, 7] = 6.0 * Fx2 / (5.0 * L)
        k_g[8, 8] = 6.0 * Fx2 / (5.0 * L)
        k_g[9, 9] = Fx2 * I_rho / (A * L)
        k_g[10, 10] = 2.0 * Fx2 * L / 15.0
        k_g[11, 11] = 2.0 * Fx2 * L / 15.0
        return k_g
    def _assemble_global_geometric_stiffness_matrix(self, U: np.ndarray) -> np.ndarray:
        n_nodes = len(self.nodes)
        n_dofs = n_nodes * 6  # 6 DOFs per node
        K_geo_global = np.zeros((n_dofs, n_dofs))

        for element in self.elements:
            # Compute internal forces
            int_forces = self.compute_member_forces(element, U)
            Fx1, Fy1, Fz1, Mx1, My1, Mz1, Fx2, Fy2, Fz2, Mx2, My2, Mz2 = int_forces

            # Get node coordinates
            n1_loc = np.array([element.node1.x, element.node1.y, element.node1.z])
            n2_loc = np.array([element.node2.x, element.node2.y, element.node2.z])
            
            # Compute element length
            L = np.linalg.norm(n2_loc - n1_loc)

            # Compute local geometric stiffness matrix
            k_geo = self.local_geometric_stiffness_matrix_3D_beam(
                L, element.A, element.I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2
            )

            # Compute transformation matrix
            gamma = self.rotation_matrix_3D(
                n1_loc[0], n1_loc[1], n1_loc[2],
                n2_loc[0], n2_loc[1], n2_loc[2],
                element.local_z
            )
            T = self.transformation_matrix_3D(gamma)

            # Transform to global coordinates
            k_geo_global = T.T @ k_geo @ T

            # Assemble into global matrix
            dofs = np.array([
                element.node1.id * 6, element.node1.id * 6 + 1, element.node1.id * 6 + 2,
                element.node1.id * 6 + 3, element.node1.id * 6 + 4, element.node1.id * 6 + 5,
                element.node2.id * 6, element.node2.id * 6 + 1, element.node2.id * 6 + 2,
                element.node2.id * 6 + 3, element.node2.id * 6 + 4, element.node2.id * 6 + 5
            ])

            for i in range(12):
                for j in range(12):
                    K_geo_global[dofs[i], dofs[j]] += k_geo_global[i, j]

        return K_geo_global
    def _get_free_dofs(self):
        free_dofs = list(range(self.ndof))
        for bc in self.bcs:
            start_dof = bc.node.id * 6
            if bc.ux: free_dofs.remove(start_dof)
            if bc.uy: free_dofs.remove(start_dof + 1)
            if bc.uz: free_dofs.remove(start_dof + 2)
            if bc.rx: free_dofs.remove(start_dof + 3)
            if bc.ry: free_dofs.remove(start_dof + 4)
            if bc.rz: free_dofs.remove(start_dof + 5)
        return free_dofs
    def compute_member_forces(self, element: Element, U: np.ndarray) -> np.ndarray:
        """Compute member internal forces and moments in local coordinates."""
        # Get node coordinates
        node1_loc = np.array([element.node1.x, element.node1.y, element.node1.z])
        node2_loc = np.array([element.node2.x, element.node2.y, element.node2.z])

        # Compute element length
        L = np.linalg.norm(node2_loc - node1_loc)

        # Compute local stiffness matrix
        k_local = self.local_elastic_stiffness_matrix_3D_beam(
            element.E, element.nu, element.A, L, element.Iy, element.Iz, element.J
        )

        # Compute transformation matrix
        gamma = self.rotation_matrix_3D(
            node1_loc[0], node1_loc[1], node1_loc[2],
            node2_loc[0], node2_loc[1], node2_loc[2],
            element.local_z
        )
        T = self.transformation_matrix_3D(gamma)

        # Get element DOFs
        dofs = np.array([
            element.node1.id * 6, element.node1.id * 6 + 1, element.node1.id * 6 + 2,
            element.node1.id * 6 + 3, element.node1.id * 6 + 4, element.node1.id * 6 + 5,
            element.node2.id * 6, element.node2.id * 6 + 1, element.node2.id * 6 + 2,
            element.node2.id * 6 + 3, element.node2.id * 6 + 4, element.node2.id * 6 + 5
        ])

        # Extract element displacements
        disp_cur = U[dofs]

        # Transform to local coordinates
        disp_local = T @ disp_cur

        # Compute internal forces
        int_forces = k_local @ disp_local

        return int_forces
    def solve_critical_buckling_load(self,U):
        # Solve the linear static problem
        self.solved_disp=U

        # Compute internal forces for each element
        self.internal_forces = {element.id: self.compute_member_forces(element, self.solved_disp) 
                                for element in self.elements}

        # Assemble global stiffness matrices
        K_elastic_global = self._assemble_global_stiffness_matrix()
        K_geo_global = self._assemble_global_geometric_stiffness_matrix(self.solved_disp)

        # Identify known (constrained) DOFs
        known_dofs = []
        for bc in self.bcs:
            node_id = bc.node.id
            dofs = [node_id * 6 + i for i in range(6)]
            if bc.ux: known_dofs.append(dofs[0])
            if bc.uy: known_dofs.append(dofs[1])
            if bc.uz: known_dofs.append(dofs[2])
            if bc.rx: known_dofs.append(dofs[3])
            if bc.ry: known_dofs.append(dofs[4])
            if bc.rz: known_dofs.append(dofs[5])

        # Identify unknown (free) DOFs
        n_dofs = K_elastic_global.shape[0]
        unknown_dofs = [i for i in range(n_dofs) if i not in known_dofs]

        # Extract submatrices
        K_e_ff = K_elastic_global[np.ix_(unknown_dofs, unknown_dofs)]
        K_g_ff = K_geo_global[np.ix_(unknown_dofs, unknown_dofs)]
        #print(K_e_ff)
        #print(K_g_ff)
        K_e_kk = K_elastic_global[np.ix_(known_dofs, known_dofs)]
        K_g_kk = K_geo_global[np.ix_(known_dofs, known_dofs)]

        # Solve the generalized eigenvalue problem
        eigvals, eigvecs = scipy.linalg.eig(K_e_ff, -K_g_ff)
        
        # Get the real parts of the eigenvalues
        real_eigvals = np.real(eigvals)
        
        # Filter for positive eigenvalues
        positive_eigvals = real_eigvals[real_eigvals > 0]
        
        if len(positive_eigvals) > 0:
            # Find the smallest positive eigenvalue
            critical_load_factor = np.min(positive_eigvals)
            
            # Find the index of the critical eigenvalue in the original array
            critical_index = np.where(np.isclose(real_eigvals, critical_load_factor))[0][0]
            
            # Get the corresponding eigenvector
            critical_mode_free = np.real(eigvecs[:, critical_index])
            
            # Normalize the eigenvector
            critical_mode_free = critical_mode_free / np.linalg.norm(critical_mode_free)

            # Expand the critical mode to include all DOFs
            critical_mode = np.zeros(self.ndof)
            critical_mode[unknown_dofs] = critical_mode_free

            return critical_load_factor, critical_mode
        else:
            return None, None
    def plot_deformed_shape(self, U: np.ndarray, buckling_mode: np.ndarray = None, scale: float = 1, num_points: int = 100,
                            show_original: bool = True, show_deformed: bool = True, show_buckling: bool = True,show=True):
        """
        Plot the interpolated deformed shape and buckling mode of the structure using Hermite shape functions.
        
        Parameters:
        - U: Displacement vector
        - buckling_mode: Buckling mode vector (optional)
        - scale: Scaling factor for displacements
        - num_points: Number of points for interpolation
        - show_original: Whether to show the original shape
        - show_deformed: Whether to show the deformed shape
        - show_buckling: Whether to show the buckling mode
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Calculate max displacement for adaptive scaling
        max_disp = np.max(np.abs(U.reshape(-1, 6)[:, :3]))
        adaptive_scale = scale * 0.1 / max_disp if max_disp > 0 else scale

        def hermite_shape_functions(xi):
            """Calculate Hermite shape functions for a given normalized coordinate xi."""
            N1 = 1 - 3*xi**2 + 2*xi**3
            N2 = xi - 2*xi**2 + xi**3
            N3 = 3*xi**2 - 2*xi**3
            N4 = -xi**2 + xi**3
            return np.array([N1, N2, N3, N4])

        def interpolate_beam_deformation(x1, y1, z1, x2, y2, z2, u1, u2, theta1, theta2, mode='deformed'):
            """Interpolate the beam deformation using Hermite shape functions."""
            L = np.linalg.norm(np.array([x2, y2, z2]) - np.array([x1, y1, z1]))
            xi = np.linspace(0, 1, num_points)
            x = np.zeros(num_points)
            y = np.zeros(num_points)
            z = np.zeros(num_points)
            
            for i, xi_i in enumerate(xi):
                N = hermite_shape_functions(xi_i)
                
                if mode == 'deformed':
                    disp = adaptive_scale * np.array([u1, u2]).flatten()
                    rot = adaptive_scale * np.array([theta1, theta2]).flatten()
                else:  # buckling mode
                    disp = adaptive_scale * np.array([u1, u2]).flatten()
                    rot = adaptive_scale * np.array([theta1, theta2]).flatten()
                
                x[i] = (1-xi_i)*x1 + xi_i*x2 + N[0]*disp[0] + N[2]*disp[3] + L*N[1]*rot[1] + L*N[3]*rot[4]
                y[i] = (1-xi_i)*y1 + xi_i*y2 + N[0]*disp[1] + N[2]*disp[4] + L*N[1]*rot[2] + L*N[3]*rot[5]
                z[i] = (1-xi_i)*z1 + xi_i*z2 + N[0]*disp[2] + N[2]*disp[5] - L*N[1]*rot[0] - L*N[3]*rot[3]
            
            return x, y, z

        # Plot nodes
        for node in self.nodes:
            x, y, z = node.coordinates
            
            if show_original:
                ax.scatter(x, y, z, c='b', marker='o', s=30, label='Original' if node == self.nodes[0] else "")
            
            if show_deformed:
                u = adaptive_scale * U[node.id * 6 : node.id * 6 + 3]
                deformed_pos = np.array([x + u[0], y + u[1], z + u[2]])
                ax.scatter(*deformed_pos, c='r', marker='o', s=30, label='Deformed' if node == self.nodes[0] else "")

            if show_buckling and buckling_mode is not None:
                b = adaptive_scale * buckling_mode[node.id * 6 : node.id * 6 + 3]
                buckling_pos = np.array([x + b[0], y + b[1], z + b[2]])
                ax.scatter(*buckling_pos, c='g', marker='o', s=30, label='Buckling Mode' if node == self.nodes[0] else "")

        # Plot elements
        for element in self.elements:
            x1, y1, z1 = element.node1.coordinates
            x2, y2, z2 = element.node2.coordinates
            
            if show_original:
                ax.plot([x1, x2], [y1, y2], [z1, z2], 'b-')

            if show_deformed:
                u1 = U[element.node1.id * 6 : element.node1.id * 6 + 3]
                u2 = U[element.node2.id * 6 : element.node2.id * 6 + 3]
                theta1 = U[element.node1.id * 6 + 3 : element.node1.id * 6 + 6]
                theta2 = U[element.node2.id * 6 + 3 : element.node2.id * 6 + 6]

                x_def, y_def, z_def = interpolate_beam_deformation(x1, y1, z1, x2, y2, z2, u1, u2, theta1, theta2, mode='deformed')
                ax.plot(x_def, y_def, z_def, 'r-')

            if show_buckling and buckling_mode is not None:
                b1 = buckling_mode[element.node1.id * 6 : element.node1.id * 6 + 3]
                b2 = buckling_mode[element.node2.id * 6 : element.node2.id * 6 + 3]
                theta_b1 = buckling_mode[element.node1.id * 6 + 3 : element.node1.id * 6 + 6]
                theta_b2 = buckling_mode[element.node2.id * 6 + 3 : element.node2.id * 6 + 6]

                x_buck, y_buck, z_buck = interpolate_beam_deformation(x1, y1, z1, x2, y2, z2, b1, b2, theta_b1, theta_b2, mode='buckling')
                ax.plot(x_buck, y_buck, z_buck, 'g-')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Structure Visualization')
        ax.legend()

        # Set aspect ratio to be equal
        ax.set_box_aspect((np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())))

        plt.tight_layout()
        if show:
            plt.show()
        return fig, ax