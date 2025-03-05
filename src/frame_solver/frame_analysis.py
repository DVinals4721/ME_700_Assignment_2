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
            
            print(f"Iteration {iteration}")
            print("K shape:", K.shape)
            print("F shape:", F.shape)
            print("K condition number:", np.linalg.cond(K))
            print("F norm:", np.linalg.norm(F))
            
            K_mod, F_mod = self._apply_boundary_conditions(K, F)
            
            if np.any(np.isnan(K_mod)) or np.any(np.isinf(K_mod)):
                raise ValueError("K_mod contains NaN or Inf values")
            if np.any(np.isnan(F_mod)) or np.any(np.isinf(F_mod)):
                raise ValueError("F_mod contains NaN or Inf values")
            
            try:
                U_mod = np.linalg.solve(K_mod, F_mod)
            except np.linalg.LinAlgError as e:
                print(f"Linear algebra error: {e}")
                print("K_mod:")
                print(K_mod)
                print("F_mod:")
                print(F_mod)
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
        k_g = self.local_geometric_stiffness_matrix_3D_beam_without_interaction_terms(
            L, element.A, element.I_rho, element.Fx2
        )
        if self.kg_included:
            return k_e + k_g
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

    def local_geometric_stiffness_matrix_3D_beam_without_interaction_terms(self,L, A, I_rho, Fx2):
        k_g = np.zeros((12, 12))
        # Upper triangle off-diagonal terms
        k_g[0, 6] = -Fx2 / L
        k_g[1, 5] = k_g[1, 11] = Fx2 / 10.0
        k_g[1, 7] = -6.0 * Fx2 / (5.0 * L)
        k_g[2, 4] = k_g[2, 10] = -Fx2 / 10.0
        k_g[2, 8] = -6.0 * Fx2 / (5.0 * L)
        k_g[3, 9] = -Fx2 * I_rho / (A * L)
        k_g[4, 8] = k_g[4, 10] = Fx2 / 10.0
        k_g[4, 10] = -Fx2 * L / 30.0
        k_g[5, 7] = k_g[5, 11] = -Fx2 / 10.0
        k_g[5, 11] = -Fx2 * L / 30.0
        k_g[7, 11] = -Fx2 / 10.0
        k_g[8, 10] = Fx2 / 10.0
        # Add symmetric lower triangle
        k_g = k_g + k_g.T
        # Diagonal terms
        k_g[0, 0] = k_g[6, 6] = Fx2 / L
        k_g[1, 1] = k_g[2, 2] = k_g[7, 7] = k_g[8, 8] = 6.0 * Fx2 / (5.0 * L)
        k_g[3, 3] = k_g[9, 9] = Fx2 * I_rho / (A * L)
        k_g[4, 4] = k_g[5, 5] = k_g[10, 10] = k_g[11, 11] = 2.0 * Fx2 * L / 15.0
        return k_g
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

        # Filter and sort eigenvalues and eigenvectors
        real_pos_mask = np.isreal(eigvals) & (eigvals > 0)
        filtered_eigvals = np.real(eigvals[real_pos_mask])
        filtered_eigvecs = eigvecs[:, real_pos_mask]

        if len(filtered_eigvals) > 0:
            # Find the index of the smallest positive real eigenvalue
            min_index = np.argmin(filtered_eigvals)
            
            # Get the smallest positive real eigenvalue and its corresponding eigenvector
            critical_load_factor = filtered_eigvals[min_index]
            critical_mode_free = filtered_eigvecs[:, min_index]
            
            # Normalize the eigenvector
            critical_mode_free = critical_mode_free / np.linalg.norm(critical_mode_free)

            # Expand the critical mode to include all DOFs
            critical_mode = np.zeros(self.ndof)
            critical_mode[unknown_dofs] = critical_mode_free

            return critical_load_factor, critical_mode
        else:
            return None, None
    def plot_member_forces(self, element: Element, forces: np.ndarray):
        """Plot member internal forces and moments in local coordinates."""
        L = np.linalg.norm(element.node2.coordinates - element.node1.coordinates)
        x = np.linspace(0, L, 100)
        
        fig, axs = plt.subplots(3, 2, figsize=(12, 15))
        titles = ['Axial Force', 'Shear Force Y', 'Shear Force Z', 'Torsion', 'Bending Moment Y', 'Bending Moment Z']
        
        for i, (ax, title) in enumerate(zip(axs.flat, titles)):
            if i in [0, 3]:  # Constant along length
                ax.plot([0, L], [forces[i], forces[i+6]])
            elif i in [1, 2]:  # Linear variation
                ax.plot(x, np.interp(x, [0, L], [forces[i], forces[i+6]]))
            else:  # Quadratic variation
                a = (forces[i+6] - forces[i]) / L
                b = forces[i]
                ax.plot(x, a * x**2 / 2 + b * x)
            
            ax.set_title(title)
            ax.set_xlabel('Length')
            ax.set_ylabel('Force/Moment')
        
        plt.tight_layout()
        plt.show()

    def hermite_polynomials(self,xi):
        """
        Calculate Hermite polynomials for beam elements.
        xi: Local coordinate (-1 to 1)
        """
        H1 = 0.25 * (1 - xi)**2 * (2 + xi)
        H2 = 0.25 * (1 - xi)**2 * (1 + xi)
        H3 = 0.25 * (1 + xi)**2 * (2 - xi)
        H4 = 0.25 * (1 + xi)**2 * (xi - 1)
        return np.array([H1, H2, H3, H4])
    def plot_original_frame(self):
        """Plot the original frame without deformations."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot all node positions
        for node in self.nodes:
            x, y, z = node.coordinates
            ax.scatter(x, y, z, c='b', marker='o', s=30)

        # Plot all elements
        for element in self.elements:
            x1, y1, z1 = element.node1.coordinates
            x2, y2, z2 = element.node2.coordinates
            
            # Original shape
            ax.plot([x1, x2], [y1, y2], [z1, z2], 'b-')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Original Frame')

        # Set aspect ratio to be equal
        ax.set_box_aspect((np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())))

        plt.tight_layout()
        plt.show()
    def plot_deformed_shape(self, U: np.ndarray, buckling_mode: np.ndarray = None, scale: float = 1, num_points: int = 100):
        """Plot the interpolated deformed shape and buckling mode of the structure using cubic interpolation."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        s = np.linspace(0, 1, num_points)

        # Dictionary to store deformed node positions
        deformed_nodes = {}
        buckling_nodes = {}

        # Calculate max displacement for adaptive scaling
        max_disp = np.max(np.abs(U.reshape(-1, 6)[:, :3]))
        adaptive_scale = scale * 0.1 / max_disp if max_disp > 0 else scale

        # First, calculate and plot all node positions
        for node in self.nodes:
            x, y, z = node.coordinates
            u = adaptive_scale * U[node.id * 6 : node.id * 6 + 3]
            deformed_pos = np.array([x + u[0], y + u[1], z + u[2]])
            deformed_nodes[node.id] = deformed_pos

            # Plot original and deformed node positions
            ax.scatter(x, y, z, c='b', marker='o', s=30, label='Original' if node == self.nodes[0] else "")
            ax.scatter(*deformed_pos, c='r', marker='o', s=30, label='Deformed' if node == self.nodes[0] else "")

            if buckling_mode is not None:
                b = adaptive_scale * buckling_mode[node.id * 6 : node.id * 6 + 3]
                buckling_pos = np.array([x + b[0], y + b[1], z + b[2]])
                buckling_nodes[node.id] = buckling_pos
                ax.scatter(*buckling_pos, c='g', marker='o', s=30, label='Buckling Mode' if node == self.nodes[0] else "")

        # Now plot the elements with cubic interpolation
        for element in self.elements:
            x1, y1, z1 = element.node1.coordinates
            x2, y2, z2 = element.node2.coordinates
            
            # Original shape
            ax.plot([x1, x2], [y1, y2], [z1, z2], 'b-')

            # Deformed shape
            u1 = deformed_nodes[element.node1.id]
            u2 = deformed_nodes[element.node2.id]
            theta1 = U[element.node1.id * 6 + 3 : element.node1.id * 6 + 6]
            theta2 = U[element.node2.id * 6 + 3 : element.node2.id * 6 + 6]

            # Element length and direction
            L = np.linalg.norm(np.array([x2, y2, z2]) - np.array([x1, y1, z1]))
            direction = np.array([x2 - x1, y2 - y1, z2 - z1]) / L

            # Cubic interpolation for deformed shape
            def cubic_interp(s, p0, p1, v0, v1):
                return (2*s**3 - 3*s**2 + 1) * p0 + \
                    (s**3 - 2*s**2 + s) * L * v0 + \
                    (-2*s**3 + 3*s**2) * p1 + \
                    (s**3 - s**2) * L * v1
        
            x_def = cubic_interp(s, u1[0], u2[0], adaptive_scale * np.cross(theta1, direction)[0], adaptive_scale * np.cross(theta2, direction)[0])
            y_def = cubic_interp(s, u1[1], u2[1], adaptive_scale * np.cross(theta1, direction)[1], adaptive_scale * np.cross(theta2, direction)[1])
            z_def = cubic_interp(s, u1[2], u2[2], adaptive_scale * np.cross(theta1, direction)[2], adaptive_scale * np.cross(theta2, direction)[2])

            ax.plot(x_def, y_def, z_def, 'r-')

            # Buckling mode shape
            if buckling_mode is not None:
                b1 = buckling_nodes[element.node1.id]
                b2 = buckling_nodes[element.node2.id]
                theta_b1 = buckling_mode[element.node1.id * 6 + 3 : element.node1.id * 6 + 6]
                theta_b2 = buckling_mode[element.node2.id * 6 + 3 : element.node2.id * 6 + 6]

                # Cubic interpolation for buckling mode
                x_buck = cubic_interp(s, b1[0], b2[0], adaptive_scale * np.cross(theta_b1, direction)[0], adaptive_scale * np.cross(theta_b2, direction)[0])
                y_buck = cubic_interp(s, b1[1], b2[1], adaptive_scale * np.cross(theta_b1, direction)[1], adaptive_scale * np.cross(theta_b2, direction)[1])
                z_buck = cubic_interp(s, b1[2], b2[2], adaptive_scale * np.cross(theta_b1, direction)[2], adaptive_scale * np.cross(theta_b2, direction)[2])

                ax.plot(x_buck, y_buck, z_buck, 'g-')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Deformed Shape and Buckling Mode (Cubic Interpolation)')
        ax.legend()

        # Set aspect ratio to be equal
        ax.set_box_aspect((np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())))

        plt.tight_layout()
        plt.show()