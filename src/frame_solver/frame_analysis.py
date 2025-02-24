import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        max_iterations = 10
        tolerance = 1e-6
        
        for iteration in range(max_iterations):
            K = self._assemble_global_stiffness_matrix()
            F = self._assemble_load_vector()
            K_mod, F_mod = self._apply_boundary_conditions(K, F)
            
            U_mod = np.linalg.solve(K_mod, F_mod)
            U = self._recover_full_displacement_vector(U_mod)
            
            if iteration < max_iterations - 1:
                self._update_geometric_stiffness(U)
            
            if iteration > 0 and np.linalg.norm(U - U_prev) < tolerance:
                break
            
            U_prev = U.copy()
        
        R = K @ U - F
        return U, R

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
        return k_e + k_g

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
    def rotation_matrix_3D(self,x1, y1, z1, x2, y2, z2, v_temp=None):
        L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        local_x = np.array([(x2 - x1) / L, (y2 - y1) / L, (z2 - z1) / L])
        
        if v_temp is None:
            v_temp = np.array([0, 0, 1.0]) if np.isclose(local_x[0], 0.0) and np.isclose(local_x[1], 0.0) else np.array([0, 1.0, 0.0])
        
        local_y = np.cross(v_temp, local_x)
        local_y /= np.linalg.norm(local_y)
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



