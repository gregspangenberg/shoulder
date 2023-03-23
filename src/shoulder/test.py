import numpy as np

# Define beam properties
E = 200e9  # Young's modulus (Pa)
I = 8.33e-6  # Moment of inertia (m^4)
L = 1.0  # Length of beam (m)

# Define element properties
n_elem = 2  # Number of elements
elem_length = L / n_elem
k = 12 * E * I / elem_length**3
elem_stiffness = np.array(
    [
        [k, 6 * k, -k, 6 * k],
        [6 * k, 4 * elem_length**2 * k, -6 * k, 2 * elem_length**2 * k],
        [-k, -6 * k, k, -6 * k],
        [6 * k, 2 * elem_length**2 * k, -6 * k, 4 * elem_length**2 * k],
    ]
)
# Assemble global stiffness matrix
n_nodes = n_elem + 1  # Number of nodes
K = np.zeros((n_nodes * 2, n_nodes * 2))
for i in range(n_elem):
    dof_start = i * 2
    dof_end = i * 2 + 3
    K[dof_start : dof_end + 1, dof_start : dof_end + 1] += elem_stiffness

# Apply boundary conditions
fixed_dof = [0, 1]
free_dof = list(range(2, n_nodes * 2))
K_ff = K[np.ix_(free_dof, free_dof)]
K_fi = K[np.ix_(free_dof, fixed_dof)]
K_if = K[np.ix_(fixed_dof, free_dof)]
K_ii = K[np.ix_(fixed_dof, fixed_dof)]
F = np.zeros((n_nodes * 2, 1))
F[-2] = -1000

# Solve for nodal displacements
U = np.zeros((n_nodes * 2, 1))
U[free_dof] = np.linalg.solve(K_ff, F[free_dof] - np.dot(K_fi, U[fixed_dof]))

# Calculate element forces and stresses
forces = np.zeros((n_elem, 1))
stresses = np.zeros((n_elem, 1))
for i in range(n_elem):
    U_elem = U[i * 2 : i * 2 + 3]
    forces[i] = elem_stiffness.dot(U_elem)
    stresses[i] = forces[i] / (E * I / elem_length)

# Print results
print("Nodal displacements: ")
print(U)
print("Element forces: ")
print(forces)
print("Element stresses: ")
print(stresses)
