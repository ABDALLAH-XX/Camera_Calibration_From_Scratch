import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg

"""Camera Calibration using Faugeras-Toscani algorithm"""

#--------------------2D and 3D Points Extraction from the 3D Chessboard--------------------
Points2D = np.array([[563, 315], [406, 204], [428, 165], [351, 326], [321, 154], [58, 244]])#, [102, 190], [426, 116]])
Points3D = np.array([[0, 7, 1], [0, 2, 2], [0, 3, 3], [2, 3, 0], [1, 0, 3], [7, 0, 2]])#, [6, 0, 3], [0, 3, 4]])

print(f"2D points shape: {Points2D.shape}")
print(f"3D points shape: {Points3D.shape}")
print("-" * 80)

#--------------------Compute the mean values for each column--------------------
Bary2D = np.mean(Points2D, axis=0)
Bary3D = np.mean(Points3D, axis=0)
print("Bary2D:\n", Bary2D, "\n")
print("Bary3D:\n", Bary3D, "\n")
print("-" * 80)

#--------------------2D & 3D Normalization of the original points--------------------
Points2DNorm = Points2D - Bary2D
Points3DNorm = Points3D - Bary3D

# Dynamic vectorization fixes the hardcoded 'range(6)' error
m2 = np.mean(np.linalg.norm(Points2DNorm, axis=1))
m3 = np.mean(np.linalg.norm(Points3DNorm, axis=1))

print("meanDist2DNorm:\n", m2, "\n")
print("meanDist3DNorm:\n", m3, "\n")

Points2DNorm *= np.sqrt(2) / m2
Points3DNorm *= np.sqrt(3) / m3

print("Normalized 2D points:\n", Points2DNorm, "\n")
print("Normalized 3D points:\n", Points3DNorm, "\n")
print("-" * 80)

#--------------------We define B & C matrices--------------------
b = []
c = []
num_points = Points2D.shape[0]

for idx in range(num_points):
    X, Y, Z = Points3DNorm[idx]
    x, y = Points2DNorm[idx]
    
    b.append([X, Y, Z, 1, 0, 0, 0, 0, -x])
    b.append([0, 0, 0, 0, X, Y, Z, 1, -y])
    
    c.append([-x * X, -x * Y, -x * Z])
    c.append([-y * X, -y * Y, -y * Z])

B = np.array(b)
C = np.array(c)

print("B Matrix:\n", B, "\n")
print("C Matrix:\n", C, "\n")
print("-" * 80)

#--------------------D Matrix--------------------
B_pinv = np.linalg.pinv(B.T @ B)
D = C.T @ C - C.T @ B @ B_pinv @ B.T @ C

eigenvalues, eigenvectors = np.linalg.eig(D)
print("D Matrix:\n", D, "\n")
print("Eigenvalues:\n", eigenvalues, "\n")

idx_min = np.argmin(eigenvalues)
print("Index of min eigenvalue: ", idx_min, "\n")

v2_raw = eigenvectors[:, idx_min]
v2 = (v2_raw / np.linalg.norm(v2_raw)).reshape(3, 1)

v1 = - B_pinv @ B.T @ C @ v2

print("v1 shape: ", v1.shape)
print("v2 shape: ", v2.shape)
print("-" * 80)

#--------------------Projection Matrix P_tilde--------------------
tmp1 = v1.ravel()
tmp2 = v2.ravel()

P_tilde = np.zeros((3, 4))
P_tilde[0, :4] = tmp1[0:4]
P_tilde[1, :4] = tmp1[4:8]
P_tilde[2, :3] = tmp2
P_tilde[2, 3] = tmp1[8]

print("P_tilde:\n", P_tilde, "\n")

#--------------------Unnormalized Matrix P--------------------
alpha = np.sqrt(2) / m2
beta = np.sqrt(3) / m3

T = np.array([[alpha, 0, -alpha * Bary2D[0]],
              [0, alpha, -alpha * Bary2D[1]],
              [0, 0, 1]])

U = np.array([[beta, 0, 0, -beta * Bary3D[0]],
              [0, beta, 0, -beta * Bary3D[1]],
              [0, 0, beta, -beta * Bary3D[2]],
              [0, 0, 0, 1]])

P = np.linalg.inv(T) @ P_tilde @ U
print("Final Denormalized Projection Matrix P:\n", P, "\n")
print("-" * 80) 

#--------------------K Extraction---------------------
# We extract the sub matrix (3x3) on P
M = P[0:3, 0:3]

# RQ decompostion to separate K (upper triangular) and R (rotation matrix)
K_estimate, R_estimate = linalg.rq(M)

# Correcting the signs on K to ensure positive focal points
for i in range(3):
    if K_estimate[i, i] < 0:
        K_estimate[:, i] = -K_estimate[:, i]

# Homogeneous normalization
K_estimate = K_estimate / K_estimate[2, 2]

print("--- INTRISIC MATRIX K ---")
print(np.round(K_estimate, 4),"\n")
print("-" * 80) 

#--------------------Validation & Evaluation Sequence--------------------
test3D = np.array([[1, 1, 0], [4, 1, 0], [0, 5, 6], [0, 6, 2], [3, 0, 2], [5, 0, 4], [0, 0, 0], [0, 0, 3], [5, 5, 0], [0, 7, 4], [2, 7, 0]])
test2D = np.array([[352, 290], [225, 318], [483, 8], [523, 244], [243, 211], [143, 134], [371, 271], [362, 148], [244, 404], [562, 135], [463, 403]])

# Build homogeneous coordinates vector matrix (4 x 11) safely
test_data = np.ones((4, test3D.shape[0]))
test_data[0:3, :] = test3D.T

projection3D = P @ test_data
projection2D = projection3D[0:2, :] / projection3D[2, :]

u_projected = projection2D[0, :]
v_projected = projection2D[1, :]

u_measured = test2D[:, 0]
v_measured = test2D[:, 1]

u_errors = u_projected - u_measured
v_errors = v_projected - v_measured

std_u = np.std(u_errors)
std_v = np.std(v_errors)

print(f"Standard deviation of the error u : {std_u:.2f} pixels")
print(f"Standard deviation of the error v : {std_v:.2f} pixels\n")

#--------------------Error Visualization Map--------------------
plt.figure(figsize=(8, 6))
scale = 10
plt.quiver(u_measured, v_measured, scale*u_errors, scale*v_errors, angles='xy', scale_units='xy', scale=1, color='blue', label='Error Vectors')
plt.scatter(u_measured, v_measured, color='red', marker='o', label='Measured Corners')
plt.title("Projection Error Vector Map (Faugeras-Toscani Calibration) — vectors scaled ×10")
plt.xlabel("u (pixels)")
plt.ylabel("v (pixels)")
plt.gca().invert_yaxis()  # Match pixel image row directions
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.savefig("reprojection_error.png", dpi=150, bbox_inches='tight')
plt.show()