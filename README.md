# Camera Calibration From Scratch (Faugeras-Toscani Method)

The goal of this project is to implement camera calibration and analyze re-projection pixel errors completely from scratch using the **Faugeras-Toscani algorithm**, avoiding any ready-to-use OpenCV calibration black-boxes (such as `cv2.calibrateCamera`).

---

## 1. Project Repository Structure

Your workspace contains only the essential files required for execution:

```text
├── .gitignore
├── camera_calibration.py    # Main pipeline: Normalization, Faugeras-Toscani core solver, evaluation
├── mire.tiff                # 3D calibration rig image (three perpendicular planes)
└── README.md                # Project documentation
```

---

## 2. Dataset & Coordinate Convention

The dataset consists of a single image `mire.tiff` representing a 3D calibration rig (*mire de calibration*).

### Setup Properties

* **Geometry:** The rig is composed of three mutually perpendicular planes intersecting at the origin $(0, 0, 0)$. Each plane contains a grid pattern of squares.
* **Scale:** By convention, each square on the grid measures exactly `1 cm × 1 cm`.
* **Orientation:**
  * X-axis: Points to the left
  * Y-axis: Points to the right
  * Z-axis: Points up (vertical axis)

As an example, counting the grid intersections from the corner origin allows mapping physical locations directly:

> A point located 4 squares left and 2 squares up on the left plane corresponds to the 3D world coordinates $(4, 0, 2)$ which is approximately $(201, 220)$ in pixels by using GIMP.

---

## 3. Mathematical Background & Normalization

To ensure numerical stability and prevent matrix ill-conditioning during SVD and inversion steps, raw 3D world points

$$
A_i = [X_i, Y_i, Z_i, 1]^T
$$

and 2D image points

$$
a_i = [u_i, v_i, 1]^T
$$

must undergo an isotropic normalization transformation:

$$
\tilde{A}_i = U A_i \quad \text{and} \quad \tilde{a}_i = T a_i
$$

### 2D Normalization Matrix ($T$)

The transformation centers the point cloud on its barycenter $(\langle x_i \rangle, \langle y_i \rangle)$ and scales the mean squared distance to $2$ ($\sqrt{2}$ average distance):

$$
T =
\begin{bmatrix}
\alpha & 0 & t_x \\
0 & \alpha & t_y \\
0 & 0 & 1
\end{bmatrix}
$$

Where $\langle \cdot \rangle$ represents the mean operator over the $n$ points:

$$
t_x = -\alpha \langle x_i \rangle
$$

$$
t_y = -\alpha \langle y_i \rangle
$$

$$
\alpha =
\frac{\sqrt{2}}
{\sqrt{\langle x_i^2 \rangle + \langle y_i^2 \rangle - (\langle x_i \rangle^2 + \langle y_i \rangle^2)}}
$$

### 3D Normalization Matrix ($U$)

Similarly, for the 3D space, centering on $(\langle X_i \rangle, \langle Y_i \rangle, \langle Z_i \rangle)$ and scaling the mean squared distance to $3$ ($\sqrt{3}$ average distance):

$$
U =
\begin{bmatrix}
\beta & 0 & 0 & t_x \\
0 & \beta & 0 & t_y \\
0 & 0 & \beta & t_z \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

$$
t_x = -\beta \langle X_i \rangle
$$

$$
t_y = -\beta \langle Y_i \rangle
$$

$$
t_z = -\beta \langle Z_i \rangle
$$

$$
\beta =
\frac{\sqrt{3}}
{\sqrt{\langle X_i^2 \rangle + \langle Y_i^2 \rangle + \langle Z_i^2 \rangle - (\langle X_i \rangle^2 + \langle Y_i \rangle^2 + \langle Z_i \rangle^2)}}
$$

---

## 4. The Faugeras-Toscani Algorithm

The parameter space of the normalized $3 \times 4$ projection matrix $\tilde{P}$ is split into two structural column vectors:

$$
v_1 = [p_{11}, p_{12}, p_{13}, p_{14}, p_{21}, p_{22}, p_{23}, p_{24}, p_{34}]^T
$$

(size $9 \times 1$)

$$
v_2 = [p_{31}, p_{32}, p_{33}]^T
$$

(size $3 \times 1$)

### Step-by-Step Pipeline

#### 1. Normalize Coordinates

Compute matrices $T$ and $U$ from the data points, then calculate all $\tilde{A}_i$ and $\tilde{a}_i$.

#### 2. Compute Component Blocks

For each correspondence

$$
\tilde{A}_i \leftrightarrow \tilde{a}_i
$$

(where $\tilde{a}_i = [x_i, y_i, 1]^T$ and $\tilde{A}_i = [X_i, Y_i, Z_i, 1]^T$), compute the $2 \times 9$ matrix $B_i$ and $2 \times 3$ matrix $C_i$:

$$
B_i =
\begin{bmatrix}
X_i & Y_i & Z_i & 1 & 0 & 0 & 0 & 0 & -x_i \\
0 & 0 & 0 & 0 & X_i & Y_i & Z_i & 1 & -y_i
\end{bmatrix}
$$

$$
C_i =
\begin{bmatrix}
-x_i X_i & -x_i Y_i & -x_i Z_i \\
-y_i X_i & -y_i Y_i & -y_i Z_i
\end{bmatrix}
$$

#### 3. Concatenate Matrix Systems

Stack all $n$ sub-matrices vertically to form the global linear blocks:

$$
B \in \mathbb{R}^{2n \times 9}
$$

and

$$
C \in \mathbb{R}^{2n \times 3}
$$

such that:

$$
B =
\begin{bmatrix}
B_1 \\
\vdots \\
B_n
\end{bmatrix},
\quad
C =
\begin{bmatrix}
C_1 \\
\vdots \\
C_n
\end{bmatrix}
$$

#### 4. Isolate the Objective Matrix $D$

Eliminate $v_1$ analytically to build the $3 \times 3$ matrix $D$:

$$
D = C^T C - C^T B (B^T B)^{-1} B^T C
$$

#### 5. Eigen-Decomposition

Compute the eigenvalues and eigenvectors of $D$.

#### 6. Extract Vector $v_2$

Isolate the eigenvector associated with the smallest eigenvalue of $D$, and normalize it so that:

$$
\|v_2\| = 1
$$

#### 7. Back-substitute for $v_1$

Solve for the remaining parameters using $v_2$:

$$
v_1 = -(B^T B)^{-1} B^T C v_2
$$

#### 8. Assemble $\tilde{P}$

Reconstruct the normalized projection matrix $\tilde{P}$ from the computed elements of $v_1$ and $v_2$.

#### 9. Denormalize

Map the matrix back to the original pixel/world physical coordinate frame:

$$
P = T^{-1} \tilde{P} U
$$

---

## 5. Script Workflow & Evaluation Tasks

The code inside `camera_calibration.py` runs a multi-stage execution layout matching the laboratory objectives.

### 1. Calibration (Matrix Estimation)

The script selects an initial subset of:

$$
n = 6
$$

point pairs distributed evenly across the rig (2 pairs per plane).

Selecting points across all 3 planes is strictly mandatory to prevent coplanarity, which would cause:

$$
B^T B
$$

to become singular and non-invertible.

### 2. Validation & Projections

To evaluate the calibration without overfitting, the script takes a validation set of:

$$
n = 10
$$

different landmark points.

It projects them onto the image space using the estimated matrix $P$:

$$

\lambda
\begin{bmatrix}
u_{proj} \\
v_{proj} \\
1
\end{bmatrix}
=
P
\begin{bmatrix}
X_{world} \\
Y_{world} \\
Z_{world} \\
1
\end{bmatrix}

$$

### 3. Precision Analysis (Error Moments)

The pipeline computes the standard deviation (2nd-order moments $\sigma_u$, $\sigma_v$) of the validation residuals:

$$

\Delta u = u_{original} - u_{proj}

$$

$$

\Delta v = v_{original} - v_{proj}

$$

The script iterates this entire process by varying the number of training point pairs from 7 up to 12, plotting the evolution of standard deviation errors to demonstrate how increasing data cardinality impacts the stability and accuracy of the linear Faugeras-Toscani solver.

---

## 6. How to Run

### Dependencies

Ensure you have Python installed along with standard scientific computing libraries:

```bash
pip install numpy opencv-python scipy matplotlib
```

> **Note:** OpenCV is solely utilized for reading the TIFF file format and rendering corner visuals; all structural matrix routines are implemented from scratch.

### Run Execution

Execute the main script via terminal:

```bash
python camera_calibration.py
```