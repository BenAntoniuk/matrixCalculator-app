import streamlit as st
import numpy as np
# from scipy import linalg I want this but not working rn looking at later
import pandas as pd


# --- Header banner ---
st.markdown(
    """
    <div style="background-color:#0A2647; padding:15px; border-radius:10px; text-align:center;">
        <h1 style="color:white;">Matrix Calculator</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Mode selection dropdown ---
mode = st.selectbox(
    "Choose Mode:",
    ["Classroom Mode", "Special Matrix Identifier"],
    index=0,
    key="mode_selector"
)

# --- Matrix input helper ---
def get_matrix(name):
    st.subheader(f"Matrix {name}")
    rows = st.number_input(f"Number of rows for {name}", min_value=1, max_value=6, value=2, key=f"rows_{name}")
    cols = st.number_input(f"Number of columns for {name}", min_value=1, max_value=6, value=2, key=f"cols_{name}")
    default_data = np.zeros((rows, cols))
    df = pd.DataFrame(default_data, dtype=float)
    st.write(f"Enter values for {name}:")
    matrix_input = st.data_editor(df, num_rows="dynamic", key=f"editor_{name}")
    return matrix_input.to_numpy()

# --- Helper: check matrix properties ---
def check_properties(M, name="Matrix"):
    st.subheader(f"Results for {name}")
    rows, cols = M.shape
    square = rows == cols
    atol = 1e-8

    if not square:
        st.info("Matrix is not square, so some checks are skipped.")

    st.write(f"Shape: {rows} × {cols}")

    # --- Sparse ---
    nnz = np.count_nonzero(np.abs(M) > atol)
    sparsity = 1 - nnz / M.size
    if sparsity > 0.5:
        st.success(f"✅ Sparse matrix ({sparsity*100:.1f}% zeros)")

    # --- Hollow (zero diagonal) ---
    if np.allclose(np.diag(M), 0, atol=atol):
        st.success("✅ Hollow matrix (zero diagonal)")

    # --- Triangular (square only) ---
    if square:
        if np.allclose(M, np.triu(M), atol=atol):
            st.success("✅ Upper triangular matrix")
        if np.allclose(M, np.tril(M), atol=atol):
            st.success("✅ Lower triangular matrix")

    # --- Symmetric / Persymmetric / Bisymmetric (square only) ---
    symmetric = False
    persymmetric = False
    if square:
        symmetric = np.allclose(M, M.T, atol=atol)
        # persymmetric: M[i,j] == M[n-1-j, n-1-i]
        persymmetric = np.allclose(M, np.flipud(np.fliplr(M.T)), atol=atol)

        if symmetric:
            st.success("✅ Symmetric matrix")
        if persymmetric:
            st.success("✅ Persymmetric matrix")
        if symmetric and persymmetric:
            st.success("✅ Bisymmetric matrix")

    # --- Idempotent (M^2 = M) ---
    if square and np.allclose(M @ M, M, atol=atol):
        st.success("✅ Idempotent matrix (M² = M)")

    # --- Orthogonal (square) ---
    if square:
        I = np.eye(rows)
        if np.allclose(M.T @ M, I, atol=atol):
            st.success("✅ Orthogonal matrix")

    # --- Hat (symmetric + idempotent) ---
    if square:
        if symmetric and np.allclose(M @ M, M, atol=atol):
            st.success("✅ Hat matrix (projection matrix)")

    # --- Hermitian (square, complex) ---
    if square and np.allclose(M, np.conjugate(M.T), atol=atol):
        st.success("✅ Hermitian matrix")

    # --- Positive definite / semidefinite (square, symmetric) ---
    if square and symmetric:
        try:
            eigvals = np.linalg.eigvalsh(M)
            if np.all(eigvals > atol):
                st.success("✅ Positive definite matrix (all eigenvalues > 0)")
            elif np.all(eigvals >= -atol):
                st.success("✅ Positive semidefinite matrix (all eigenvalues ≥ 0)")
        except np.linalg.LinAlgError:
            st.warning("⚠️ Could not compute eigenvalues for definiteness check")

    # --- Hadamard (±1 entries + orthogonal rows) ---
    if square:
        if np.all(np.isclose(np.abs(M), 1, atol=atol)):
            if np.allclose(M @ M.T, rows * np.eye(rows), atol=1e-6):
                st.success("✅ Hadamard matrix (±1 entries, orthogonal rows)")

    # --- Hankel (constant along anti-diagonals) ---
    # check via flipped matrix diagonals
    flipped = np.fliplr(M)
    is_hankel = True
    for k in range(-rows + 1, cols):
        diag = np.diag(flipped, k)
        if diag.size == 0:
            continue
        if not np.allclose(diag, diag[0], atol=atol):
            is_hankel = False
            break
    if is_hankel:
        st.success("✅ Hankel matrix (constant along anti-diagonals)")

    # --- Hilbert (exact pattern) ---
    if square:
        hilbert = np.fromfunction(lambda i, j: 1.0 / (i + j + 1), (rows, cols), dtype=float)
        if np.allclose(M, hilbert, atol=1e-6):
            st.success("✅ Hilbert matrix")

    # --- Lehmer (exact pattern) ---
    if square:
        lehmer = np.fromfunction(lambda i, j: np.minimum(i + 1, j + 1) / np.maximum(i + 1, j + 1),
                                 (rows, cols), dtype=float)
        if np.allclose(M, lehmer, atol=1e-6):
            st.success("✅ Lehmer matrix")

    # --- Generalized permutation (one nonzero per row & col) ---
    if square:
        nonzeros_per_row = np.sum(np.abs(M) > atol, axis=1)
        nonzeros_per_col = np.sum(np.abs(M) > atol, axis=0)
        if np.all(nonzeros_per_row == 1) and np.all(nonzeros_per_col == 1):
            st.success("✅ Generalized permutation matrix (one nonzero per row/col)")

    # --- Metzler (off-diagonal >= 0) ---
    if square:
        offdiag = M - np.diag(np.diag(M))
        if np.all(offdiag >= -atol):
            st.success("✅ Metzler matrix (off-diagonal elements ≥ 0)")

    # --- Markov (rows sum to 1, nonnegative) ---
    # works for rectangular too (row-stochastic)
    if np.all(M >= -atol) and np.allclose(M.sum(axis=1), 1, atol=atol):
        st.success("✅ Markov matrix (rows sum to 1, nonnegative)")

    # --- Bidiagonal (upper or lower) ---
    rows_idx, cols_idx = np.nonzero(np.abs(M) > atol)
    upper_bidiag = False
    lower_bidiag = False
    if rows_idx.size > 0:
        diffs = cols_idx - rows_idx
        if np.all(np.isin(diffs, [0, 1])):
            upper_bidiag = True
        if np.all(np.isin(diffs, [0, -1])):
            lower_bidiag = True
    if upper_bidiag:
        st.success("✅ Upper bidiagonal matrix")
    elif lower_bidiag:
        st.success("✅ Lower bidiagonal matrix")

    # --- Bandwidth and Band matrix (robust to empty nonzero set) ---
    if rows_idx.size == 0:
        bandwidth = 0
    else:
        bandwidth = int(np.max(np.abs(rows_idx - cols_idx)))
    # Example threshold: report if bandwidth ≤ 2 (you can change threshold)
    if bandwidth <= 2:
        st.success(f"✅ Band matrix (bandwidth ≤ {bandwidth})")

    # --- Arrowhead: only first row, first col, and diagonal may be nonzero ---
    allowed_mask = np.zeros_like(M, dtype=bool)
    allowed_mask[0, :] = True
    allowed_mask[:, 0] = True
    d = min(rows, cols)
    idx = np.arange(d)
    allowed_mask[idx, idx] = True
    # True if every entry outside allowed_mask is (near) zero
    if np.all((np.abs(M) <= atol) | allowed_mask):
        st.success("✅ Arrowhead matrix (nonzero first row/col + diagonal)")

    # --- Display eigenvalues (square only) ---
    if square:
        try:
            vals, vecs = np.linalg.eig(M)
            st.write("**Eigenvalues:**")
            st.write(np.round(vals, 6))
            st.write("**Eigenvectors:**")
            st.write(np.round(vecs, 6))
        except np.linalg.LinAlgError:
            st.error("Eigenvalue calculation failed.")


# --- Classroom Mode ---
if mode == "Classroom Mode":
    use_two_matrices = st.checkbox("Work with two matrices (A and B)?", value=False)

    A = get_matrix("A")
    B = get_matrix("B") if use_two_matrices else None

    st.write("**Matrix A:**")
    st.write(A)
    if B is not None:
        st.write("**Matrix B:**")
        st.write(B)

    # --- Operation selection ---
    if use_two_matrices:
        operation = st.selectbox("Choose an operation:", ["A × B"])
    else:
        operation = st.selectbox(
            "Choose an operation:",
            ["Transpose", "Inverse", "Multiply by Itself", "Eigenvalues",
             "Check Orthogonal", "Check Hat Matrix"]
        )

    # --- Operations ---
    if operation == "Transpose":
        st.write("**Transpose:**")
        st.write(A.T)

    elif operation == "Inverse":
        try:
            st.write("**Inverse:**")
            st.write(np.linalg.inv(A))
        except np.linalg.LinAlgError:
            st.error("Matrix is singular and cannot be inverted.")

    elif operation == "Multiply by Itself":
        try:
            st.write("**A × A:**")
            st.write(np.dot(A, A))
        except Exception as e:
            st.error(f"Error: {e}")

    elif operation == "Eigenvalues":
        try:
            vals, vecs = np.linalg.eig(A)
            st.write("**Eigenvalues:**")
            st.write(vals)
            st.write("**Eigenvectors:**")
            st.write(vecs)
        except np.linalg.LinAlgError:
            st.error("Eigenvalue calculation failed.")

    elif operation == "Check Orthogonal":
        if A.shape[0] != A.shape[1]:
            st.error("Matrix must be square to check orthogonality.")
        else:
            I = np.eye(A.shape[0])
            test = A.T @ A
            if np.allclose(test, I, atol=1e-8):
                st.success("✅ Matrix A is orthogonal.")
            else:
                st.warning("❌ Matrix A is NOT orthogonal.")

    elif operation == "Check Hat Matrix":
        if A.shape[0] != A.shape[1]:
            st.error("Matrix must be square to check if it's a hat matrix.")
        else:
            symmetric = np.allclose(A, A.T, atol=1e-8)
            idempotent = np.allclose(A @ A, A, atol=1e-8)

            if symmetric and idempotent:
                st.success("✅ Matrix A is a hat matrix.")
            else:
                if not symmetric and not idempotent:
                    st.warning("❌ Matrix A is NOT a hat matrix (fails symmetry and idempotence).")
                elif not symmetric:
                    st.warning("❌ Matrix A is NOT a hat matrix (fails symmetry).")
                else:
                    st.warning("❌ Matrix A is NOT a hat matrix (fails idempotence).")

    elif operation == "A × B":
        try:
            if A.shape[1] != B.shape[0]:
                st.error("Number of columns in A must equal number of rows in B.")
            else:
                st.write("**A × B:**")
                st.write(np.dot(A, B))
        except Exception as e:
            st.error(f"Error: {e}")

# --- Special Matrix Identifier Mode ---
elif mode == "Special Matrix Identifier":
    use_two_matrices = st.checkbox("Work with two matrices (A and B)?", value=False)

    A = get_matrix("A")
    B = get_matrix("B") if use_two_matrices else None

    # Analyze A
    check_properties(A, "Matrix A")

    # Analyze B (if provided)
    if B is not None:
        check_properties(B, "Matrix B")

        # Multiply A × B
        if A.shape[1] == B.shape[0]:
            C = A @ B
            st.subheader("**Result of A × B:**")
            st.write(C)
            check_properties(C, "Matrix A × B")
        else:
            st.warning("⚠️ Cannot multiply A × B (dimension mismatch).")
