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
    st.subheader(f"🔎 Results for {name}")

    rows, cols = M.shape
    square = rows == cols
    atol = 1e-8  # tolerance for float comparisons

    # --- Basic Info ---
    st.write(f"Shape: {rows} × {cols}")

    # --- Sparse check ---
    sparsity = 1 - (np.count_nonzero(M) / M.size)
    if sparsity > 0.9:
        st.success(f"✅ Sparse matrix ({sparsity*100:.1f}% zeros)")

    # --- Hollow check ---
    if np.allclose(np.diag(M), 0, atol=atol):
        st.success("✅ Hollow matrix (zero diagonal)")

    # --- Triangular checks ---
    if square:
        if np.allclose(M, np.triu(M), atol=atol):
            st.success("✅ Upper triangular matrix")
        if np.allclose(M, np.tril(M), atol=atol):
            st.success("✅ Lower triangular matrix")

    # --- Symmetric / Persymmetric / Bisymmetric ---
    if square:
        symmetric = np.allclose(M, M.T, atol=atol)
        persymmetric = np.allclose(M, np.flipud(np.fliplr(M.T)), atol=atol)

        if symmetric:
            st.success("✅ Symmetric matrix")
        if persymmetric:
            st.success("✅ Persymmetric matrix")
        if symmetric and persymmetric:
            st.success("✅ Bisymmetric matrix")

    # --- Idempotent ---
    if square and np.allclose(M @ M, M, atol=atol):
        st.success("✅ Idempotent matrix (M² = M)")

    # --- Orthogonal ---
    if square:
        I = np.eye(rows)
        if np.allclose(M.T @ M, I, atol=atol):
            st.success("✅ Orthogonal matrix")

    # --- Hat matrix (symmetric + idempotent) ---
    if square:
        symmetric = np.allclose(M, M.T, atol=atol)
        idempotent = np.allclose(M @ M, M, atol=atol)
        if symmetric and idempotent:
            st.success("✅ Hat matrix (projection matrix)")

    # --- Hermitian ---
    if square and np.allclose(M, np.conjugate(M.T), atol=atol):
        st.success("✅ Hermitian matrix")

    # --- Positive (semi)definite ---
    if square:
        try:
            eigvals = np.linalg.eigvalsh(M)
            if np.all(eigvals > atol):
                st.success("✅ Positive definite matrix (all eigenvalues > 0)")
            elif np.all(eigvals >= -atol):
                st.success("✅ Positive semidefinite matrix (all eigenvalues ≥ 0)")
        except np.linalg.LinAlgError:
            st.warning("⚠️ Could not compute eigenvalues for definiteness check")

    # --- Hadamard ---
    if square and np.all(np.isin(M, [-1, 1])):
        if np.allclose(M @ M.T, rows * np.eye(rows), atol=atol):
            st.success("✅ Hadamard matrix (±1 entries, orthogonal rows)")

    # --- Hankel ---
    if np.allclose(M, np.fliplr(np.triu(np.fliplr(M))), atol=atol):
        # Check if constant along anti-diagonals
        is_hankel = all(np.allclose(np.diag(np.fliplr(M), k), np.diag(np.fliplr(M), k)[0], atol=atol)
                        for k in range(-rows + 1, cols))
        if is_hankel:
            st.success("✅ Hankel matrix (constant along anti-diagonals)")

    # --- Hilbert ---
    if square and np.allclose(M, [[1 / (i + j + 1) for j in range(cols)] for i in range(rows)], atol=1e-6):
        st.success("✅ Hilbert matrix")

    # --- Lehmer ---
    if square and np.allclose(M, [[min(i + 1, j + 1) / max(i + 1, j + 1) for j in range(cols)] for i in range(rows)], atol=1e-6):
        st.success("✅ Lehmer matrix")

    # --- Generalized permutation ---
    if square:
        nonzeros_per_row = np.sum(M != 0, axis=1)
        nonzeros_per_col = np.sum(M != 0, axis=0)
        if np.all(nonzeros_per_row == 1) and np.all(nonzeros_per_col == 1):
            st.success("✅ Generalized permutation matrix (one nonzero per row/col)")

    # --- Metzler ---
    if square and np.all(M - np.diag(np.diag(M)) >= -atol):
        st.success("✅ Metzler matrix (off-diagonal elements ≥ 0)")

    # --- Markov ---
    if np.all(M >= -atol) and np.allclose(M.sum(axis=1), 1, atol=atol):
        st.success("✅ Markov matrix (rows sum to 1, nonnegative)")

    # --- Bidiagonal ---
    if square:
        is_upper_bidiag = np.allclose(M, np.triu(M, -1), atol=atol) and np.allclose(M, np.triu(M, 0), atol=atol)
        is_lower_bidiag = np.allclose(M, np.tril(M, 1), atol=atol) and np.allclose(M, np.tril(M, 0), atol=atol)
        if is_upper_bidiag:
            st.success("✅ Upper bidiagonal matrix")
        elif is_lower_bidiag:
            st.success("✅ Lower bidiagonal matrix")

    # --- Band matrix (bandwidth ≤ 2 example) ---
    if square:
        bandwidth = np.max(np.abs(np.nonzero(M - np.diag(np.diag(M)))[0] - np.nonzero(M - np.diag(np.diag(M)))[1]))
        if bandwidth <= 2:
            st.success(f"✅ Band matrix (bandwidth ≤ {bandwidth})")

    # --- Arrowhead ---
    if square:
        A = M.copy()
        A[0, 0] = 0
        A[0, 1:] = 0
        A[1:, 0] = 0
        if np.count_nonzero(A - np.diag(np.diag(A))) == 0:
            st.success("✅ Arrowhead matrix (nonzero first row/col + diagonal)")

    # --- Display Eigenvalues (for user insight) ---
    if square:
        try:
            vals, vecs = np.linalg.eig(M)
            st.write("**Eigenvalues:**", np.round(vals, 4))
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
