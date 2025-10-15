import streamlit as st
import numpy as np
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


# --- Matrix property checks ---
def check_properties(M, name="Matrix"):
    rows, cols = M.shape
    st.subheader(f"🔎 Results for {name}")

    if rows != cols:
        st.info("Matrix is not square, so some checks are skipped.")
    else:
        # --- Basic Checks ---
        if np.allclose(M, M.T, atol=1e-8):
            st.success("✅ Symmetric")

        if np.allclose(M, np.triu(M), atol=1e-8):
            st.success("✅ Upper Triangular")

        if np.allclose(M, np.tril(M), atol=1e-8):
            st.success("✅ Lower Triangular")

        if np.allclose(M, np.diag(np.diag(M)), atol=1e-8):
            st.success("✅ Diagonal matrix")

        # --- Orthogonal ---
        I = np.eye(rows)
        if np.allclose(M.T @ M, I, atol=1e-8):
            st.success("✅ Orthogonal")

        # --- Idempotent ---
        if np.allclose(M @ M, M, atol=1e-8):
            st.success("✅ Idempotent")

        # --- Hat matrix ---
        symmetric = np.allclose(M, M.T, atol=1e-8)
        idempotent = np.allclose(M @ M, M, atol=1e-8)
        if symmetric and idempotent:
            st.success("✅ Hat matrix")

        # --- Hermitian ---
        if np.allclose(M, np.conjugate(M.T), atol=1e-8):
            st.success("✅ Hermitian")

        # --- Positive definite / semidefinite ---
        try:
            eigvals = np.linalg.eigvals(M)
            if np.all(eigvals > 0):
                st.success("✅ Positive definite")
            elif np.all(eigvals >= 0):
                st.success("✅ Positive semidefinite")
        except np.linalg.LinAlgError:
            pass

        # --- Persymmetric ---
        if np.allclose(M, np.fliplr(np.flipud(M)), atol=1e-8):
            st.success("✅ Persymmetric")

        # --- Bisymmetric ---
        if symmetric and np.allclose(M, np.fliplr(np.flipud(M)), atol=1e-8):
            st.success("✅ Bisymmetric")

        # --- Hadamard ---
        if np.all(np.isin(M, [-1, 1])):
            if np.allclose(M @ M.T, rows * np.eye(rows), atol=1e-8):
                st.success("✅ Hadamard matrix")


        # SPACE THE CHECK FOR BLOCK MATRIX


        
        # --- Hollow ---
        if np.allclose(np.diag(M), np.zeros(rows), atol=1e-8):
            st.success("✅ Hollow matrix")

        # --- Generalized permutation ---
        row_counts = np.sum(M != 0, axis=1)
        col_counts = np.sum(M != 0, axis=0)
        if np.all((row_counts == 1) | (row_counts == 0)) and np.all((col_counts == 1) | (col_counts == 0)):
            st.success("✅ Generalized permutation matrix")

        # --- Hankel ---
        if np.allclose(M, np.fliplr(np.triu(np.flipud(M))), atol=1e-8):
            st.success("✅ Hankel-like structure detected")

        # --- Hilbert ---
        hilbert = np.fromfunction(lambda i, j: 1.0 / (i + j + 1), (rows, cols))
        if np.allclose(M, hilbert, atol=1e-8):
            st.success("✅ Hilbert matrix")

        # --- Lehmer ---
        lehmer = np.fromfunction(lambda i, j: np.minimum(i + 1, j + 1) / np.maximum(i + 1, j + 1), (rows, cols))
        if np.allclose(M, lehmer, atol=1e-8):
            st.success("✅ Lehmer matrix")

        # --- Markov ---
        if np.all(M >= 0) and np.allclose(np.sum(M, axis=1), 1, atol=1e-8):
            st.success("✅ Markov matrix")

        # --- Metzler ---
        if np.all(M - np.diag(np.diag(M)) >= 0):
            st.success("✅ Metzler matrix")

        
        # --- Arrowhead (custom definition, must be square) ---
        if rows == cols:
            first_row_ones = np.allclose(M[0, :], np.ones(cols), atol=1e-8)
            first_col_ones = np.allclose(M[:, 0], np.ones(rows), atol=1e-8)
            diag_ones = np.allclose(np.diag(M), np.ones(rows), atol=1e-8)

            # Everything else (non-first row/col and off-diagonal) should be 0
            mask = np.ones_like(M, dtype=bool)
            mask[0, :] = False
            mask[:, 0] = False
            np.fill_diagonal(mask, False)
            others_zero = np.allclose(M[mask], np.zeros(np.count_nonzero(mask)), atol=1e-8)

            if first_row_ones and first_col_ones and diag_ones and others_zero:
                st.success("✅ Arrowhead matrix")






        #THIS DUDE DEFINITLY DOES NOT WORK
        # --- Band / Bidiagonal ---
        nonzero = np.nonzero(M - np.diag(np.diag(M)))
        if len(nonzero[0]) > 0:
            offsets = np.abs(nonzero[0] - nonzero[1])
            bandwidth = np.max(offsets)
            if bandwidth == 1:
                st.success("✅ Bidiagonal matrix")
            elif bandwidth <= 2:
                st.success(f"✅ Band matrix (bandwidth {bandwidth})")

        # --- Sparse ---
        sparsity = 1.0 - (np.count_nonzero(M) / M.size)
        if sparsity > 0.5:
            st.success(f"✅ Sparse matrix ({sparsity*100:.1f}% zeros)")

    # --- Eigen info ---
    if rows == cols:
        try:
            vals, vecs = np.linalg.eig(M)
            st.write("**Eigenvalues:**", vals)
            st.write("**Eigenvectors:**")
            st.write(vecs)
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
            if np.allclose(A.T @ A, I, atol=1e-8):
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
                st.warning("❌ Matrix A is NOT a hat matrix.")

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
    A = get_matrix("A")
    check_properties(A, "Matrix A")
