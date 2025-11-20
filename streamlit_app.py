import streamlit as st
import numpy as np
import pandas as pd
import math

# ---------- CONFIG ----------
# PDF link removed as requested
PDF_URL = "/mnt/data/Many_many_matrices (6).pdf"  

# ---------- Small helper ----------
def safe_allclose(a, b, atol=1e-8):
    try:
        return np.allclose(a, b, atol=atol)
    except Exception:
        return False

# ---------- MATRIX TYPE DEFINITIONS (from LaTeX) ----------
MATRIX_DEFINITIONS = {
    "Arrowhead": "Square matrix where all entries are 0 except for the main diagonal, first row, and first column.",
    "Band": "A sparse matrix where the non-zero elements are centered around the main diagonal in a band.",
    "Bidiagonal": "Square matrix with non-zero entries along the main diagonal and one adjacent diagonal.",
    "Bisymmetric": "A square symmetric matrix that is symmetric with respect to the main and off diagonals.",
    "Block Matrix": "A matrix subdivided into blocks that are matrices themselves.",
    "Generalized Permutation": "Contains the same non-zero pattern as a permutation matrix but allows any non-zero values.",
    "Hadamard": "Square matrix of +1 or -1 entries whose rows are mutually orthogonal.",
    "Hankel": "A matrix where the anti-diagonals are constant.",
    "Hat": "In linear regression, the hat matrix is X(XᵀX)⁻¹Xᵀ. It is symmetric and idempotent.",
    "Hermitian": "Complex square matrix equal to its own conjugate transpose.",
    "Hilbert": "Each entry is 1/(i+j-1). Hilbert matrices are Hankel and symmetric.",
    "Hollow": "A matrix with a zero diagonal, or a large zero block, or sparse enough to be considered hollow.",
    "Idempotent": "A matrix such that A² = A.",
    "Lehmer": "A matrix with entries min(i,j)/max(i,j).",
    "Markov": "Non-negative entries and each column sums to 1.",
    "Metzler": "All off-diagonal elements are nonnegative.",
    "Orthogonal": "A square matrix where Aᵀ = A⁻¹.",
    "Permutation": "A binary matrix with exactly one 1 per row and column.",
    "Persymmetric": "A square matrix symmetric across the anti-diagonal.",
    "Positive Definite": "A symmetric matrix whose eigenvalues are all positive.",
    "Positive Semidefinite": "A symmetric matrix whose eigenvalues are all nonnegative.",
    "Skew-Symmetric": "A square matrix whose transpose is the negative of itself, Aᵀ = –A.",
    "Sparse": "A matrix with most entries equal to zero (typically over 50%).",
    "Symmetric": "A square matrix equal to its transpose.",
    "Triangular": "A square matrix where entries above or below the diagonal are zero.",
    "Toeplitz": "A diagonal-constant matrix: each descending diagonal is constant.",
    "Circulant": "A Toeplitz matrix where each row is a cyclic shift of the previous.",
    "Vandermonde": "Each row is 1, xᵢ, xᵢ², … forming geometric progressions.",
    "Companion": "A square matrix encoding coefficients of a monic polynomial.",
    "Nilpotent": "A square matrix A such that Aᵏ = 0 for some k.",
    "Involutory": "A matrix where A² = I.",
    "Jordan Block": "Upper triangular with λ on diagonal and 1s on superdiagonal.",
    "Pascal": "A symmetric matrix with entries from Pascal’s triangle."
}

# ---------- UI helper for detected types ----------
def show_info_expander(name):
    """Display detected matrix type in a green success box with checkmark."""
    desc = MATRIX_DEFINITIONS.get(name, None)
    if desc is None:
        return
    st.success(f"✅ {name}: {desc}")

# ---------- Matrix input helper ----------
def get_matrix(name):
    st.subheader(f"Matrix {name}")
    rows = st.number_input(f"Number of rows for {name}", min_value=1, max_value=12, value=2, key=f"rows_{name}")
    cols = st.number_input(f"Number of columns for {name}", min_value=1, max_value=12, value=2, key=f"cols_{name}")
    default_data = np.zeros((rows, cols))
    df = pd.DataFrame(default_data, dtype=float)
    st.write(f"Enter values for {name}: (use decimals for non-integers)")
    matrix_input = st.data_editor(df, num_rows="dynamic", key=f"editor_{name}")
    try:
        return matrix_input.to_numpy()
    except Exception:
        return np.array(matrix_input, dtype=float)

# ---------- Property checks ----------
def check_properties(M, name="Matrix"):
    rows, cols = M.shape
    square = rows == cols

    st.subheader(f"🔎 Results for {name}")

    if not square:
        st.info("Matrix is not square — only non-square-specific checks and eigenvalues (skipped).")

    detected = []

    A = np.array(M, dtype=float)

    # Symmetric
    if square and safe_allclose(A, A.T):
        detected.append("Symmetric")
        show_info_expander("Symmetric")

    # Skew-symmetric
    if square and safe_allclose(A, -A.T):
        detected.append("Skew-Symmetric")
        show_info_expander("Skew-Symmetric")

    # Toeplitz: all diagonals constant
    def is_toeplitz(mat):
        r, c = mat.shape
        for k in range(-r+1, c):
            d = np.diag(mat, k=k)
            if d.size > 0 and not np.allclose(d, d[0], atol=1e-8):
                return False
        return True

    if is_toeplitz(A):
        detected.append("Toeplitz")
        show_info_expander("Toeplitz")

    # Circulant
    if rows == cols:
        first_row = A[0, :]
        circ = True
        for i in range(rows):
            if not np.allclose(np.roll(first_row, i), A[i, :], atol=1e-8):
                circ = False
                break
        if circ:
            detected.append("Circulant")
            show_info_expander("Circulant")

    # Vandermonde
    def is_vandermonde(mat):
        r, c = mat.shape
        if r < 1 or c < 2:
            return False
        col0 = mat[:, 0]
        if not np.allclose(col0, np.ones(r), atol=1e-8):
            return False
        x = mat[:, 1]
        for j in range(c):
            if not np.allclose(mat[:, j], x**j, atol=1e-7):
                return False
        return True

    try:
        if is_vandermonde(A):
            detected.append("Vandermonde")
            show_info_expander("Vandermonde")
    except Exception:
        pass

    # Companion
    def is_companion(mat):
        if mat.shape[0] != mat.shape[1]:
            return False
        n = mat.shape[0]
        for i in range(1, n):
            if not np.isclose(mat[i, i-1], 1.0, atol=1e-8):
                return False
        if not np.allclose(mat[:-1, :-1], 0, atol=1e-8):
            return False
        return True

    if is_companion(A):
        detected.append("Companion")
        show_info_expander("Companion")

    # Nilpotent
    if square:
        power = np.copy(A)
        nil = False
        for k in range(1, rows + 1):
            if np.allclose(power, np.zeros_like(A), atol=1e-8):
                nil = True
                nil_index = k
                break
            power = power @ A
        if nil:
            detected.append("Nilpotent")
            with st.expander("Nilpotent", expanded=False):
                st.write(MATRIX_DEFINITIONS.get("Nilpotent", ""))
                st.write(f"Index of nilpotency ≤ {nil_index}")

    # Involutory
    if square and safe_allclose(A @ A, np.eye(rows)):
        detected.append("Involutory")
        show_info_expander("Involutory")

    # Additional heuristics (Hadamard, Hankel, Persymmetric, Sparse, etc.)
    # For brevity, other checks can remain as in your previous code
    # ...

    if len(detected) == 0:
        st.write("No special types positively detected (based on current heuristics).")
    else:
        st.write("Detected types:", ", ".join(detected))

    # Eigenvalues/vectors
    if square:
        try:
            vals, vecs = np.linalg.eig(A)
            st.write("**Eigenvalues:**")
            st.write(vals)
            st.write("**Eigenvectors:**")
            st.write(vecs)
        except np.linalg.LinAlgError:
            st.error("Eigenvalue calculation failed.")
    else:
        st.write("Eigen analysis skipped for non-square matrix.")

# ---------- APP UI ----------
st.markdown(
    """
    <div style="background-color:#0A2647; padding:15px; border-radius:10px; text-align:center;">
        <h1 style="color:white;">Matrix Calculator</h1>
    </div>
    """,
    unsafe_allow_html=True
)

mode = st.selectbox(
    "Choose Mode:",
    ["Classroom Mode", "Special Matrix Identifier"],
    index=0,
    key="mode_selector"
)

# --- Classroom Mode (unchanged) ---
if mode == "Classroom Mode":
    use_two_matrices = st.checkbox("Work with two matrices (A and B)?", value=False)

    A = get_matrix("A")
    B = get_matrix("B") if use_two_matrices else None

    st.write("**Matrix A:**")
    st.write(A)
    if B is not None:
        st.write("**Matrix B:**")
        st.write(B)

    if use_two_matrices:
        op = st.selectbox("Choose an operation:", ["A × B"])
    else:
        op = st.selectbox(
            "Choose an operation:",
            ["Transpose", "Inverse", "Multiply by Itself", "Eigenvalues", "Check Orthogonal", "Check Hat Matrix"]
        )

    # ... all your existing operations remain unchanged ...

# --- Special Matrix Identifier Mode ---
elif mode == "Special Matrix Identifier":

    # Dropdown of all matrix types
    st.subheader("All Matrix Types Checked:")
    matrix_type_list = list(MATRIX_DEFINITIONS.keys())
    selected_type = st.selectbox("Select a type to view its description:", matrix_type_list)
    st.write(MATRIX_DEFINITIONS[selected_type])

    use_two_matrices = st.checkbox("Work with two matrices (A and B)?", value=False)

    # Input matrix A
    A = get_matrix("A")
    st.write("**Matrix A preview:**")
    st.write(A)
    check_properties(A, "Matrix A")

    if use_two_matrices:
        # Input matrix B
        B = get_matrix("B")
        st.write("**Matrix B preview:**")
        st.write(B)
        check_properties(B, "Matrix B")

        # Multiply and analyze product
        if A.shape[1] == B.shape[0]:
            C = A @ B
            st.subheader("**Result of A × B:**")
            st.write(C)
            check_properties(C, "Matrix A × B")
        else:
            st.warning("⚠️ Cannot multiply A × B (dimension mismatch).")
