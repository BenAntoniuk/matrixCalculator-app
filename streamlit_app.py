import streamlit as st
import numpy as np
import pandas as pd
import math

# ---------- CONFIG ----------
PDF_URL = "xxxxxxxxx"  # external file provided by you

# ---------- Small helper ----------
def safe_allclose(a, b, atol=1e-8):
    try:
        return np.allclose(a, b, atol=atol)
    except Exception:
        return False

# ---------- Descriptions dictionary (MATRIX_INFO) ----------
# Short definitions pulled / paraphrased from your LaTeX doc.
MATRIX_INFO = {
    "Symmetric": {
        "description": "A square matrix equal to its transpose (A = Aᵀ).",
        "link": PDF_URL
    },
    "Skew-Symmetric": {
        "description": "A square matrix whose transpose equals its negative (Aᵀ = −A). Diagonal entries are 0.",
        "link": PDF_URL
    },
    "Toeplitz": {
        "description": "Diagonal-constant matrix: each descending diagonal from left to right is constant.",
        "link": PDF_URL
    },
    "Circulant": {
        "description": "A special Toeplitz matrix where each row is a cyclic shift of the previous row (completely defined by the first row).",
        "link": PDF_URL
    },
    "Vandermonde": {
        "description": "Rows are geometric progressions of elements x_i: row i = [1, x_i, x_i^2, ..., x_i^{n-1}].",
        "link": PDF_URL
    },
    "Companion": {
        "description": "Companion matrix of a monic polynomial: ones on the subdiagonal and the last column (negated coefficients) encode polynomial coefficients.",
        "link": PDF_URL
    },
    "Nilpotent": {
        "description": "A square matrix N such that N^k = 0 for some positive integer k (index of nilpotency).",
        "link": PDF_URL
    },
    "Involutory": {
        "description": "A matrix equal to its own inverse: A^2 = I.",
        "link": PDF_URL
    },
    "Jordan Block": {
        "description": "Upper-triangular block with a single eigenvalue λ on the diagonal and 1s on the superdiagonal.",
        "link": PDF_URL
    },
    "Pascal": {
        "description": "Matrix made from binomial coefficients: P_{ij} = C(i+j-2, i-1) (1-based).",
        "link": PDF_URL
    },
    "Hat matrix": {
        "description": "Projection (hat) matrix H = X(XᵀX)^{-1}Xᵀ — always symmetric and idempotent.",
        "link": PDF_URL
    },
    "Orthogonal": {
        "description": "Square matrix A where A^T = A^{-1}, equivalently A A^T = I (preserves lengths/angles).",
        "link": PDF_URL
    },
    "Hermitian": {
        "description": "Complex matrix equal to its conjugate transpose (A = A*) — real analogue is symmetric.",
        "link": PDF_URL
    },
    "Hilbert": {
        "description": "Matrix with entries 1/(i+j-1) (ill-conditioned classic example).",
        "link": PDF_URL
    },
    "Toeplitz-like": {
        "description": "General diagonal-constant structure (used when it's Toeplitz but not identically captured elsewhere).",
        "link": PDF_URL
    },
    "Hadamard": {
        "description": "Entries ±1 with mutually orthogonal rows; H H^T = n I.",
        "link": PDF_URL
    },
    # add more short entries as needed...
}

# ---------- UI helpers ----------
def show_info_expander(name):
    info = MATRIX_INFO.get(name, None)
    if info is None:
        return
    with st.expander(name, expanded=False):
        st.write(info["description"])
        st.markdown(f"[View Full Documentation]({info['link']})")

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
    """
    Detects many matrix types; for each positive detection, shows an expander
    with short description and link to full PDF. Also prints eigenvalues/vectors.
    """
    rows, cols = M.shape
    square = rows == cols

    st.subheader(f"🔎 Results for {name}")

    if not square:
        st.info("Matrix is not square — only non-square-specific checks and eigenvalues (skipped).")

    detected = []

    # Safe numeric copy
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

    # Circulant: each row is roll of first row
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

    # Vandermonde: columns are powers of some vector x, first col all ones
    def is_vandermonde(mat):
        r, c = mat.shape
        if r < 1 or c < 2:
            return False
        col0 = mat[:, 0]
        if not np.allclose(col0, np.ones(r), atol=1e-8):
            return False
        x = mat[:, 1]  # candidate base vector
        # check subsequent columns j: mat[:, j] == x**j
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

    # Companion: ones on subdiagonal and upper-left (n-1)x(n-1) block zeros (heuristic)
    def is_companion(mat):
        if mat.shape[0] != mat.shape[1]:
            return False
        n = mat.shape[0]
        # ones on subdiagonal?
        for i in range(1, n):
            if not np.isclose(mat[i, i-1], 1.0, atol=1e-8):
                return False
        # upper-left (n-1)x(n-1) should be zero matrix (heuristic)
        if not np.allclose(mat[:-1, :-1], 0, atol=1e-8):
            return False
        return True

    if is_companion(A):
        detected.append("Companion")
        show_info_expander("Companion")

    # Nilpotent: exists k <= n such that A^k = 0
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
                st.write(MATRIX_INFO.get("Nilpotent", {})["description"])
                st.write(f"Index of nilpotency ≤ {nil_index}")
                st.markdown(f"[View Full Documentation]({PDF_URL})")

    # Involutory: A^2 = I
    if square and safe_allclose(A @ A, np.eye(rows)):
        detected.append("Involutory")
        show_info_expander("Involutory")

    # Jordan block detection: constant diagonal and ones on superdiagonal and zero elsewhere
    if square:
        diag_vals = np.diag(A)
        if np.allclose(np.diag(A, k=1), np.ones(rows - 1), atol=1e-8):
            # check all other off-super entries zero
            J_minus = A - np.diag(diag_vals) - np.diag(np.ones(rows - 1), 1)
            if np.allclose(J_minus, 0, atol=1e-8):
                detected.append("Jordan Block")
                show_info_expander("Jordan Block")

    # Pascal matrix detection (small sizes): P_{ij} = comb(i+j, i) for 0-based
    if square:
        pascal_like = True
        for i in range(rows):
            for j in range(cols):
                try:
                    expected = math.comb(i + j, i)  # zero-based equivalent
                except Exception:
                    pascal_like = False
                    break
                if not np.allclose(A[i, j], expected, atol=1e-8):
                    pascal_like = False
                    break
            if not pascal_like:
                break
        if pascal_like:
            detected.append("Pascal")
            show_info_expander("Pascal")

    # Hadamard detection
    if square and np.all(np.isin(A, [-1, 1])):
        if np.allclose(A @ A.T, rows * np.eye(rows), atol=1e-8):
            detected.append("Hadamard")
            show_info_expander("Hadamard")

    # Hilbert detection
    hilbert = np.fromfunction(lambda i, j: 1.0 / (i + j + 1), (rows, cols))
    if safe_allclose(A, hilbert):
        detected.append("Hilbert")
        show_info_expander("Hilbert")

    # Toeplitz-like / Hankel / Persymmetric tests (heuristic)
    # Hankel: constant anti-diagonals: check if A[i,j] depends only on i+j
    def is_hankel(mat):
        r, c = mat.shape
        for s in range(r + c - 1):
            vals = []
            for i in range(r):
                j = s - i
                if 0 <= j < c:
                    vals.append(mat[i, j])
            if len(vals) > 1:
                if not np.allclose(vals, vals[0], atol=1e-8):
                    return False
        return True

    if is_hankel(A):
        detected.append("Hankel")
        show_info_expander("Hankel")

    # Persymmetric (symmetric wrt anti-diagonal)
    if square and np.allclose(A, np.fliplr(np.flipud(A)), atol=1e-8):
        detected.append("Persymmetric")
        show_info_expander("Persymmetric")

    # Sparse: >50% zeros heuristic
    sparsity = 1.0 - (np.count_nonzero(A) / A.size)
    if sparsity >= 0.5:
        detected.append("Sparse")
        with st.expander("Sparse", expanded=False):
            st.write("Matrix has high sparsity (≥ 50% zeros).")
            st.write(f"Sparsity: {sparsity*100:.1f}%")
            st.markdown(f"[View Full Documentation]({PDF_URL})")

    # Generalized permutation / permutation detection
    row_counts = np.sum(A != 0, axis=1)
    col_counts = np.sum(A != 0, axis=0)
    if np.all((row_counts == 1) | (row_counts == 0)) and np.all((col_counts == 1) | (col_counts == 0)):
        detected.append("Generalized permutation")
        with st.expander("Generalized permutation matrix", expanded=False):
            st.write("Matrix has at most one nonzero per row/column — a generalized permutation pattern.")
            st.markdown(f"[View Full Documentation]({PDF_URL})")

    # Orthogonal
    if square and safe_allclose(A.T @ A, np.eye(rows)):
        detected.append("Orthogonal")
        show_info_expander("Orthogonal")

    # Hermitian
    if square and safe_allclose(A, np.conjugate(A.T)):
        detected.append("Hermitian")
        show_info_expander("Hermitian")

    # Idempotent, Hat matrix
    if square and safe_allclose(A @ A, A):
        detected.append("Idempotent")
        with st.expander("Idempotent", expanded=False):
            st.write("A matrix that satisfies A^2 = A.")
            st.markdown(f"[View Full Documentation]({PDF_URL})")
        # hat matrix is idempotent + symmetric
        if safe_allclose(A, A.T):
            detected.append("Hat matrix")
            show_info_expander("Hat matrix")

    # Positive definite / semidefinite
    if square:
        try:
            eigvals = np.linalg.eigvals(A)
            if np.all(eigvals > -1e-10):  # small tolerance for numerical noise
                if np.all(eigvals > 0):
                    detected.append("Positive definite")
                    with st.expander("Positive definite", expanded=False):
                        st.write("All eigenvalues are positive.")
                        st.markdown(f"[View Full Documentation]({PDF_URL})")
                else:
                    detected.append("Positive semidefinite")
                    with st.expander("Positive semidefinite", expanded=False):
                        st.write("All eigenvalues are nonnegative.")
                        st.markdown(f"[View Full Documentation]({PDF_URL})")
        except Exception:
            pass

    # Report summary
    if len(detected) == 0:
        st.write("No special types positively detected (based on current heuristics).")
    else:
        st.write("Detected types:", ", ".join(detected))

    # Always compute eigenvalues/eigenvectors for square matrices
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

# Classroom Mode (keeps your existing operations)
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

    if op == "Transpose":
        st.write("**Transpose:**")
        st.write(A.T)

    elif op == "Inverse":
        try:
            st.write("**Inverse:**")
            st.write(np.linalg.inv(A))
        except np.linalg.LinAlgError:
            st.error("Matrix is singular and cannot be inverted.")

    elif op == "Multiply by Itself":
        try:
            st.write("**A × A:**")
            st.write(np.dot(A, A))
        except Exception as e:
            st.error(f"Error: {e}")

    elif op == "Eigenvalues":
        try:
            vals, vecs = np.linalg.eig(A)
            st.write("**Eigenvalues:**")
            st.write(vals)
            st.write("**Eigenvectors:**")
            st.write(vecs)
        except np.linalg.LinAlgError:
            st.error("Eigenvalue calculation failed.")

    elif op == "Check Orthogonal":
        if A.shape[0] != A.shape[1]:
            st.error("Matrix must be square to check orthogonality.")
        else:
            if np.allclose(A.T @ A, np.eye(A.shape[0]), atol=1e-8):
                st.success("✅ Matrix A is orthogonal.")
            else:
                st.warning("❌ Matrix A is NOT orthogonal.")

    elif op == "Check Hat Matrix":
        if A.shape[0] != A.shape[1]:
            st.error("Matrix must be square to check if it's a hat matrix.")
        else:
            symmetric = np.allclose(A, A.T, atol=1e-8)
            idempotent = np.allclose(A @ A, A, atol=1e-8)
            if symmetric and idempotent:
                st.success("✅ Matrix A is a hat matrix.")
            else:
                st.warning("❌ Matrix A is NOT a hat matrix.")

    elif op == "A × B":
        try:
            if A.shape[1] != B.shape[0]:
                st.error("Number of columns in A must equal number of rows in B.")
            else:
                st.write("**A × B:**")
                C = np.dot(A, B)
                st.write(C)
        except Exception as e:
            st.error(f"Error: {e}")

# Special Matrix Identifier Mode
elif mode == "Special Matrix Identifier":
    use_two_matrices = st.checkbox("Work with two matrices (A and B)?", value=False)

    A = get_matrix("A")
    st.write("**Matrix A preview:**")
    st.write(A)
    check_properties(A, "Matrix A")

    if use_two_matrices:
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
