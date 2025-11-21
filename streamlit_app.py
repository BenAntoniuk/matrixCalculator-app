import streamlit as st
import numpy as np
import pandas as pd
import math

# ---------- CONFIG ----------
PDF_URL = "/mnt/data/Many_many_matrices (6).pdf"

def safe_allclose(a, b, atol=1e-8):
    try:
        return np.allclose(a, b, atol=atol)
    except Exception:
        return False

# ---------- MATRIX TYPE DEFINITIONS ----------
MATRIX_DEFINITIONS = {
    "Arrowhead": "Square matrix where all entries are 0 except for the main diagonal, first row, and first column.",
    "Band": "A sparse matrix where the non-zero elements are centered around the main diagonal in a band.",
    "Bidiagonal": "Square matrix with non-zero entries along the main diagonal and one adjacent diagonal.",
    "Bisymmetric": "A square symmetric matrix that is symmetric with respect to the main and off diagonals.",
    "Block Matrix": "A matrix subdivided into blocks that are matrices themselves.",
    "Companion": "A square matrix encoding coefficients of a monic polynomial.",
    "Circulant": "A Toeplitz matrix where each row is a cyclic shift of the previous.",
    "Generalized Permutation": "Contains the same non-zero pattern as a permutation matrix but allows any non-zero values.",
    "Hadamard": "Square matrix of +1 or -1 entries whose rows are mutually orthogonal.",
    "Hankel": "A matrix where the anti-diagonals are constant.",
    "Hat": "In linear regression, the hat matrix is X(XᵀX)⁻¹Xᵀ. It is symmetric and idempotent.",
    "Hermitian": "Complex square matrix equal to its own conjugate transpose.",
    "Hilbert": "Each entry is 1/(i+j-1). Hilbert matrices are Hankel and symmetric.",
    "Hollow": "A matrix with a zero diagonal, or a large zero block, or sparse enough to be considered hollow.",
    "Idempotent": "A matrix such that A² = A.",
    "Involutory": "A matrix where A² = I.",
    "Jordan Block": "Upper triangular with λ on diagonal and 1s on superdiagonal.",
    "Lehmer": "A matrix with entries min(i,j)/max(i,j).",
    "Markov": "Non-negative entries and each column sums to 1.",
    "Metzler": "All off-diagonal elements are nonnegative.",
    "Nilpotent": "A square matrix A such that Aᵏ = 0 for some k.",
    "Orthogonal": "A square matrix where Aᵀ = A⁻¹.",
    "Pascal": "A symmetric matrix with entries from Pascal’s triangle.",
    "Permutation": "A binary matrix with exactly one 1 per row and column.",
    "Persymmetric": "A square matrix symmetric across the anti-diagonal.",
    "Positive Definite": "A symmetric matrix whose eigenvalues are all positive.",
    "Positive Semidefinite": "A symmetric matrix whose eigenvalues are all nonnegative.",
    "Skew-Symmetric": "A square matrix whose transpose is the negative of itself, Aᵀ = –A.",
    "Sparse": "A matrix with most entries equal to zero.",
    "Symmetric": "A square matrix equal to its transpose.",
    "Toeplitz": "A diagonal-constant matrix: each descending diagonal is constant.",
    "Triangular": "A square matrix where entries above or below the diagonal are zero.",
    "Vandermonde": "Each row is 1, xᵢ, xᵢ², … forming geometric progressions."
}

# ---------- UI helper ----------
def show_info_expander(name, extra_info=None):
    desc = MATRIX_DEFINITIONS.get(name, "")
    full_desc = desc if extra_info is None else f"{desc} ({extra_info})"
    st.success(f"✅ **{name}**: {full_desc}")

# ---------- Matrix input ----------
def get_matrix(name):
    st.subheader(f"Matrix {name}")
    rows = st.number_input(f"Number of rows for {name}", 1, 12, 2, key=f"rows_{name}")
    cols = st.number_input(f"Number of columns for {name}", 1, 12, 2, key=f"cols_{name}")
    df = pd.DataFrame(np.zeros((rows, cols)), dtype=float)
    st.write(f"Enter values for {name}:")
    mat = st.data_editor(df, num_rows="dynamic", key=f"editor_{name}")
    try:
        return mat.to_numpy()
    except:
        return np.array(mat)

# ---------- Property checker ----------
def check_properties(A, name="Matrix"):
    rows, cols = A.shape
    sq = rows == cols

    st.subheader(f"🔎 Results for {name}")

    # Symmetric
    if sq and safe_allclose(A, A.T):
        show_info_expander("Symmetric")

    # Skew-symmetric
    if sq and safe_allclose(A, -A.T):
        show_info_expander("Skew-Symmetric")

    # Toeplitz
    def is_toeplitz(M):
        r, c = M.shape
        for k in range(-r+1, c):
            diag = np.diag(M, k)
            if diag.size and not np.allclose(diag, diag[0]):
                return False
        return True

    if is_toeplitz(A):
        show_info_expander("Toeplitz")

    # Circulant
    if sq:
        first = A[0]
        if all(np.allclose(np.roll(first, i), A[i]) for i in range(rows)):
            show_info_expander("Circulant")

    # Vandermonde
    def is_vand(M):
        r, c = M.shape
        if c < 2:
            return False
        if not np.allclose(M[:, 0], 1):
            return False
        x = M[:, 1]
        for j in range(c):
            if not np.allclose(M[:, j], x**j):
                return False
        return True

    try:
        if is_vand(A):
            show_info_expander("Vandermonde")
    except:
        pass

    # Companion
    def is_comp(M):
        if M.shape[0] != M.shape[1]:
            return False
        n = M.shape[0]
        if not np.allclose(M[1:, :-1], np.eye(n-1), atol=1e-8):
            return False
        if not np.allclose(M[:-1, :-1], 0):
            return False
        return True

    if is_comp(A):
        show_info_expander("Companion")

    # Nilpotent
    if sq:
        P = A.copy()
        for k in range(1, rows + 1):
            if np.allclose(P, 0):
                show_info_expander("Nilpotent", f"Index ≤ {k}")
                break
            P = P @ A

    # Involutory
    if sq and safe_allclose(A @ A, np.eye(rows)):
        show_info_expander("Involutory")

    # Orthogonal
    if sq and safe_allclose(A.T @ A, np.eye(rows)):
        show_info_expander("Orthogonal")

    # Hermitian
    if sq and safe_allclose(A, np.conjugate(A.T)):
        show_info_expander("Hermitian")

    # Idempotent (and Hat)
    if sq and safe_allclose(A @ A, A):
        show_info_expander("Idempotent")
        if safe_allclose(A, A.T):
            show_info_expander("Hat")

    # Eigenvalues
    if sq:
        try:
            vals, vecs = np.linalg.eig(A)
            st.write("**Eigenvalues:**")
            st.write(vals)
            st.write("**Eigenvectors:**")
            st.write(vecs)
        except:
            st.error("Eigenvalue computation failed.")

# ---------- APP UI ----------
st.markdown(
    """
    <div style="background-color:#0A2647; padding:15px; border-radius:10px; text-align:center;">
        <h1 style="color:white;">Matrix Calculator</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# STRICT dropdown (typing not allowed)
mode = st.selectbox("Choose Mode:", ["Classroom Mode", "Special Matrix Identifier"])

# ---------------------- CLASSROOM MODE ----------------------
if mode == "Classroom Mode":

    use_two = st.checkbox("Work with two matrices (A and B)?", value=False)

    A = get_matrix("A")
    B = get_matrix("B") if use_two else None

    if use_two:
        operation = st.selectbox("Choose an operation:", ["A × B"])
    else:
        operation = st.selectbox(
            "Choose an operation:",
            ["Transpose", "Inverse", "Multiply by Itself", "Eigenvalues", 
             "Check Orthogonal", "Check Hat Matrix", "Hat Matrix Calculator"]   # NEW OPTION
        )

    # ------------ OPERATIONS ------------
    if operation == "Transpose":
        st.write("**Aᵀ:**")
        st.write(A.T)

    elif operation == "Inverse":
        try:
            st.write("**A⁻¹:**")
            st.write(np.linalg.inv(A))
        except:
            st.error("Matrix is singular.")

    elif operation == "Multiply by Itself":
        st.write("**A × A:**")
        st.write(A @ A)

    elif operation == "Eigenvalues":
        vals, vecs = np.linalg.eig(A)
        st.write("**Eigenvalues:**")
        st.write(vals)
        st.write("**Eigenvectors:**")
        st.write(vecs)

    elif operation == "Check Orthogonal":
        if A.shape[0] != A.shape[1]:
            st.error("Matrix must be square.")
        else:
            if np.allclose(A.T @ A, np.eye(A.shape[0])):
                st.success("✅ A is orthogonal.")
            else:
                st.warning("❌ A is NOT orthogonal.")

    elif operation == "Check Hat Matrix":
        if A.shape[0] != A.shape[1]:
            st.error("Matrix must be square.")
        else:
            if np.allclose(A @ A, A) and np.allclose(A, A.T):
                st.success("✅ A is a hat matrix.")
            else:
                st.warning("❌ A is NOT a hat matrix.")

    # ------------ NEW: HAT MATRIX CALCULATOR ------------
    elif operation == "Hat Matrix Calculator":
        try:
            XtX = A.T @ A
            XtX_inv = np.linalg.inv(XtX)
            H = A @ XtX_inv @ A.T
            st.subheader("🎩 Hat Matrix (H = X(XᵀX)⁻¹Xᵀ)")
            st.write(H)

            # Leverage values
            leverages = np.diag(H)
            st.subheader("🔍 Leverage Values")
            st.write(leverages)

        except np.linalg.LinAlgError:
            st.error("XᵀX is not invertible — hat matrix cannot be computed.")

    elif operation == "A × B":
        if A.shape[1] != B.shape[0]:
            st.error("Column count of A must equal row count of B.")
        else:
            st.write("**A × B:**")
            st.write(A @ B)

# ---------------------- SPECIAL MATRIX IDENTIFIER ----------------------
else:
    st.sidebar.subheader("All Matrix Types")
    types_sorted = sorted(MATRIX_DEFINITIONS.keys())
    sel = st.sidebar.selectbox("Select a type:", types_sorted)
    st.sidebar.markdown(f"**{sel}**  \n{MATRIX_DEFINITIONS[sel]}")

    use_two = st.checkbox("Work with two matrices (A and B)?", value=False)

    A = get_matrix("A")
    st.write("**Matrix A Preview:**")
    st.write(A)
    check_properties(A, "Matrix A")

    if use_two:
        B = get_matrix("B")
        st.write("**Matrix B Preview:**")
        st.write(B)
        check_properties(B, "Matrix B")

        if A.shape[1] == B.shape[0]:
            C = A @ B
            st.subheader("A × B:")
            st.write(C)
            check_properties(C, "Matrix A × B")
        else:
            st.warning("⚠️ Cannot multiply A × B (dimension mismatch).")
