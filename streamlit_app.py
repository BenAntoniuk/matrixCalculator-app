# app.py — Full copy-paste ready
import streamlit as st
import numpy as np
import pandas as pd
import math
from fractions import Fraction

st.set_page_config(page_title="Matrix Toolkit", layout="wide")

# ---------- CONFIG ----------
PDF_URL = "/mnt/data/Many_many_matrices (6).pdf"  # kept for reference

# ---------- Small helper ----------
def safe_allclose(a, b, atol=1e-8):
    try:
        return np.allclose(a, b, atol=atol)
    except Exception:
        return False

# ---------- MATRIX TYPE DEFINITIONS (alphabetized) ----------
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
    "Jordan Block": "Upper-triangular block with a single eigenvalue λ on the diagonal and 1s on the superdiagonal.",
    "Lehmer": "A matrix with entries min(i,j)/max(i,j).",
    "Markov": "Non-negative entries and each column sums to 1.",
    "Metzler": "All off-diagonal elements are nonnegative.",
    "Nilpotent": "A square matrix N such that N^k = 0 for some positive integer k (index of nilpotency).",
    "Orthogonal": "A square matrix where Aᵀ = A⁻¹.",
    "Pascal": "Matrix made from binomial coefficients: P_{ij} = C(i+j-2, i-1) (1-based).",
    "Permutation": "A binary matrix with exactly one 1 per row and column.",
    "Persymmetric": "A square matrix symmetric across the anti-diagonal.",
    "Positive Definite": "A symmetric matrix whose eigenvalues are all positive.",
    "Positive Semidefinite": "A symmetric matrix whose eigenvalues are all nonnegative.",
    "Skew-Symmetric": "A square matrix whose transpose is the negative of itself, Aᵀ = –A.",
    "Sparse": "A matrix with most entries equal to zero (typically ≥ 50%).",
    "Symmetric": "A square matrix equal to its transpose.",
    "Toeplitz": "Diagonal-constant matrix: each descending diagonal from left to right is constant.",
    "Triangular": "A square matrix where entries above or below the diagonal are zero.",
    "Vandermonde": "Rows are geometric progressions of elements x_i: row i = [1, x_i, x_i^2, ...]."
}

# Alphabetize list for sidebar usage
ALL_TYPES_SORTED = sorted(MATRIX_DEFINITIONS.keys(), key=lambda s: s.lower())

# ---------- UI helper for detected types ----------
def show_info_expander(name, extra_info=None):
    """Display detected matrix type in green checkmark style with bold name and newline."""
    desc = MATRIX_DEFINITIONS.get(name, "")
    full_desc = desc if extra_info is None else f"{desc} ({extra_info})"
    st.success(f"✅ **{name}**  \n{full_desc}")

# ---------- Fraction parser ----------
def parse_fraction_safe(x):
    """Convert integers, floats, or fraction strings like '3/4' safely to float."""
    try:
        # handle pandas / numpy nan directly
        if x is None:
            return float("nan")
        if isinstance(x, float) and np.isnan(x):
            return float("nan")
        if isinstance(x, str):
            x = x.strip()
            if x == "":
                return float("nan")
            if "/" in x:
                return float(Fraction(x))
            return float(x)
        return float(x)
    except Exception:
        return float("nan")

# ---------- Matrix input helper (data_editor) ----------
def get_matrix(name, default_rows=2, default_cols=2, key_prefix=None):
    """
    Display a data_editor for matrix entry and parse entries (supports fractions).
    Returns numpy array of floats (NaN for unparsable entries).
    """
    if key_prefix is None:
        key_prefix = name.replace(" ", "_")

    st.subheader(f"Matrix {name}")
    rows = st.number_input(f"Number of rows for {name}", min_value=1, max_value=50, value=default_rows, key=f"rows_{key_prefix}")
    cols = st.number_input(f"Number of columns for {name}", min_value=1, max_value=50, value=default_cols, key=f"cols_{key_prefix}")

    # default as strings so user can type fractions
    default_data = pd.DataFrame([["0" for _ in range(cols)] for _ in range(rows)])
    st.write(f"Enter values for {name}: (integers, decimals, or fractions like 1/3)")
    matrix_input = st.data_editor(default_data, num_rows="dynamic", key=f"editor_{key_prefix}")

    # Convert to numeric numpy array using fraction parser
    parsed = np.zeros((len(matrix_input), len(matrix_input.columns)), dtype=float)
    for i in range(len(matrix_input)):
        for j in range(len(matrix_input.columns)):
            parsed[i, j] = parse_fraction_safe(matrix_input.iloc[i, j])

    return parsed

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

    # Toeplitz
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

    # Companion (heuristic)
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
        nil_index = None
        for k in range(1, rows + 1):
            if np.allclose(power, np.zeros_like(A), atol=1e-8):
                nil = True
                nil_index = k
                break
            power = power @ A
        if nil:
            detected.append("Nilpotent")
            show_info_expander("Nilpotent", extra_info=f"Index of nilpotency ≤ {nil_index}")

    # Involutory
    if square and safe_allclose(A @ A, np.eye(rows)):
        detected.append("Involutory")
        show_info_expander("Involutory")

    # Orthogonal
    if square and safe_allclose(A.T @ A, np.eye(rows)):
        detected.append("Orthogonal")
        show_info_expander("Orthogonal")

    # Hermitian
    if square and safe_allclose(A, np.conjugate(A.T)):
        detected.append("Hermitian")
        show_info_expander("Hermitian")

    # Idempotent / Hat
    if square and safe_allclose(A @ A, A):
        detected.append("Idempotent")
        show_info_expander("Idempotent")
        if safe_allclose(A, A.T):
            detected.append("Hat")
            show_info_expander("Hat")

    # Sparse heuristic
    sparsity = 1.0 - (np.count_nonzero(A) / A.size)
    if sparsity >= 0.5:
        detected.append("Sparse")
        show_info_expander("Sparse", extra_info=f"Sparsity: {sparsity*100:.1f}%")

    # Pascal detection (small sizes)
    if square:
        pascal_like = True
        for i in range(rows):
            for j in range(cols):
                try:
                    expected = math.comb(i + j, i)
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

    # Hankel detection
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

    # Persymmetric
    if square and np.allclose(A, np.fliplr(np.flipud(A)), atol=1e-8):
        detected.append("Persymmetric")
        show_info_expander("Persymmetric")

    # Positive definite / semidefinite
    if square:
        try:
            eigvals = np.linalg.eigvals(A)
            if np.all(eigvals > -1e-10):
                if np.all(eigvals > 0):
                    detected.append("Positive Definite")
                    show_info_expander("Positive Definite")
                else:
                    detected.append("Positive Semidefinite")
                    show_info_expander("Positive Semidefinite")
        except Exception:
            pass

    # Summary
    if len(detected) == 0:
        st.write("No special types positively detected (based on current heuristics).")
    else:
        st.write("Detected types:", ", ".join(detected))

    # Eigen analysis for square matrices
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

# Strict dropdown for mode selection (typing disabled)
mode = st.selectbox(
    "Choose Mode:",
    ["Classroom Mode", "Hat Matrix Calculator", "Special Matrix Identifier"],
    index=0,
    key="mode_selector"
)

# ---------- Classroom Mode ----------
if mode == "Classroom Mode":
    use_two_matrices = st.checkbox("Work with two matrices (A and B)?", value=False)

    A = get_matrix("A", default_rows=2, default_cols=2, key_prefix="classA")
    B = get_matrix("B", default_rows=2, default_cols=2, key_prefix="classB") if use_two_matrices else None

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
            ["Transpose", "Inverse", "Multiply by Itself", "Eigenvalues", "Check Orthogonal"]
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

    elif op == "A × B":
        try:
            if A.shape[1] != B.shape[0]:
                st.error("Number of columns in A must equal number of rows in B.")
            else:
                C = A @ B
                st.write("**A × B:**")
                st.write(C)
        except Exception as e:
            st.error(f"Error: {e}")



# ---------- Hat Matrix Mode ----------
elif mode == "Hat Matrix Calculator":
    st.subheader("Hat Matrix Calculator 🎩")
    input_method = st.radio("Choose input method:", ["Manual Entry", "Import Google Sheet CSV"])
    
    X = None
    if input_method == "Manual Entry":
        X = get_matrix("X")
    
    elif input_method == "Import Google Sheet CSV":
        csv_url = st.text_input("Enter the published Google Sheet CSV URL:")
        if csv_url:
            try:
                df = pd.read_csv(csv_url)
                # Remove columns named 'Timestamp' or 'score' (case-insensitive)
                df = df.loc[:, ~df.columns.str.lower().isin(['timestamp', 'score'])]
                # Keep only numeric columns and drop completely empty rows
                df_numeric = df.select_dtypes(include=[np.number]).dropna(how="all")
                
                if df_numeric.empty:
                    st.error("No numeric data found in the sheet after filtering.")
                else:
                    # Add intercept column of 1's
                    df_numeric.insert(0, "Intercept", 1)
                    X = df_numeric.to_numpy()
                    st.write("**Imported Matrix X:**")
                    st.write(X)
            except Exception as e:
                st.error(f"Error loading CSV: {e}")

    if X is not None:
        try:
            XtX = X.T @ X
            if np.linalg.matrix_rank(XtX) < XtX.shape[0]:
                st.error("❌ Cannot compute hat matrix: (XᵀX) is not invertible.")
            else:
                XtX_inv = np.linalg.inv(XtX)
                H = X @ XtX_inv @ X.T
                st.subheader("🎩 Hat Matrix (H = X (XᵀX)⁻¹ Xᵀ)")
                st.write(H)
                
                leverages = np.diag(H)
                st.subheader("🔍 Leverage Values (diag(H))")
                st.write(leverages)

                # --- Residuals ---
                st.subheader("📝 Residuals")
                # Let user input response vector y
                y_input = st.text_area(
                    "Enter response vector y (comma-separated or space-separated):",
                    value=""
                )
                if y_input.strip():
                    try:
                        y_values = [parse_fraction_safe(v) for v in y_input.replace(",", " ").split()]
                        y = np.array(y_values, dtype=float).reshape(-1, 1)
                        if y.shape[0] != H.shape[0]:
                            st.error("Length of y does not match number of rows in X.")
                        else:
                            y_hat = H @ y
                            residuals = y - y_hat
                            st.write("**Predicted y (ŷ):**")
                            st.write(y_hat)
                            st.write("**Residuals (y - ŷ):**")
                            st.write(residuals)
                    except Exception as e:
                        st.error(f"Error processing y vector: {e}")
        except Exception as e:
            st.error(f"Error computing hat matrix: {e}")


# ---------- Special Matrix Identifier Mode ----------
elif mode == "Special Matrix Identifier":
    # Sidebar dropdown for all matrix types (alphabetized)
    st.sidebar.subheader("All Matrix Types")
    selected_type = st.sidebar.selectbox("Select a type to view its description:", ALL_TYPES_SORTED)
    st.sidebar.markdown(f"**{selected_type}**  \n{MATRIX_DEFINITIONS[selected_type]}")

    use_two_matrices = st.checkbox("Work with two matrices (A and B)?", value=False)

    A = get_matrix("A", default_rows=2, default_cols=2, key_prefix="specA")
    st.write("**Matrix A preview:**")
    st.write(A)
    check_properties(A, "Matrix A")

    if use_two_matrices:
        B = get_matrix("B", default_rows=2, default_cols=2, key_prefix="specB")
        st.write("**Matrix B preview:**")
        st.write(B)
        check_properties(B, "Matrix B")

        if A.shape[1] == B.shape[0]:
            C = A @ B
            st.subheader("**Result of A × B:**")
            st.write(C)
            check_properties(C, "Matrix A × B")
        else:
            st.warning("⚠️ Cannot multiply A × B (dimension mismatch).")
