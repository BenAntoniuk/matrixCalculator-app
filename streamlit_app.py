import streamlit as st
import numpy as np
import pandas as pd

st.title("Matrix Calculator")

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

# --- Main logic ---
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
