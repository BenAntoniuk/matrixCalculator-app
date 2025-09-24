import streamlit as st
import numpy as np

st.title("Matrix Calculator")

st.write("Enter your matrix (rows separated by semicolons `;`, numbers by spaces).")
matrix_input = st.text_area("Matrix Input", "1 2; 3 4")

# --- Parse the matrix input ---
def parse_matrix(input_str):
    try:
        rows = input_str.strip().split(";")
        matrix = [list(map(float, row.split())) for row in rows]
        return np.array(matrix)
    except Exception:
        return None

A = parse_matrix(matrix_input)

if A is None:
    st.error("Invalid matrix format. Example: `1 2; 3 4`")
else:
    st.write("**Matrix A:**")
    st.write(A)

    # Buttons for operations
    operation = st.selectbox("Choose an operation:", 
                             ["Transpose", "Inverse", "Multiply by Itself", "Eigenvalues"])

    if operation == "Transpose":
        st.write("**Transpose:**")
        st.write(A.T)

    elif operation == "Inverse":
        try:
            inv = np.linalg.inv(A)
            st.write("**Inverse:**")
            st.write(inv)
        except np.linalg.LinAlgError:
            st.error("Matrix is singular and cannot be inverted.")

    elif operation == "Multiply by Itself":
        result = np.dot(A, A)
        st.write("**A × A:**")
        st.write(result)

    elif operation == "Eigenvalues":
        try:
            vals, vecs = np.linalg.eig(A)
            st.write("**Eigenvalues:**")
            st.write(vals)
            st.write("**Eigenvectors:**")
            st.write(vecs)
        except np.linalg.LinAlgError:
            st.error("Eigenvalue calculation failed.")
