import streamlit as st
import numpy as np
import pandas as pd

st.title("Matrix Calculator")

# Ask for matrix dimensions
rows = st.number_input("Number of rows", min_value=1, max_value=6, value=2)
cols = st.number_input("Number of columns", min_value=1, max_value=6, value=2)

# Create empty DataFrame for user to fill
default_data = np.zeros((rows, cols))
df = pd.DataFrame(default_data, dtype=float)

st.write("Enter your matrix values:")
matrix_input = st.data_editor(df, num_rows="dynamic")

# Convert DataFrame to numpy array
A = matrix_input.to_numpy()

st.write("**Matrix A:**")
st.write(A)

# Pick operation
operation = st.selectbox("Choose an operation:", 
                         ["Transpose", "Inverse", "Multiply by Itself", "Eigenvalues"])

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
