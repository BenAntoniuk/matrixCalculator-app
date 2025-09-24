import streamlit as st
import pandas as pd

st.title("Matrix Calculator")
st.write(
    "Block of code [docs.streamlit.io](https://docs.streamlit.io/)."
)



df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})

df
