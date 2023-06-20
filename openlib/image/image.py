import streamlit as st

# 이미지 업로드
uploaded_file = st.file_uploader("Choose a PNG file", type="png")

# 이미지 다운로드
if uploaded_file is not None:
    with open("howtoresult.png", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("Image saved")

