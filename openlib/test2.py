import streamlit as st
import PyPDF2
from pdf2image import convert_from_path
from streamlit.components.v1 import html
import pandas as pd
import altair as alt

# PDF 파일 경로
csv_path = "/workspace/openlib/tsne/ADB_bert-base-uncased_oos_0.75.csv"

# CSV 파일 로드
data = pd.read_csv(csv_path)

# 점 그래프 생성
st.vega_lite_chart(data, {
    'mark': {'type': 'circle', 'tooltip': True},
    'encoding': {
        'x': {'field': 'x축_데이터_열', 'type': 'quantitative'},
        'y': {'field': 'y축_데이터_열', 'type': 'quantitative'},
        'color': {'field': '라벨_데이터_열', 'type': 'quantitative'}
#        'size': {'field': '데이터_설명_열', 'type': 'quantitative'},
    },
})


chart = alt.Chart(data).mark_circle().encode(
    x='x:Q',
    y='y:Q',
    color=alt.Color('label:N', scale=alt.Scale(scheme='plasma')),
    tooltip=['x', 'y', 'label']
)

# 그래프를 Streamlit에서 표시
st.altair_chart(chart, use_container_width=True)
