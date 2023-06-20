import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode, JsCode
from st_aggrid.shared import GridUpdateMode

# 데이터프레임 생성
data_info = pd.read_csv("data_info.csv")

# 선택된 행의 정보를 저장할 변수

#selected_row_indexes = []

def aggrid_interactive_table(df: pd.DataFrame):

    options = GridOptionsBuilder.from_dataframe(
        df, enableRowGroup=True, enableValue=True, enablePivot=True
    )
    options.configure_side_bar()
    options.configure_selection("single")

# ag-Grid 테이블 생성
    grid_response = AgGrid(
        df,
        enable_enterprise_modules=True,
        gridOptions=options.build(),
        width='100%',
        height='400px',
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=True,
    )

    return grid_response

grid_response = aggrid_interactive_table(df=data_info)

if grid_response:
#    st.write("You selected:")
#    st.json(grid_response["selected_rows"])

    selected_rows = grid_response["selected_rows"]

    # 선택된 행에서 불필요한 _selectedRowNodeInfo 제거
    selected_rows_cleaned = []
    for row in selected_rows:
        row_cleaned = {key: value for key, value in row.items() if not key.startswith("_")}
        selected_rows_cleaned.append(row_cleaned)

    # 선택된 행의 원하는 데이터 출력
    if selected_rows_cleaned:
        #selected_data = pd.DataFrame(selected_rows_cleaned)
        #st.write(selected_data)
        
        for row in selected_rows_cleaned:
            #st.write("Method :", row["Method"])
            #st.write("Model name or path :", row["Model name or path"])
            #st.write("Dataset :", row["Dataset"])
            st.write("Seed :", row["Seed"])
            st.write("Max Epoch :", row["Max epoch"])
            st.write("Known cls ratio :", row["Known cls ratio"])


