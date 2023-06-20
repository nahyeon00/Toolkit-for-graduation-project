import requests
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import os
import io
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import ast
from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode, JsCode, ColumnsAutoSizeMode
from st_aggrid.shared import GridUpdateMode
import time

import PyPDF2
from pdf2image import convert_from_path
import altair as alt
import seaborn as sns

#st.set_page_config(layout="wide")

# 파일 정보가 저장된 CSV 파일이 있는지 확인하고 없으면 새로 생성합니다.
if not os.path.isfile("file_info.csv"):
    df = pd.DataFrame(columns=["Name", "Description"])
    df.to_csv("file_info.csv", index=False)


# 파일 정보가 저장된 CSV 파일이 있는지 확인하고 없으면 새로 생성합sl다.
if not os.path.isfile("data_info.csv"):
    df = pd.DataFrame(columns=["Method", "Model name or path", "Dataset", "Known cls ratio", "Max epoch"])
    df.to_csv("data_info.csv", index=False)

st.title('오픈 의도 분류 웹사이트')
#st.subheader('')

with st.sidebar:
    selected = option_menu("Main Menu", ["Home", "Data Upload", "Learning", "Results"], icons=['house','upload', 'book', 'card-text'], menu_icon="cast", default_index=0)
    selected

if selected == 'Home':
   
    #사용 방법 정의한 사진
    st.image("/workspace/openlib/image/main.png", use_column_width=True)
    st.markdown('[Link Click](https://github.com/nahyeon00/ADB_study)')

elif selected == 'Data Upload':
    if st.sidebar.button('💡어떻게 사용하는지 모르겠어요.'):
        st.sidebar.image("/workspace/openlib/image/howtoupload.png", use_column_width=True)
    
    #st.sidebar.expander('❓사용 방법❓'):
    #    st.sidebar.image('/workspace/openlib/image/howtoupload.png', use_column_width=True)

    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.info("데이터 폴더를 업로드하는 페이지입니다.")
    st.caption("다음과 같은 형식의 압축된 폴더만 업로드 가능합니다.")
    st.image("/workspace/openlib/image/upload.png")

    # 기존에 업로드한 파일 정보가 저장된 파일이 있다면 불러옵니다.
    if os.path.isfile("file_info.csv"):
        file_info = pd.read_csv("file_info.csv")
        new_file_info = st.data_editor(file_info, num_rows="dynamic")

        if not new_file_info.empty:          
            deleted_row_indices = set(file_info.index) - set(new_file_info.index)
            if deleted_row_indices:
                for i in deleted_row_indices:
                    deleted_file_name = file_info.loc[i, "Name"]
                    response = requests.delete(f"http://210.117.181.115:8010/delete/{deleted_file_name}")
            new_file_info.to_csv("file_info.csv", index=False)

    with st.expander('데이터 분석하기'):
        if os.path.isfile("file_info.csv"):
            file_info = pd.read_csv("file_info.csv")

        def aggrid_interactive_table(df: pd.DataFrame):

            options = GridOptionsBuilder.from_dataframe(df, enableRowGroup=True, enableValue=True, enablePivot=True, suppress_horizontal_scroll=True)
        #options.configure_column("all",width=200)
        #options.configure_default_column_width(100)
            options.configure_side_bar()
            options.configure_selection("single")
        #grid_options = GridOptionsBuilder().from_dataframe(df).build()
        #grid_options.auto_size_columns = True


# ag-Grid 테이블 생성
            grid_response = AgGrid(
                df,
                enable_enterprise_modules=True,
                gridOptions=options.build(),
                width='100%',
                height='300px',
                columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
                update_mode=GridUpdateMode.MODEL_CHANGED,
                allow_unsafe_jscode=True,
            )

            return grid_response

        grid_response = aggrid_interactive_table(df=file_info)

        if grid_response:
            selected_rows = grid_response["selected_rows"]
            selected_rows_cleaned = []
            for row in selected_rows:
                row_cleaned = {key: value for key, value in row.items() if not key.startswith("_")}
                selected_rows_cleaned.append(row_cleaned)

            # 선택된 행의 원하는 데이터 출력
            if selected_rows_cleaned:
                per_info=False

                for row in selected_rows_cleaned:
                    tsv=st.selectbox(" ", ["train", "dev", "test"])
                    ggg = pd.read_csv('/workspace/openlib/data/'+row["Name"]+'/'+tsv+'.tsv', delimiter='\t')
                    st.table(ggg.head(5))

                    label_counts = ggg['label'].value_counts()
                    top_labels = label_counts.head(5)
                    # 그래프 생성
                    fig, ax = plt.subplots()
                    colors = ['#3366FF', '#6690FF', '#84A9FF', '#ADC8FF', '#D6E4FF']
                    ax.bar(top_labels.index, top_labels.values, color=colors)

                    # 그래프 축 및 타이틀 설정
                    ax.set_xlabel('Label')
                    ax.set_ylabel('Count')
                    ax.set_title('Top 5 Label Counts')

                    # 그래프 출력
                    st.pyplot(fig)

                  

#버튼 크기 950px/ 기본 : 620px
    st.markdown("""
    <style>
    .stTextInput {
        margin-top : -40px;
    }
    .stFileUploader {
        margin-top : -40px;
    }
    .stButton {
        display: flex;
        justify-content: flex-end;
        color: #FF4B4B;
    }
    </style>
    <h2 style='font-size: 20px;'>Name</h2>
    """, unsafe_allow_html=True)
    file_name = st.text_input(' ', placeholder='업로드 할 폴더 이름을 입력하세요.',help="업로드 할 폴더 이름을 입력해주세요.")
    st.markdown("<h2 style='font-size: 20px;'>Description</h2>", unsafe_allow_html=True)
    description = st.text_input(' ',placeholder='폴더에 대한 설명을 입력하세요.', help="업로드 할 데이터 설명을 입력해주세요.")
    # 업로드할 파일 선택
    file = st.file_uploader("Zip 파일 업로드 하기", type="zip")
    uploaded_file = file

    if st.button("Upload") and uploaded_file is not None:
        #size = f'{round(len(file.getvalue()) / 1024, 2)} KB'
        uploaded_file.seek(0)
        #time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        files = {"file": uploaded_file}
        data = {"file_name": file_name}
        response = requests.post("http://210.117.181.115:8010/upload", files=files, data=data)
        
        # 결과를 출력합니다.
        st.write(response.json())

        # 새로 업로드한 파일 정보를 DataFrame으로 만듭니다.
        new_file_info = pd.DataFrame({
            "Name": [file_name],
            "Description": [description]
            #"Size": [size],
            #"Upload Time": [time]
        })

        # 기존 파일 정보와 새로운 파일 정보를 합쳐서 파일에 저장합니다.
        file_info = pd.concat([file_info, new_file_info], ignore_index=True)
        file_info.to_csv("file_info.csv", index=False)
          
        st.experimental_rerun()

elif selected == 'Learning':
    
    if st.sidebar.button('💡어떻게 사용하는지 모르겠어요.'):
        st.sidebar.image("/workspace/openlib/image/howtolearn.png", use_column_width=True)


    st.info("원하는 조건으로 모델을 학습하는 페이지입니다.")
    st.caption(" ")
    col1, col2 = st.columns([3,1])

    
    # 기존에 업로드한 파일 정보가 저장된 파일이 있다면 불러옵니다.
    if os.path.isfile("data_info.csv"):
        data_info = pd.read_csv("data_info.csv")
        new_data_info = st.data_editor(data_info, num_rows="dynamic")
        new_data_info.to_csv("data_info.csv", index=False)
    
    #temp_info = pd.read_csv("data_info.csv")
    
    st.markdown("""
    <style>
    .stTextInput {
        margin-top : -40px;
    }
    </style>
    <style>
    .stSelectbox {
        margin-top : -40px;
    }
    <style>
    .stCheckbox {
        margin-bottom : -10px;
    }
    </style>
    <h2 style='font-size: 20px;'>Method</h2>
    """, unsafe_allow_html=True)
    method = st.selectbox(" ",["ADB", "KNN-contrastive learning", "K+1"])
    with st.expander("Method 모델 설명"):
        model = st.selectbox(" ", ["ADB", "KNN-contrastive learning", "K+1"], key='model')
        if model == "ADB":
            st.image("/workspace/openlib/image/ADB.png")
            st.markdown('[What is ADB?](https://github.com/thuiar/Adaptive-Decision-Boundary)')
        elif model == 'K+1':
            st.image("/workspace/openlib/image/K+1.png")
            st.markdown('[What is (K+1)-way?](https://github.com/fanolabs/out-of-scope-intent-detection)')
        else :
            st.image("/workspace/openlib/image/KNN.png")
            st.markdown('[What is KNNCL?](https://github.com/zyh190507/KnnContrastiveForOOD)')
        
    st.markdown("<h2 style='font-size: 20px;'>Pretrained Language Model</h2>", unsafe_allow_html=True)
    model_name_or_path = st.text_input(' ', placeholder='Hugging Face에서 지원하는 모델만 가능합니다.')
    st.markdown('[Hugging Face에서 모델명 찾기](https://huggingface.co/models)')
    st.markdown("<h2 style='font-size: 20px;'>Dataset</h2>", unsafe_allow_html=True)
    dataset = st.text_input(' ', placeholder='폴더명을 입력하세요.')


    with st.expander('저장된 dataset 폴더명 보기'):
        if os.path.isfile("file_info.csv"):
            file_info = pd.read_csv("file_info.csv")
            st.dataframe(file_info)
            
#    st.markdown("<h2 style='font-size: 20px;'>Seed</h2>", unsafe_allow_html=True)
#    seed = st.text_input(' ', key='seed')
    st.markdown("<h2 style='font-size: 20px;'>Max epoch</h2>", unsafe_allow_html=True)
    #max_epoch = st.slider(' ', 1,100)
    max_epoch = st.text_input(' ', key='max_epoch', placeholder='1 이상의 자연수를 입력하세요.')
    st.markdown("<h2 style='font-size: 20px;'>Known cls ratio</h2>", unsafe_allow_html=True)
    known_cls_ratio = st.selectbox(" ",["0.25","0.5","0.75"])

#wide 일땐 950px, 기본 일땐 620px

    st.markdown("""
    <style>
    .stButton {
        display: flex;
        justify-content: flex-end;
        color: #FF4B4B;
    }
    </style>
    """, unsafe_allow_html=True)


    if st.button('Training') and method  and model_name_or_path and dataset and max_epoch and known_cls_ratio:
        #st.markdown('[Link to the thesis](https://github.com/nahyeon00/ADB_study)')

        # 전송할 데이터
        data = {
            "method": method,
            "model_name_or_path": model_name_or_path,
            "dataset": dataset,
            #"seed": seed,
            "max_epoch" : max_epoch,
            "known_cls_ratio" : known_cls_ratio

        }
            
        #with st.spinner('Wait for it...'):
        response = requests.post("http://210.117.181.115:8010/train", json=data)
        st.write(response.json())

       # result = response.json().get("result")
        #if result == 0:
        new_data_info = pd.DataFrame({
            "Method": [method],
            "Model name or path": [model_name_or_path],
            "Dataset": [dataset],
            #"Seed" : [seed],
            "Max epoch": [max_epoch],
            "Known cls ratio": [known_cls_ratio]
        })
        data_info = pd.concat([data_info, new_data_info], ignore_index=True)
        data_info.to_csv("data_info.csv", index=False)

        st.experimental_rerun()

    #st.markdown(f'<button style="{stop_button_style}">Stop</button>', unsafe_allow_html=True)
 
    if st.button("🚫", key="stop_button"):
        try:
            if os.path.isfile("data_info.csv"):
                data_info = pd.read_csv("data_info.csv")
        #st.experimental_rerun()
                data_info = data_info.drop(data_info.index[-1])
                data_info.to_csv("data_info.csv", index=False)
                response = requests.post("http://210.117.181.115:8010/stop")
                st.experimental_rerun()

            #response = requests.post("http://210.117.181.115:8010/stop")

        #    response.raise_for_status()
                st.write(response.json())
          #data_info = data_info.drop(data_info.index[-1])
            #data_into.to_csv("data_info.csv", index=False)
            #st.experimental_rerun()
        #    st.experimental_rerun()

        except requests.exceptions.RequestException as e:
        #    temp_info.to_csv("data_info.csv", index=False)
            st.experimental_rerun()
    #st.experimental_rerun()


elif selected == 'Results':

    st.sidebar.markdown("""
    <style>
    .stButton {
        display: flex;
        justify-content: flex-end;
        color: #FF4B4B;
    }
    </style>
    """, unsafe_allow_html=True)


    if st.sidebar.button('💡어떻게 사용하는지 모르겠어요.'):
        st.sidebar.image("/workspace/openlib/image/howtoresult.png", use_column_width=True)
    st.info("힉습된 모델의 결과를 보는 페이지입니다.")
    col1, col2 = st.columns([3,1])
    
    
    if os.path.isfile("data_info.csv"):
        data_info = pd.read_csv("data_info.csv")
    

    def aggrid_interactive_table(df: pd.DataFrame):

        options = GridOptionsBuilder.from_dataframe(df, enableRowGroup=True, enableValue=True, enablePivot=True, suppress_horizontal_scroll=True)
        #options.configure_column("all",width=200)
        #options.configure_default_column_width(100)
        options.configure_side_bar()
        options.configure_selection("single")
        #grid_options = GridOptionsBuilder().from_dataframe(df).build()
        #grid_options.auto_size_columns = True


# ag-Grid 테이블 생성
        grid_response = AgGrid(
            df,
            enable_enterprise_modules=True,
            gridOptions=options.build(),
            width='100%',
            height='300px',
            columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
            update_mode=GridUpdateMode.MODEL_CHANGED,
            allow_unsafe_jscode=True,
        )

        return grid_response

    with col1:
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
            per_info=False

            for row in selected_rows_cleaned:
                #with col1:
                data = {
                    "method": row["Method"],
                    "model_name_or_path": row["Model name or path"],
                    "dataset": row["Dataset"],
                        #"seed": row["seed"],
                    "max_epoch" : row["Max epoch"],
                    "known_cls_ratio" : row["Known cls ratio"]
                }

                            #if os.path.isfile("data_info.csv"):
                            #    data_info = pd.read_csv("data_info.csv")
                            #    st.dataframe(data_info)

                            #per_info=pd.read_csv("/workspace/openlib/csv/ADB_bert-base-uncased_banking_0.25.csv")
                            #st.dataframe(per_info)
                            #if os.path.isfile('./workspace/openlib/csv/'+row["Method"]+'_'+row["Model name or path"]+'_'+row["Dataset"]+'_'+str(row["Known cls ratio"])+'.csv'):
                if row["Method"] == "KNN-contrastive learning":
                    row["Method"] = "KNN"

                    #pdf_file_path = "/workspace/openlib/tsne/"+row["Method"]+'_'+row["Model name or path"]+'_'+row["Dataset"]+'_'+str(row["Known cls ratio"])+".pdf"
                    #images = convert_from_path(pdf_file_path)

                    #for i, image in enumerate(images):
                        #st.image(image, caption=f"Page {i+1}", use_column_width=True)

                
                if os.path.isfile('/workspace/openlib/csv/'+row["Method"]+'_'+row["Model name or path"]+'_'+row["Dataset"]+'_'+str(row["Known cls ratio"])+'_'+str(row["Max epoch"])+'.csv'):
                    per_info = pd.read_csv('/workspace/openlib/csv/'+row["Method"]+'_'+row["Model name or path"]+'_'+row["Dataset"]+'_'+str(row["Known cls ratio"])+'_'+str(row["Max epoch"])+'.csv')
                    col2.dataframe(per_info.columns[1:])

                else :
                    st.text(" ")


                #wide : 253px / 기본 : 173px
                #margin-top : -54px;
                #margin-left: 70px;

                st.markdown("""
                <style>
                .stTextInput {
                margin-top : -40px;
                }
                </style>
                <h2 style='font-size: 20px;'>Input</h2>
                """, unsafe_allow_html=True)
                   
                ip = st.text_input(" ", placeholder="검색 단어를 입력하세요.")
                data = {
                    "method": row["Method"],
                    "model_name_or_path": row["Model name or path"],
                    "dataset": row["Dataset"],
                    #"seed": row["seed"],
                    "max_epoch" : row["Max epoch"],
                    "known_cls_ratio" : row["Known cls ratio"],
                    "ip" : ip
                }

                if os.path.isfile("/workspace/openlib/tsne/"+row["Method"]+'_'+row["Model name or path"]+'_'+row["Dataset"]+'_'+str(row["Known cls ratio"])+'_'+str(row["Max epoch"])+".csv"):
                    graph = pd.read_csv("/workspace/openlib/tsne/"+row["Method"]+'_'+row["Model name or path"]+'_'+row["Dataset"]+'_'+str(row["Known cls ratio"])+'_'+str(row["Max epoch"])+".csv")
#                else :
#                    st.warning("아직 학습을 진행중 입니다.")

                    if st.button('검색') and ip:
                        st.empty()
                        if os.path.isfile("output.txt"):
                            os.remove("output.txt")
                        with st.spinner('Wait for it...'):
                            response = requests.post("http://210.117.181.115:8010/iput", json=data)
                        
                            while True:
                                if os.path.isfile("output.txt"):
                                    with open("output.txt", "r") as file:
                                        text = file.read()
                                    st.info("예측된 Intent Label : " + text)
                                    break
                                else :
                                    time.sleep(1)
                        #else :
                            #st.warning("
                            #col1.dataframe(per_info.columns[1:])


                #csv_path = "/workspace/openlib/tsne/"+row["Method"]+'_'+row["Model name or path"]+'_'+row["Dataset"]+'_'+str(row["Known cls ratio"])+".csv"
                #graph = pd.read_csv(csv_path)
                        last_row = graph.iloc[-1]

                        new_point = pd.DataFrame({'x':[last_row['x']], 'y':[last_row['y']], 'label': [last_row['label']], 'sentence':[ip]})
                    #graph2 = graph.append(new_point, ignore_index=True)
                        chart = alt.Chart(graph).mark_circle().encode(
                            x='x:Q',
                            y='y:Q',
                            color=alt.Color('label:N', scale=alt.Scale(scheme='plasma')),
                            tooltip=['x', 'y', 'label','sentence']
                            #tooltip=['x', 'y', 'label']
                        ).interactive()
                        data['color']='red'
                        chart1 = alt.Chart(new_point).mark_point(shape='triangle', filled=True, color='green').encode(
                            x='x:Q',
                            y='y:Q',
                            size=alt.value(300),
                            tooltip=['x', 'y', 'label','sentence']
                            #tooltip=['x', 'y', 'label']
                        ).interactive()
                    

            # 그래프를 Streamlit에서 표시
                        st.altair_chart(chart+chart1, use_container_width=True)
                        graph=graph.drop(graph.index[-1])
                        graph.to_csv("/workspace/openlib/tsne/"+row["Method"]+'_'+row["Model name or path"]+'_'+row["Dataset"]+'_'+str(row["Known cls ratio"])+'_'+str(row["Max epoch"])+".csv", index=False)


                       # st.experimental_rerun()
                    else :
#                        data1=pd.read_csv("/workspace/openlib/tsne/"+row["Method"]+'_'+row["Model name or path"]+'_'+row["Dataset"]+'_'+str(row["Known cls ratio"])+'_'+str(row["Max epoch"])+".csv")
#                        data2=pd.read_csv("/workspace/openlib/csv/"+row["Method"]+'_'+row["Model name or path"]+'_'+row["Dataset"]+'_'+str(row["Known cls ratio"])+'_'+str(row["Max epoch"])+".csv")

#                        merged_data=pd.merge(data1, data2, left_on='label', right_on='0')
                        chart = alt.Chart(graph).mark_circle().encode(
                        #chart = alt.Chart(merged_data).mark_circle().encode(
                            x='x:Q',
                            y='y:Q',
                            color=alt.Color('label:N', scale=alt.Scale(scheme='plasma')),
                           #color = alt.Color("label:N", scale=alt.Scale(domain=[i for i in range(label_counts)])),
                            tooltip=['x', 'y', 'label','sentence']
                            #tooltip=['x', 'y', 'label']
                        ).interactive()
                        st.altair_chart(chart, use_container_width=True)

                else :
                    st.text(" ")

                if os.path.isfile('/workspace/openlib/tsne/'+row["Method"]+'_'+row["Model name or path"]+'_'+row["Dataset"]+'_'+str(row["Known cls ratio"])+'_'+str(row["Max epoch"])+'.pdf'):
                    pdf_path = '/workspace/openlib/tsne/'+row["Method"]+'_'+row["Model name or path"]+'_'+row["Dataset"]+'_'+str(row["Known cls ratio"])+'_'+str(row["Max epoch"])+'.pdf'
                    images = convert_from_path(pdf_path)
                    for i, image in enumerate(images):
                        st.image(image, caption=f"Page {i+1}", use_column_width=True)
                else :
                    st.warning("아직 학습이 진행중 입니다.")
