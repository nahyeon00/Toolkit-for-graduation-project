from typing import Union
from typing import Dict

from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Form
import subprocess
from fastapi.responses import JSONResponse
import os
from pydantic import BaseModel

import shutil

import uvicorn

app = FastAPI()

class Input(BaseModel):
    method: str
    model_name_or_path: str
    dataset: str
    seed: int
    max_epoch: int
    known_cls_ratio: float
    ip: str

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), file_name: str =Form(...)):

    folder_path = f"/workspace/openlib/data/{file_name}"
    os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(folder_path, file.filename)

    with open(file_path, "wb") as f:
        contents = await file.read()
        f.write(contents)


    shutil.unpack_archive(file_path, folder_path)
    os.remove(file_path)
#    extracted_folder = os.path.join(folder_path, file_name)

#    for filename in os.listdir(file_path):
#        file_path2 = os.path.join(file_path,filename)
#        shutil.move(file_path2, folder_path)

#    os.remove(extracted_folder)

@app.post("/run")
async def run(input: Dict[str, Union[str, int]]):
    method = input.get('method')
    model_name_or_path = input.get('model_name_or_path')
    dataset = input.get('dataset')
    #seed = input.get('seed')
    #max_epoch = input.get('max_epoch')
    known_cls_ratio = input.get('known_cls_ratio')

    if method == 'ADB':
        subprocess.Popen(["./personalize.sh", str(method), str(model_name_or_path) ,str(dataset), str(known_cls_ratio)])

    elif method == 'KNN-contrastive learning':
        method = 'KNN'
        subprocess.Popen(["./knncl_personalize.sh", str(method), str(model_name_or_path) ,str(dataset), str(known_cls_ratio)])
    
    #return JSONResponse({result})

@app.post("/iput")
async def iput(input: Dict[str, Union[str, int]]):
#async def iput(input: Input):
    method = input.get('method')
    model_name_or_path = input.get('model_name_or_path')
    dataset = input.get('dataset')
    #seed = input.get('seed')
    max_epoch = input.get('max_epoch')
    known_cls_ratio = input.get('known_cls_ratio')
    ip = input.get('ip')

    if method == 'ADB':
        subprocess.Popen(["./personalize_input.sh", str(method), str(model_name_or_path) ,str(dataset), str(known_cls_ratio), str(ip), str(max_epoch)])

#이거 보류!!!!!
    elif method == 'K+1':
        subprocess.Popen(["./k_1_personalize.sh", str(method), str(model_name_or_path) ,str(dataset), str(known_cls_ratio), str(ip), str(max_epoch)])
    
    else :
        method = 'KNN'
#        print(method)
        subprocess.Popen(["./knncl_personalize_input.sh", str(method), str(model_name_or_path) ,str(dataset), str(known_cls_ratio), str(ip), str(max_epoch)])

    #return {"result": result}

@app.delete("/delete/{file_name}")
async def delete_file(file_name: str):

    folder_path = f"/workspace/openlib/data/{file_name}"

    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        #os.remove(file_name)
        return {"success":True}
    else :
        return {"message": "File not found."}

@app.post("/train")
async def train(input: Dict[str, Union[str, int]]):
    method = input.get('method')
    model_name_or_path = input.get('model_name_or_path')
    dataset = input.get('dataset')
    #seed = input.get('seed')
    max_epoch = input.get('max_epoch')
    known_cls_ratio = input.get('known_cls_ratio')

    #process = subprocess.Popen(["./all.sh", str(model_name_or_path) ,str(dataset), str(known_cls_ratio)])
    try:
        if method == 'ADB':
            subprocess.Popen(["./all.sh", str(method), str(model_name_or_path) ,str(dataset), str(known_cls_ratio), str(max_epoch)])
        elif method == 'KNN-contrastive learning':
            method = 'KNN'
            subprocess.Popen(["./KNN_all.sh", str(method), str(model_name_or_path) ,str(dataset), str(known_cls_ratio), str(max_epoch)])

        elif method == 'K+1':
            subprocess.Popen(["./K_all.sh", str(method), str(model_name_or_path) ,str(dataset), str(known_cls_ratio), str(max_epoch)])
        
    except subprocess.CalledProcessError:
        return JSONResponse({"result": 1})

@app.post("/stop")
async def stop():
    subprocess.run(["sh", "stop2.sh"])
    #model_name_or_path = input.get('model_name_or_path')
    #dataset = input.get('dataset')
    #known_cls_ratio = input.get('known_cls_ratio')
    #process = subprocess.Popen(["./stop.sh", str(model_name_or_path) ,str(dataset), str(known_cls_ratio)])

