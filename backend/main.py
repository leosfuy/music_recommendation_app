#api
# 啟動伺服器：uvicorn main:app --reload --port 8000
from fastapi import FastAPI, UploadFile, File
import uuid
import shutil
from audio_extractor import audio_extractor_msd_like

app = FastAPI()

@app.post("/upload") #連線到/upload就用下面那個upload函式回應這個連線，一個@app對應一個函式
async def upload(file: UploadFile = File(...)): # 從 HTTP multipart/form-data 裡，拿一個必填的檔案欄位，名字叫 file，型別是 UploadFile
    # 產生唯一檔名，避免重複，避免同名檔案被覆蓋
    file_id = str(uuid.uuid4())
    save_path = f"upload_audios/{file_id}_{file.filename}"

    # 存音檔
    with open(save_path, "wb") as buffer: #二進位格式，不改變檔案內容，原本是什麼檔就是什麼檔
        shutil.copyfileobj(file.file, buffer)

    # 抓特徵
    UploadAudioFeature = audio_extractor_msd_like(save_path)
    # 轉embedding

    return {
        "fileName": file.filename,
        "result": result
    }


