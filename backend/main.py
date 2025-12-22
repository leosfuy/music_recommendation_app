#api
# 啟動伺服器：uvicorn main:app --reload --host 0.0.0.0 --port 8000
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
import uuid
import shutil
import os
from audio_extractor import audio_extractor_msd_like
from getEmbedding import getEmbedding
from recommend import recommend_from_embedding

from spotifyapi import play_song
from pydantic import BaseModel


app = FastAPI()

# 允許跨來源（Flutter 前端要能連到這裡）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允許所有來源（開發階段方便用）
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload") #連線到/upload就用下面那個upload函式回應這個連線，一個@app對應一個函式
async def upload(file: UploadFile = File(...)): # 從 HTTP multipart/form-data 裡，拿一個必填的檔案欄位，名字叫 file，型別是 UploadFile
    # 產生唯一檔名，避免重複，避免同名檔案被覆蓋
    file_id = str(uuid.uuid4())
    save_path = f"upload_audios/{file_id}_{file.filename}"

    upload_dir = "upload_audios"
    os.makedirs(upload_dir, exist_ok=True)  # 如果資料夾不存在就建立
    # 存音檔
    with open(save_path, "wb") as buffer: #二進位格式，不改變檔案內容，原本是什麼檔就是什麼檔
        shutil.copyfileobj(file.file, buffer)

    # 抓特徵
    UploadAudioFeature = audio_extractor_msd_like(save_path)
    # 轉embedding
    UploadAudioEmbedding = getEmbedding(UploadAudioFeature["timbre_segments"], UploadAudioFeature["pitches_segments"])
    #取最相似5首歌特徵
    result = recommend_from_embedding(UploadAudioEmbedding)
    print(result)

    return result

# 新增功能：點選某首歌 → 呼叫 /play → 播放
class PlayRequest(BaseModel):  # 定義前端要傳什麼資料
    title: str | None = None
    artist_name: str | None = None
    song_query: str | None = None  # 前端也可以直接傳 "歌名 歌手"


@app.post("/play")
async def play(req: PlayRequest):
    """
    前端點選某首歌時呼叫這個 API
    你可以傳：
      1) song_query: "歌名 歌手"
      或
      2) title + artist_name
    """

    # 決定要拿什麼去搜尋 Spotify
    if req.song_query and req.song_query.strip():
        query = req.song_query.strip()
    else:
        title = (req.title or "").strip()
        artist = (req.artist_name or "").strip()
        query = f"{title} {artist}".strip()

    if not query:
        return {"status": "error", "message": "缺少 song_query 或 title/artist_name"}

    # 呼叫你 spotifyapi.py 的 play_song 來播放   把同步 Spotify 呼叫丟進 thread pool
    success, message = await run_in_threadpool(play_song, query)

    if success:
        return {"status": "success", "playing": message}
    else:
        return {"status": "error", "message": message}


