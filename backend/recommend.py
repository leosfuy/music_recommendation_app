import mysql.connector
import numpy as np
import faiss
import os
from dotenv import load_dotenv #載環境
import base64
import requests

load_dotenv()

#檔案被import時就會執行
# 載入 index
INDEX_PATH = "song_embeddings.index"
IDS_PATH = "ids.npy"
CLIENT_ID = 'c40907cd4ead4ce2ada0f89079471c95'
CLIENT_SECRET = '022585f2f73c439a8e8aec24180458c3'

if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError(
        f"找不到 FAISS index 檔案：{INDEX_PATH}，請先執行 build_faiss()"
    )

if not os.path.exists(IDS_PATH):
    raise FileNotFoundError(
        f"找不到 ids 檔案：{IDS_PATH}，請先執行 build_faiss()"
    )

index = faiss.read_index(INDEX_PATH)
ids = np.load(IDS_PATH)

def recommend_from_embedding(input_embedding, top_k=5):
    # 正規化（cosine）
    input_embedding = input_embedding.astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(input_embedding)

    # 查詢
    scores, indices = index.search(input_embedding, top_k) # scores, indices是二維，可以一次查多筆

    matched_ids = ids[indices[0]]
    similarities = scores[0]

    # 查 MySQL 撈歌曲資訊
    conn = mysql.connector.connect(
        host = os.getenv("DB_HOST"),
        port = int(os.getenv("DB_PORT", "3306")),
        user = os.getenv("DB_USER"),
        password = os.getenv("DB_PASSWORD"),
        database = os.getenv("DB_NAME")
    )
    cursor = conn.cursor(dictionary=True)

    format_strings = ",".join(["%s"] * len(matched_ids)) # len(matched_ids)個%s 用,分隔
    cursor.execute(f"""
        SELECT id, title, artist_name, year,
               duration, tempo, loudness, `key`, mode
        FROM songs
        WHERE id IN ({format_strings})
    """, tuple(map(int, matched_ids)))

    rows = cursor.fetchall()
    conn.close()

    song_map = {r["id"]: r for r in rows} # dict r["id"]是key r是value  eg 18392: {"id": 18392, "title": "Song A", "artist": "Artist1"}

    results = []
    for sid, sim in zip(matched_ids, similarities):
        r = song_map[sid]
        preview_url = get_preview_url(r["title"], r["artist_name"])  # <-- 呼叫 Spotify API
        results.append({
            "title": str(r["title"]),
            "artist": str(r["artist_name"]),
            "year": str(r["year"]),
            "duration": str(r["duration"]),
            "tempo": str(r["tempo"]),
            "loudness": str(r["loudness"]),
            "key": str(r["key"]),
            "mode": str(r["mode"]),
            "similarity": str(float(sim)),
            "preview_url": preview_url
        })

    return results

def get_access_token():
    auth_str = f"{CLIENT_ID}:{CLIENT_SECRET}"
    b64_auth_str = base64.b64encode(auth_str.encode()).decode()
    headers = {"Authorization": f"Basic {b64_auth_str}"}
    data = {"grant_type": "client_credentials"}
    res = requests.post("https://accounts.spotify.com/api/token", headers=headers, data=data)
    res.raise_for_status()
    return res.json()["access_token"]

def get_preview_url(title, artist):
    token = get_access_token()
    headers = {"Authorization": f"Bearer {token}"}
    query = f"track:{title} artist:{artist}"
    params = {"q": query, "type": "track", "limit": 1}
    res = requests.get("https://api.spotify.com/v1/search", headers=headers, params=params)
    res.raise_for_status()
    items = res.json()["tracks"]["items"]
    if not items:
        return None
    return items[0]["preview_url"]