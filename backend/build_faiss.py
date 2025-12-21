import mysql.connector
import numpy as np
import faiss
import os
from dotenv import load_dotenv #載環境

load_dotenv()

def build_faiss(index_path="song_embeddings.index"):
    conn = mysql.connector.connect(
        host = os.getenv("DB_HOST"),
        port = int(os.getenv("DB_PORT", "3306")),
        user = os.getenv("DB_USER"),
        password = os.getenv("DB_PASSWORD"),
        database = os.getenv("DB_NAME")
    )
    
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, embedding
        FROM songs
        WHERE embedding IS NOT NULL
        ORDER BY id
    """)
    rows = cursor.fetchall()
    conn.close()

    ids = []
    vectors = []

    for id, blob in rows:
        vec = np.frombuffer(blob, dtype=np.float32)
        vectors.append(vec)
        ids.append(id)

    X = np.vstack(vectors).astype(np.float32)

    # cosine similarity = inner product 
    

    dim = X.shape[1]
    # 使用精確查詢的 IndexFlatIP
    index = faiss.IndexFlatIP(dim)
    index.add(X)

    faiss.write_index(index, index_path)
    np.save("ids.npy", np.array(ids))

    print("Faiss index 建立完成")

if __name__ == "__main__":
    build_faiss()
