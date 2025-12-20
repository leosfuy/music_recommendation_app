import mysql.connector
import pandas as pd
import random
import os
from dotenv import load_dotenv

load_dotenv()
#環境帳秘自訂
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", 3306)),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
}

OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "triplets.csv")
NUM_TRIPLETS = 10000

def generate_triplet_csv():
    print("📂 連線至 MySQL (msd.training_songs) ...")
    conn = mysql.connector.connect(**DB_CONFIG)

    print("⏳ 讀取歌曲清單 (song_id, artist_name)...")
    df = pd.read_sql("SELECT song_id, artist_name FROM training_songs", conn)
    conn.close()

    print(f"📊 共有 {len(df)} 首歌，開始整理作者分組...")

    artist_groups = df.groupby("artist_name")["song_id"].apply(list).to_dict()
    valid_artists = [a for a, songs in artist_groups.items() if len(songs) >= 2]

    if len(valid_artists) < 2:
        print("❌ 有效作者數量不足，無法生成三元組！")
        return

    triplets = []
    print(f"🚀 開始生成 {NUM_TRIPLETS} 組 triplets...")

    for _ in range(NUM_TRIPLETS):
        anchor_artist = random.choice(valid_artists)
        songs_of_anchor = artist_groups[anchor_artist]

        anchor_song = random.choice(songs_of_anchor)

        positive_song = random.choice(songs_of_anchor)
        while positive_song == anchor_song:
            positive_song = random.choice(songs_of_anchor)

        negative_artist = random.choice(valid_artists)
        while negative_artist == anchor_artist:
            negative_artist = random.choice(valid_artists)

        negative_song = random.choice(artist_groups[negative_artist])

        triplets.append([anchor_song, positive_song, negative_song])

    triplet_df = pd.DataFrame(triplets, columns=["anchor_id", "positive_id", "negative_id"])
    triplet_df.to_csv(OUTPUT_CSV, index=False)

    print("✅ 生成完成:", OUTPUT_CSV)
    print(triplet_df.head(3))

if __name__ == "__main__":
    generate_triplet_csv()
