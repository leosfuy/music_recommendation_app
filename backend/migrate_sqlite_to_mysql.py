import sqlite3
import pymysql
import os

# ===== SQLite =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SQLITE_DB = os.path.join(BASE_DIR, "training_data.db")

sqlite_conn = sqlite3.connect(SQLITE_DB)
sqlite_cur = sqlite_conn.cursor()

# ===== MySQL =====
mysql_conn = pymysql.connect(
    host="localhost",
    user="root",
    password="ying1234",
    database="music_recommendation",
    charset="utf8mb4"
)
mysql_cur = mysql_conn.cursor()

# 讀取 SQLite 資料
sqlite_cur.execute("""
SELECT song_id, title, artist_name,
       segments_timbre, segments_pitches, embedding
FROM training_songs
""")

rows = sqlite_cur.fetchall()
print(f"準備搬移 {len(rows)} 筆資料...")

# 清空 MySQL（避免重複）
mysql_cur.execute("DELETE FROM training_songs")

insert_sql = """
INSERT INTO training_songs
(song_id, title, artist_name,
 segments_timbre, segments_pitches, embedding)
VALUES (%s, %s, %s, %s, %s, %s)
"""

count = 0
for row in rows:
    mysql_cur.execute(insert_sql, row)
    count += 1
    if count % 1000 == 0:
        mysql_conn.commit()
        print(f"已搬移 {count} 筆")

mysql_conn.commit()

sqlite_conn.close()
mysql_conn.close()

print("🎉 SQLite → MySQL 資料轉移完成")
