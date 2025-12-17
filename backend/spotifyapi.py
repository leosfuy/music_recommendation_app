import spotipy
from spotipy.oauth2 import SpotifyOAuth

# 1. 設定你的憑證 (填入第一步拿到的資料)
CLIENT_ID = 'c40907cd4ead4ce2ada0f89079471c95'
CLIENT_SECRET = '022585f2f73c439a8e8aec24180458c3'
REDIRECT_URI = 'http://127.0.0.1:8888/callback'

# 2. 連線並取得權限 (Scope 很重要，這裡是用來控制播放的)
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope="user-modify-playback-state user-read-playback-state"
))

def play_song(song_name):
    try:
        # 搜尋歌曲
        results = sp.search(q=song_name, limit=1, type='track')
        if not results['tracks']['items']:
            print(f"找不到歌曲: {song_name}")
            return

        track_uri = results['tracks']['items'][0]['uri']
        track_name = results['tracks']['items'][0]['name']
        
        # 開始播放
        sp.start_playback(uris=[track_uri])
        print(f"正在播放: {track_name}")
        
    except Exception as e:
        print("播放失敗 (請確認你的 Spotify App 是打開的，且你是 Premium 會員)")
        print(f"錯誤訊息: {e}")

# --- 測試 ---
# 把這裡換成你模型推薦出來的歌名
play_song("Never Gonna Give You Up Rick Astley")