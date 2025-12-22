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

def play_song(song_name: str) -> tuple[bool, str]:
    try:
        results = sp.search(q=song_name, limit=1, type='track')
        if not results['tracks']['items']:
            return False, f"找不到歌曲: {song_name}"

        track = results['tracks']['items'][0]
        track_uri = track['uri']
        track_name = track['name']
        artist_name = track['artists'][0]['name'] if track.get('artists') else ""

        sp.start_playback(uris=[track_uri])
        return True, f"{track_name} - {artist_name}"

    except Exception as e:
        return False, f"播放失敗: {e}"

if __name__ == "__main__":
    ok, msg = play_song("Never Gonna Give You Up Rick Astley")
    print(ok, msg)