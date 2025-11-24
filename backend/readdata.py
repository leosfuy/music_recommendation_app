import os
import h5py
subset_path="/Users/leeangel/Desktop/資料庫專題/MillionSongSubset"
# 指定你的 subset 主資料夾位置
def get_first_h5(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".h5"):
                return os.path.join(root, file)
    return None

h5_file = get_first_h5(subset_path)

print("Loaded file:", h5_file)

with h5py.File(h5_file, "r") as f:
    print("Available groups:", list(f.keys()))
    
    print("\n--- Song Metadata ---")
    print("Title:", f['metadata']['songs']['title'][0])
    print("Artist:", f['metadata']['songs']['artist_name'][0])
    
    print("\n--- Audio Analysis ---")
    print("Tempo:", f['analysis']['songs']['tempo'][0])
    print("Loudness:", f['analysis']['songs']['loudness'][0])
    print("Key:", f['analysis']['songs']['key'][0])
    print("Mode:", f['analysis']['songs']['mode'][0])
    
    # 每首歌最重要的特徵：timbre（類似 MFCC）
    timbre = f['analysis']['segments_timbre'][:]
    print("\nTimbre shape:", timbre.shape)
