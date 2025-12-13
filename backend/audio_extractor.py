import os
import json
import numpy as np
import librosa


def audio_extractor(audio_path, sr=22050, hop_length=512):
    """
    使用 librosa 產出「結構與語意」盡量貼近 MSD / Echo Nest 的 audio analysis
    """

    # =========================
    # 1. 載入音訊
    # =========================
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    duration = float(len(y) / sr)

    # =========================
    # 2. Track-level features
    # =========================
    tempo, beat_frames = librosa.beat.beat_track(
        y=y, sr=sr, hop_length=hop_length
    )
    tempo = float(np.asarray(tempo).item())

    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    loudness = float(librosa.amplitude_to_db(rms, ref=np.max).mean())

    # =========================
    # 3. Beat / Bar / Tatum
    # =========================
    beats_start = librosa.frames_to_time(
        beat_frames, sr=sr, hop_length=hop_length
    ).tolist()

    # MSD 有 bars / tatums，但 librosa 沒有 → 只能用 beat 近似
    bars_start = beats_start[::4]      # 假設 4 beats = 1 bar
    tatums_start = librosa.frames_to_time(
        librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length),
        sr=sr, hop_length=hop_length
    ).tolist()

    # =========================
    # 4. Segment-level features（MSD 核心）
    # =========================
    # 使用 onset 作為 segment 邊界（最接近 Echo Nest 的概念）
    segment_frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=hop_length, backtrack=True
    )
    segment_frames = np.append(segment_frames, len(rms))

    # frame-level features
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, hop_length=hop_length
    )
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=12, hop_length=hop_length
    )

    segments_start = []
    segments_pitches = []
    segments_timbre = []
    segments_loudness_start = []
    segments_loudness_max = []
    segments_loudness_max_time = []

    for i in range(len(segment_frames) - 1):
        a, b = segment_frames[i], segment_frames[i + 1]
        if b <= a:
            continue

        start_time = librosa.frames_to_time(
            a, sr=sr, hop_length=hop_length
        )

        # pitches = chroma (12)
        pitch = np.mean(chroma[:, a:b], axis=1)
        pitch = (pitch / (np.max(pitch) + 1e-9)).tolist()

        # timbre = MFCC-like (12)
        timbre = np.mean(mfcc[:, a:b], axis=1).tolist()

        # loudness
        loud_start = float(rms[a])
        loud_max = float(np.max(rms[a:b]))
        loud_max_frame = a + np.argmax(rms[a:b])
        loud_max_time = librosa.frames_to_time(
            loud_max_frame, sr=sr, hop_length=hop_length
        ) - start_time

        segments_start.append(float(start_time))
        segments_pitches.append(pitch)
        segments_timbre.append(timbre)
        segments_loudness_start.append(loud_start)
        segments_loudness_max.append(loud_max)
        segments_loudness_max_time.append(float(loud_max_time))

    # =========================
    # 5. 組成 MSD-like 結構
    # =========================
    return {
        "track": {
            "duration": duration,
            "tempo": tempo,
            "loudness": loudness
        },
        "bars_start": bars_start,
        "beats_start": beats_start,
        "tatums_start": tatums_start,

        "segments_start": segments_start,
        "segments_pitches": segments_pitches,
        "segments_timbre": segments_timbre,
        "segments_loudness_start": segments_loudness_start,
        "segments_loudness_max": segments_loudness_max,
        "segments_loudness_max_time": segments_loudness_max_time
    }


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    audio = os.path.join(base, "test_data","test_song.wav")

    data = audio_extractor(audio)
    print(json.dumps(data, indent=2, ensure_ascii=False))
