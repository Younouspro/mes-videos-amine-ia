import json, sys, base64, requests, concurrent.futures as cf
W = sys.argv[1]; speed = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
seg = json.load(open(f"{W}/code/script_fr.json"))
VOIX = "Jrq4GqCKqYpigdQsZRkP"
def gen(i):
    s = seg[i]
    body = {"text": s["texte"], "model_id": "eleven_multilingual_v2", "language_code": "fr",
            "voice_settings": {"stability": 0.45, "similarity_boost": 0.8, "style": 0.25, "use_speaker_boost": True, "speed": speed},
            "previous_text": seg[i-1]["texte"] if i else None,
            "next_text": seg[i+1]["texte"] if i+1 < len(seg) else None}
    r = requests.post(f"https://api.elevenlabs.io/v1/text-to-speech/{VOIX}/with-timestamps?output_format=mp3_44100_128",
                      headers={"xi-api-key": "dummy"}, json=body, timeout=120)
    r.raise_for_status(); d = r.json()
    open(f"{W}/tts/seg{i}.mp3", "wb").write(base64.b64decode(d["audio_base64"]))
    json.dump(d["alignment"], open(f"{W}/tts/seg{i}.json", "w"))
    return i, d["alignment"]["character_end_times_seconds"][-1]
with cf.ThreadPoolExecutor(3) as ex:
    res = dict(ex.map(gen, range(len(seg))))
for i, s in enumerate(seg):
    fen = (seg[i+1]["debut"] if i+1 < len(seg) else 33.57) - s["debut"]
    print(f"{i+1}. durée voix {res[i]:.2f}s / place {fen:.2f}s {'TROP LONG' if res[i] > fen else 'ok'}")
