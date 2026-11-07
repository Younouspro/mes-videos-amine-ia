# Assemble la voix française : coupe les silences, place chaque phrase, calcule les mots minutés
import json, sys, subprocess, numpy as np
W = sys.argv[1]; SR = 44100; FIN_VIDEO = 1096 / 30; CARTE = 3.0
seg = json.load(open(f"{W}/code/script_fr.json"))
def lire(p):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.float32)
total = FIN_VIDEO + CARTE
piste = np.zeros(int(total * SR) + SR, np.float32)
mots = []; t_libre = 0.0
for i, s in enumerate(seg):
    a = lire(f"{W}/tts/seg{i}.mp3"); al = json.load(open(f"{W}/tts/seg{i}.json"))
    env = np.convolve(np.abs(a), np.ones(441) / 441, "same"); seuil = 10 ** (-42 / 20)
    idx = np.where(env > seuil)[0]; d0 = max(0, idx[0] / SR - 0.03); d1 = min(len(a) / SR, idx[-1] / SR + 0.08)
    clip = a[int(d0 * SR):int(d1 * SR)]
    depart = max(s["debut"], t_libre + 0.12)
    n = int(depart * SR); piste[n:n + len(clip)] += clip
    t_libre = depart + len(clip) / SR
    # mots minutés
    ch, st, en = al["characters"], al["character_start_times_seconds"], al["character_end_times_seconds"]
    cur, c0 = "", None
    for c, a0, a1 in list(zip(ch, st, en)) + [(" ", None, None)]:
        if c.isspace():
            if cur:
                if cur in {"?", "!", ":", "%", ";"} and mots: mots[-1]["mot"] += " " + cur  # ponctuation collée au mot
                else: mots.append({"mot": cur, "t0": depart + c0 - d0, "t1": depart + fin - d0, "phrase": i})
                if mots: mots[-1]["t1"] = max(mots[-1]["t1"], depart + fin - d0)
            cur, c0 = "", None
        else:
            if c0 is None: c0 = a0
            cur += c; fin = a1
    print(f"{i+1}. {depart:5.2f}s → {t_libre:5.2f}s  {s['texte'][:60]}")
piste = piste[:int(total * SR)]
piste = piste / max(1e-6, np.abs(piste).max()) * 0.89
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-", "-af", "loudnorm=I=-15:TP=-1.5:LRA=11", "-ar", str(SR), "-ac", "2", f"{W}/voix_fr.wav"], input=piste.tobytes(), check=True)
json.dump(mots, open(f"{W}/mots.json", "w"), ensure_ascii=False, indent=0)
print(len(mots), "mots minutés, durée totale", total, "s")
