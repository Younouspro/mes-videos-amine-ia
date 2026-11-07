# Rendu de la vidéo VUZENA n°2 : sous-titres en bas, fond reconstruit (LaMa) sans flou
import json, sys, subprocess, os
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "video"      # "gomme" | "video" | "apercu"
MODELE = os.environ.get("LAMA", "")
sys.path.insert(0, f"{W}/code")

FPS, LARG, HAUT = 30, 720, 1280
NAVY, BLEU, BLANC = (15, 23, 42), (30, 145, 255), (255, 255, 255)
F_BLACK = "/usr/share/fonts/opentype/inter/Inter-Black.otf"
F_SEMI = "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf"
NB = int(cv2.VideoCapture(f"{W}/original.mp4").get(cv2.CAP_PROP_FRAME_COUNT))
FIN_VIDEO, CARTE = NB / FPS, 3.0
BANDE = (760, 1100)            # lignes de l'image où l'on reconstruit le fond
DOSSIER_GOMME = f"{W}/gomme"
logo = Image.open(f"{W}/marque/logo.png").convert("RGBA")
def police(p, t): return ImageFont.truetype(p, t)
def rect_arrondi(size, box, r, fill, ss=3):
    im = Image.new("L", (size[0] * ss, size[1] * ss), 0)
    ImageDraw.Draw(im).rounded_rectangle([c * ss for c in box], r * ss, fill=fill)
    return im.resize(size, Image.LANCZOS)

# ---------- masques des sous-titres suédois ----------
def masques():
    boites = json.load(open(f"{W}/boites.json"))
    def ok(p):
        p = np.array(p); h = p[:, 1].max() - p[:, 1].min(); w = p[:, 0].max() - p[:, 0].min()
        return 13 <= h <= 80 and w >= 10
    filt = [[np.array(p) for p in b if ok(p)] for b in boites]
    out = []
    for i in range(len(filt)):
        m = np.zeros((HAUT, LARG), np.uint8)
        for j in (i - 1, i, i + 1):          # union avec les images voisines (fondus, apparitions)
            if 0 <= j < len(filt):
                for p in filt[j]:
                    x0, y0 = p.min(0); x1, y1 = p.max(0)
                    cv2.rectangle(m, (int(x0) - 7, int(y0) - 7), (int(x1) + 7, int(y1) + 7), 255, -1)
        out.append(m)
    return out

def gommer_tout(debut, fin):
    from gomme import inpaint
    os.makedirs(DOSSIER_GOMME, exist_ok=True)
    ms = masques(); cap = cv2.VideoCapture(f"{W}/original.mp4"); cap.set(cv2.CAP_PROP_POS_FRAMES, debut)
    for i in range(debut, fin):
        ok, f = cap.read()
        if not ok: break
        p = f"{DOSSIER_GOMME}/{i:05d}.png"
        if os.path.exists(p): continue
        r = inpaint1(f, ms[i]) if ms[i].any() else f
        cv2.imwrite(p, r[BANDE[0]:BANDE[1]])
        if i % 50 == 0: print(f"  gomme {i}/{fin}", flush=True)

def inpaint1(img, m):
    """une seule fenêtre 512 si le texte y tient, sinon deux"""
    from gomme import lama
    out = img.copy(); ys, xs = np.where(m > 0)
    cy = int(np.clip((ys.min() + ys.max()) // 2, 256, HAUT - 256)); y0 = cy - 256
    if xs.max() - xs.min() < 500:
        fen = [int(np.clip((xs.min() + xs.max()) // 2 - 256, 0, LARG - 512))]
    else: fen = [0, LARG - 512]
    for x0 in fen:
        mm = m[y0:y0 + 512, x0:x0 + 512]
        if not mm.any(): continue
        im = cv2.cvtColor(out[y0:y0 + 512, x0:x0 + 512], cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        r = lama(MODELE).run(None, {"image": im.transpose(2, 0, 1)[None], "mask": (mm > 0).astype(np.float32)[None, None]})[0][0].transpose(1, 2, 0)
        if r.max() <= 1.5: r = r * 255
        r = cv2.cvtColor(np.clip(r, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR).astype(np.float32)
        a = cv2.GaussianBlur((mm > 0).astype(np.float32), (0, 0), 1.5)[..., None]
        z = out[y0:y0 + 512, x0:x0 + 512].astype(np.float32)
        out[y0:y0 + 512, x0:x0 + 512] = (r * a + z * (1 - a)).astype(np.uint8)
    return out

# ---------- bandeau titre ----------
BY0 = 96
def faire_bandeau():
    calque = Image.new("RGBA", (LARG, HAUT), (0, 0, 0, 0))
    x0, y0, x1, y1 = 62, BY0, 658, BY0 + 182
    ombre = rect_arrondi((LARG, HAUT), (x0, y0 + 6, x1, y1 + 6), 32, 90).filter(ImageFilter.GaussianBlur(10))
    calque.paste((0, 0, 0, 255), (0, 0), ombre)
    calque.paste((255, 255, 255, 255), (0, 0), rect_arrondi((LARG, HAUT), (x0, y0, x1, y1), 32, 255))
    lg = logo.resize((int(logo.width * 34 / logo.height), 34), Image.LANCZOS)
    calque.alpha_composite(lg, ((LARG - lg.width) // 2, y0 + 20))
    d = ImageDraw.Draw(calque); f = police(F_BLACK, 40)
    lignes = [[("Un cadeau qu'il va", NAVY)], [("vraiment", BLEU), (" utiliser", NAVY)]]
    y = y0 + 72
    for ligne in lignes:
        larg = sum(d.textlength(t, font=f) for t, _ in ligne); x = (LARG - larg) / 2
        for t, c in ligne: d.text((x, y), t, font=f, fill=c); x += d.textlength(t, font=f)
        y += 50
    return calque
BANDEAU = faire_bandeau()
def faire_pastille():
    calque = Image.new("RGBA", (LARG, HAUT), (0, 0, 0, 0))
    lg = logo.resize((int(logo.width * 26 / logo.height), 26), Image.LANCZOS)
    w = lg.width + 44; x0 = (LARG - w) // 2; y0 = 26
    ombre = rect_arrondi((LARG, HAUT), (x0, y0 + 4, x0 + w, y0 + 50), 25, 80).filter(ImageFilter.GaussianBlur(6))
    calque.paste((0, 0, 0, 255), (0, 0), ombre)
    calque.paste((255, 255, 255, 255), (0, 0), rect_arrondi((LARG, HAUT), (x0, y0, x0 + w, y0 + 46), 23, 255))
    calque.alpha_composite(lg, (x0 + 22, y0 + 10))
    return calque
PASTILLE = faire_pastille()
T_TITRE = 999   # titre suédois présent toute la vidéo : bandeau permanent

# ---------- sous-titres français (en bas, sans fond) ----------
mots = json.load(open(f"{W}/mots.json"))
fus = []
for m in mots:
    if fus and fus[-1]["mot"].lower() == "moins" and m["mot"].startswith("50"):
        fus[-1] = {**fus[-1], "mot": "-" + m["mot"], "t1": m["t1"]}
    else: fus.append(dict(m))
blocs, cur = [], []
for i, m in enumerate(fus):
    cur.append(m); txt = " ".join(x["mot"] for x in cur); nxt = fus[i + 1] if i + 1 < len(fus) else None
    if nxt is not None and nxt["mot"].lower().startswith("mètre") and nxt["phrase"] == m["phrase"]: continue
    if (nxt is None or nxt["phrase"] != m["phrase"] or m["mot"][-1] in ",.?!:" or len(cur) >= 3
            or len(txt) + 1 + len(nxt["mot"]) > 20):
        blocs.append(cur); cur = []
for k, b in enumerate(blocs):
    b_t0 = 0.0 if k == 0 else b[0]["t0"]
    b_t1 = blocs[k + 1][0]["t0"] if k + 1 < len(blocs) else FIN_VIDEO
    blocs[k] = {"mots": b, "t0": b_t0, "t1": b_t1}

CY_SOUS, LMAX = 1040, 640
_mes = ImageDraw.Draw(Image.new("L", (1, 1)))
for b in blocs:
    ws = [m["mot"].upper() for m in b["mots"]]; b["ws"] = ws
    taille = 52
    while _mes.textlength(" ".join(ws), font=police(F_BLACK, taille)) > LMAX and taille > 30: taille -= 2
    b["police"] = police(F_BLACK, taille)

def calque_sous(b, t):
    f = b["police"]; ws = b["ws"]; txt = " ".join(ws)
    tw = int(_mes.textlength(txt, font=f)) + 40; th = f.size + 40
    im = Image.new("RGBA", (tw, th), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    # ombre douce
    sh = Image.new("RGBA", (tw, th), (0, 0, 0, 0)); ds = ImageDraw.Draw(sh); x = 20
    for w in ws:
        ds.text((x + 2, 24), w, font=f, fill=(0, 0, 0, 150), stroke_width=7, stroke_fill=(0, 0, 0, 150)); x += _mes.textlength(w + " ", font=f)
    im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(5)))
    x = 20
    for i, w in enumerate(ws):
        m = b["mots"][i]; fin = b["mots"][i + 1]["t0"] if i + 1 < len(ws) else b["t1"]
        actif = m["t0"] - 0.03 <= t < fin
        d.text((x, 20), w, font=f, fill=BLEU if actif else BLANC, stroke_width=5, stroke_fill=(0, 0, 0))
        x += _mes.textlength(w + " ", font=f)
    # petit effet d'apparition
    e = min(1.0, (t - b["t0"]) / 0.12) if b["t0"] > 0 else 1.0
    s = 0.86 + 0.14 * (1 - (1 - e) ** 3)
    if s < 0.999: im = im.resize((max(1, int(tw * s)), max(1, int(th * s))), Image.LANCZOS)
    return im

def habiller(frame, i, t):
    p = f"{DOSSIER_GOMME}/{i:05d}.png"
    if os.path.exists(p): frame[BANDE[0]:BANDE[1]] = cv2.imread(p)
    im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    t_ = i / FPS
    if t_ < T_TITRE: im.alpha_composite(BANDEAU)
    elif t_ < T_TITRE + 0.25:
        a = (t_ - T_TITRE) / 0.25
        im = Image.blend(Image.alpha_composite(im, BANDEAU), Image.alpha_composite(im, PASTILLE), a)
    else: im.alpha_composite(PASTILLE)
    b = next((b for b in blocs if b["t0"] <= t < b["t1"]), None)
    if b is not None:
        c = calque_sous(b, t); im.alpha_composite(c, ((LARG - c.width) // 2, int(CY_SOUS - c.height / 2)))
    return cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)

# ---------- carte de fin ----------
def ease(x): x = min(1, max(0, x)); return 1 - (1 - x) ** 3
def carte_fin(fond_bgr, u, derniere):
    base = Image.fromarray(cv2.cvtColor(cv2.GaussianBlur(fond_bgr, (0, 0), 25), cv2.COLOR_BGR2RGB)).convert("RGBA")
    base.alpha_composite(Image.new("RGBA", (LARG, HAUT), (255, 255, 255, 222)))
    d = ImageDraw.Draw(base)
    def el(k): e = ease((u - 0.12 * k) / 0.45); return e, (1 - e) * 40
    e, dy = el(0)
    lg = logo.resize((480, int(logo.height * 480 / logo.width)), Image.LANCZOS)
    a = lg.copy(); a.putalpha(a.getchannel("A").point(lambda v: int(v * e)))
    base.alpha_composite(a, ((LARG - lg.width) // 2, int(420 + dy)))
    e, dy = el(1); f = police(F_SEMI, 34); s = "Un cadeau qu'il va vraiment utiliser"
    d.text(((LARG - d.textlength(s, font=f)) / 2, 540 + dy), s, font=f, fill=NAVY + (int(255 * e),))
    e, dy = el(2); f = police(F_BLACK, 132); s = "-50 %"
    d.text(((LARG - d.textlength(s, font=f)) / 2, 610 + dy), s, font=f, fill=BLEU + (int(255 * e),))
    e, dy = el(3); f = police(F_BLACK, 44); s = "+ LIVRAISON OFFERTE"
    d.text(((LARG - d.textlength(s, font=f)) / 2, 785 + dy), s, font=f, fill=NAVY + (int(255 * e),))
    e, dy = el(4); f = police(F_BLACK, 44); s = "vuzena.com"; tw = d.textlength(s, font=f)
    bx0, by0 = (LARG - tw) / 2 - 44, 885 + dy
    pill = Image.new("RGBA", (LARG, HAUT), (0, 0, 0, 0))
    pill.paste(NAVY + (255,), (0, 0), rect_arrondi((LARG, HAUT), (bx0, by0, bx0 + tw + 88, by0 + 92), 46, int(255 * e)))
    base.alpha_composite(pill); d = ImageDraw.Draw(base)
    d.text((bx0 + 44, by0 + 18), s, font=f, fill=(255, 255, 255, int(255 * e)))
    out = cv2.cvtColor(np.array(base.convert("RGB")), cv2.COLOR_RGB2BGR)
    if u < 0.3: out = cv2.addWeighted(derniere, 1 - u / 0.3, out, u / 0.3, 0)
    return out

# ---------- programme ----------
if MODE == "gomme":
    gommer_tout(int(sys.argv[3]), int(sys.argv[4])); sys.exit()

cap = cv2.VideoCapture(f"{W}/original.mp4")
if MODE == "apercu":
    imgs = []
    for t in [float(x) for x in sys.argv[3].split(",")]:
        i = int(round(t * FPS))
        if i < NB:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read()
            if not os.path.exists(f"{DOSSIER_GOMME}/{i:05d}.png"):
                ms = masques(); g = inpaint1(f, ms[i]) if ms[i].any() else f
                os.makedirs(DOSSIER_GOMME, exist_ok=True); cv2.imwrite(f"{DOSSIER_GOMME}/{i:05d}.png", g[BANDE[0]:BANDE[1]])
            imgs.append(habiller(f, i, t))
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, NB - 3); ok, f = cap.read()
            imgs.append(carte_fin(f, t - FIN_VIDEO, habiller(f.copy(), NB - 3, FIN_VIDEO - 0.1)))
    cv2.imwrite(f"{W}/test/apercu.jpg", cv2.resize(np.hstack(imgs), None, fx=0.5, fy=0.5), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("blocs:", " | ".join(" ".join(b["ws"]) for b in blocs)); sys.exit()

enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{LARG}x{HAUT}", "-r", str(FPS), "-i", "-",
                        "-i", f"{W}/voix_fr.wav", "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", f"{W}/VUZENA_camera5_FR.mp4"], stdin=subprocess.PIPE)
n, brut, dern = 0, None, None
while True:
    ok, f = cap.read()
    if not ok: break
    brut = f.copy(); dern = habiller(f, n, n / FPS); enc.stdin.write(dern.tobytes()); n += 1
nfin = int(round((FIN_VIDEO + CARTE) * FPS)) - n
for k in range(nfin): enc.stdin.write(carte_fin(brut, k / FPS, dern).tobytes())
enc.stdin.close(); enc.wait(); print("terminé :", n, "+", nfin, "images")
