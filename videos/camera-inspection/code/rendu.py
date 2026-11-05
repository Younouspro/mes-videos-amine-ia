# Rendu de la vidéo VUZENA en français
import json, sys, subprocess, math
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "video"      # "video" ou "apercu"
sys.path.insert(0, f"{W}/code")
from detect import texte_sous

FPS, LARG, HAUT = 30, 720, 1280
NAVY, BLEU, BLANC = (15, 23, 42), (30, 145, 255), (255, 255, 255)
F_BLACK = "/usr/share/fonts/opentype/inter/Inter-Black.otf"
F_XBOLD = "/usr/share/fonts/opentype/inter/Inter-ExtraBold.otf"
F_SEMI = "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf"
FIN_VIDEO, CARTE = 37.68, 3.0
logo = Image.open(f"{W}/marque/logo.png").convert("RGBA")

def police(p, t): return ImageFont.truetype(p, t)

def rect_arrondi(size, box, r, fill, ss=3):
    """masque/rectangle arrondi anti-aliasé"""
    im = Image.new("L", (size[0] * ss, size[1] * ss), 0)
    ImageDraw.Draw(im).rounded_rectangle([c * ss for c in box], r * ss, fill=fill)
    return im.resize(size, Image.LANCZOS)

# ---------- bandeau titre (statique) ----------
def faire_bandeau():
    calque = Image.new("RGBA", (LARG, HAUT), (0, 0, 0, 0))
    x0, y0, x1, y1 = 62, 96, 658, 278
    ombre = rect_arrondi((LARG, HAUT), (x0, y0 + 6, x1, y1 + 6), 32, 90).filter(ImageFilter.GaussianBlur(10))
    calque.paste((0, 0, 0, 255), (0, 0), ombre)
    fond = rect_arrondi((LARG, HAUT), (x0, y0, x1, y1), 32, 255)
    calque.paste((255, 255, 255, 255), (0, 0), fond)
    lg = logo.resize((int(logo.width * 34 / logo.height), 34), Image.LANCZOS)
    calque.alpha_composite(lg, ((LARG - lg.width) // 2, y0 + 20))
    d = ImageDraw.Draw(calque); f = police(F_BLACK, 42)
    lignes = [[("Un cadeau qu'il va", NAVY)], [("vraiment", BLEU), (" utiliser", NAVY)]]
    y = y0 + 70
    for ligne in lignes:
        larg = sum(d.textlength(t, font=f) for t, _ in ligne); x = (LARG - larg) / 2
        for t, c in ligne:
            d.text((x, y), t, font=f, fill=c); x += d.textlength(t, font=f)
        y += 50
    return calque
BANDEAU = faire_bandeau()

# ---------- sous-titres : découpage en blocs ----------
mots = json.load(open(f"{W}/mots.json"))
fus = []
for m in mots:  # « moins 50 % » -> « -50 % »
    if fus and fus[-1]["mot"].lower() == "moins" and m["mot"].startswith("50"):
        fus[-1] = {**fus[-1], "mot": "-" + m["mot"], "t1": m["t1"]}
    else: fus.append(dict(m))
blocs, cur = [], []
for i, m in enumerate(fus):
    cur.append(m)
    txt = " ".join(x["mot"] for x in cur)
    nxt = fus[i + 1] if i + 1 < len(fus) else None
    coupe = (nxt is None or nxt["phrase"] != m["phrase"] or m["mot"][-1] in ",.?!:"
             or len(cur) >= 3 or len(txt) + 1 + len(nxt["mot"]) > 18)
    if coupe: blocs.append(cur); cur = []
for k, b in enumerate(blocs):
    b_t0 = 0.0 if k == 0 else b[0]["t0"]
    b_t1 = blocs[k + 1][0]["t0"] if k + 1 < len(blocs) else FIN_VIDEO
    for m in b: m["b"] = k
    blocs[k] = {"mots": b, "t0": b_t0, "t1": b_t1}

F_SOUS = police(F_BLACK, 50)
CY = 916
_mes = ImageDraw.Draw(Image.new("L", (1, 1)))
def lignes_bloc(b):
    ws = [m["mot"].upper() for m in b["mots"]]
    tout = " ".join(ws)
    if _mes.textlength(tout, font=F_SOUS) <= 470 or len(ws) == 1: return [list(range(len(ws)))]
    best = None
    for c in range(1, len(ws)):
        a, z = " ".join(ws[:c]), " ".join(ws[c:])
        s = max(_mes.textlength(a, font=F_SOUS), _mes.textlength(z, font=F_SOUS))
        if best is None or s < best[0]: best = (s, c)
    return [list(range(best[1])), list(range(best[1], len(ws)))]
for b in blocs:
    b["lignes"] = lignes_bloc(b)
    ws = [m["mot"].upper() for m in b["mots"]]
    lw = max(_mes.textlength(" ".join(ws[i] for i in l), font=F_SOUS) for l in b["lignes"])
    b["taille"] = 50 if lw <= 470 else int(50 * 470 / lw)
    b["police"] = police(F_BLACK, b["taille"]); lw = min(lw, 470)
    w = max(528, lw + 56); h = max(170, len(b["lignes"]) * 58 + 40)
    b["boite"] = (int(360 - w / 2), int(CY - h / 2), int(360 + w / 2), int(CY + h / 2))

def bloc_a(t):
    for b in blocs:
        if b["t0"] <= t < b["t1"]: return b
    return None

def sous_titre(frame, t):
    b = bloc_a(t)
    if b is None: return frame
    x0, y0, x1, y1 = b["boite"]
    x0, x1 = max(0, x0), min(LARG, x1)
    # 1) gommer le texte suédois dans la boîte, 2) verre dépoli
    _, m = texte_sous(frame)
    zone = frame[y0:y1, x0:x1].copy()
    if m is not None and m[y0:y1, x0:x1].any():
        zone = cv2.inpaint(zone, m[y0:y1, x0:x1], 6, cv2.INPAINT_TELEA)
    flou = cv2.GaussianBlur(zone, (0, 0), 16)
    teinte = np.full_like(flou, NAVY[::-1])
    verre = cv2.addWeighted(flou, 0.5, teinte, 0.5, 0)
    msk = np.array(rect_arrondi((x1 - x0, y1 - y0), (0, 0, x1 - x0, y1 - y0), 26, 255))[..., None] / 255.0
    frame[y0:y1, x0:x1] = (verre * msk + frame[y0:y1, x0:x1] * (1 - msk)).astype(np.uint8)
    # 3) texte français, mot prononcé en bleu
    im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(im)
    ws = [m["mot"].upper() for m in b["mots"]]
    FS = b["police"]; dec = (50 - b["taille"]) * 0.55
    hl = len(b["lignes"]) * 58; y = CY - hl / 2 - 6 + dec
    for l in b["lignes"]:
        txt = " ".join(ws[i] for i in l); x = 360 - d.textlength(txt, font=FS) / 2
        for i in l:
            mo = b["mots"][i]; actif = mo["t0"] - 0.03 <= t < (b["mots"][i + 1]["t0"] if i + 1 < len(b["mots"]) else b["t1"])
            d.text((x + 2, y + 3), ws[i], font=FS, fill=(0, 0, 0))
            d.text((x, y), ws[i], font=FS, fill=BLEU if actif else BLANC)
            x += d.textlength(ws[i] + " ", font=FS)
        y += 58
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

def habiller(frame, t):
    frame = sous_titre(frame, t)
    im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    im.alpha_composite(BANDEAU)
    return cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)

# ---------- carte de fin ----------
def ease(x): x = min(1, max(0, x)); return 1 - (1 - x) ** 3
def carte_fin(fond_bgr, u, derniere):
    """u = secondes depuis le début de la carte"""
    base = Image.fromarray(cv2.cvtColor(cv2.GaussianBlur(fond_bgr, (0, 0), 25), cv2.COLOR_BGR2RGB)).convert("RGBA")
    base.alpha_composite(Image.new("RGBA", (LARG, HAUT), (255, 255, 255, 222)))
    d = ImageDraw.Draw(base)
    def el(k):  # apparition décalée
        e = ease((u - 0.12 * k) / 0.45); return e, (1 - e) * 40
    e, dy = el(0)
    lg = logo.resize((480, int(logo.height * 480 / logo.width)), Image.LANCZOS)
    a = lg.copy(); a.putalpha(a.getchannel("A").point(lambda v: int(v * e)))
    base.alpha_composite(a, ((LARG - lg.width) // 2, int(420 + dy)))
    e, dy = el(1)
    f = police(F_SEMI, 34); s = "Un cadeau qu'il va vraiment utiliser"
    d.text(((LARG - d.textlength(s, font=f)) / 2, 540 + dy), s, font=f, fill=NAVY + (int(255 * e),))
    e, dy = el(2)
    f = police(F_BLACK, 132); s = "-50 %"
    d.text(((LARG - d.textlength(s, font=f)) / 2, 610 + dy), s, font=f, fill=BLEU + (int(255 * e),))
    e, dy = el(3)
    f = police(F_BLACK, 44); s = "+ LIVRAISON OFFERTE"
    d.text(((LARG - d.textlength(s, font=f)) / 2, 785 + dy), s, font=f, fill=NAVY + (int(255 * e),))
    e, dy = el(4)
    f = police(F_BLACK, 44); s = "vuzena.com"; tw = d.textlength(s, font=f)
    bx0, by0 = (LARG - tw) / 2 - 44, 885 + dy
    pill = Image.new("RGBA", (LARG, HAUT), (0, 0, 0, 0))
    pm = rect_arrondi((LARG, HAUT), (bx0, by0, bx0 + tw + 88, by0 + 92), 46, int(255 * e))
    pill.paste(NAVY + (255,), (0, 0), pm); base.alpha_composite(pill)
    d = ImageDraw.Draw(base); d.text((bx0 + 44, by0 + 18), s, font=f, fill=(255, 255, 255, int(255 * e)))
    out = cv2.cvtColor(np.array(base.convert("RGB")), cv2.COLOR_RGB2BGR)
    if u < 0.3:  # fondu depuis la dernière image
        out = cv2.addWeighted(derniere, 1 - u / 0.3, out, u / 0.3, 0)
    return out

# ---------- programme ----------
cap = cv2.VideoCapture(f"{W}/original.mp4")
if MODE == "apercu":
    temps = [float(x) for x in sys.argv[3].split(",")]; imgs = []
    for t in temps:
        if t < FIN_VIDEO:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, f = cap.read(); imgs.append(habiller(f, t))
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 3); ok, f = cap.read()
            imgs.append(carte_fin(f, t - FIN_VIDEO, habiller(f.copy(), FIN_VIDEO - 0.05)))
    g = np.hstack(imgs); cv2.imwrite(f"{W}/test/apercu.jpg", cv2.resize(g, None, fx=0.5, fy=0.5), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("blocs:", " | ".join(" ".join(m["mot"] for m in b["mots"]) for b in blocs))
    sys.exit()

enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{LARG}x{HAUT}", "-r", str(FPS), "-i", "-",
                        "-i", f"{W}/voix_fr.wav", "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", f"{W}/VUZENA_camera_FR.mp4"], stdin=subprocess.PIPE)
n, brut, dern = 0, None, None
while True:
    ok, f = cap.read()
    if not ok: break
    t = n / FPS; brut = f.copy(); dern = habiller(f, t); enc.stdin.write(dern.tobytes()); n += 1
    if n % 150 == 0: print(f"  {n} images…", flush=True)
nfin = int(round((FIN_VIDEO + CARTE) * FPS)) - n
for k in range(nfin):
    enc.stdin.write(carte_fin(brut, k / FPS, dern).tobytes())
enc.stdin.close(); enc.wait()
print("terminé :", n, "+", nfin, "images")
