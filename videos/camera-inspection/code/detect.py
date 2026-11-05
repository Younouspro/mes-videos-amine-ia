import cv2, numpy as np, json, sys
X0, Y0, X1, Y1 = 120, 815, 680, 1010
def texte_sous(img):
    roi = img[Y0:Y1, X0:X1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    b = ((hsv[..., 2] > 165) & (hsv[..., 1] < 50)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, 8)
    keep = np.zeros_like(b); boxes = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 25 or h < 12 or h > 62 or w > 140: continue
        fill = a / (w * h)
        if fill < 0.15 or fill > 0.9: continue
        keep[lab == i] = 1; boxes.append((x, y, w, h))
    # garder les composantes alignées (ligne de texte = au moins 3 lettres sur la même bande)
    good = np.zeros_like(b); rows = []
    for (x, y, w, h) in boxes:
        cy = y + h / 2
        same = sum(1 for (x2, y2, w2, h2) in boxes if abs((y2 + h2 / 2) - cy) < 12)
        if same >= 3:
            good[y:y + h, x:x + w] |= keep[y:y + h, x:x + w]; rows.append((x, y, x + w, y + h))
    if not rows: return None, None
    r = np.array(rows)
    bb = (int(r[:, 0].min() + X0), int(r[:, 1].min() + Y0), int(r[:, 2].max() + X0), int(r[:, 3].max() + Y0))
    m = np.zeros(img.shape[:2], np.uint8); m[Y0:Y1, X0:X1] = cv2.dilate(good, np.ones((9, 9), np.uint8))
    return bb, m
if __name__ == "__main__":
    cap = cv2.VideoCapture(sys.argv[1]); res = []
    while True:
        ok, f = cap.read()
        if not ok: break
        bb, _ = texte_sous(f); res.append(bb)
    json.dump(res, open(sys.argv[2], "w"))
    v = [b for b in res if b]
    a = np.array(v)
    print(len(res), "images,", len(v), "avec sous-titre détecté")
    for k, nom in enumerate(["x min", "y min", "x max", "y max"]):
        print(f"{nom}: min {a[:,k].min()}  p5 {np.percentile(a[:,k],5):.0f}  médiane {np.median(a[:,k]):.0f}  p95 {np.percentile(a[:,k],95):.0f}  max {a[:,k].max()}")

# petits mots fantômes qui apparaissent sous la bande (bas gauche)
BX0, BY0, BX1, BY1 = 100, 940, 420, 1012
def fantomes(img):
    roi = img[BY0:BY1, BX0:BX1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV); v = hsv[..., 2]
    th = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, np.ones((11, 11), np.uint8))
    b = ((th > 38) & (hsv[..., 1] < 80) & (v > 120)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, 8)
    keep = np.zeros_like(b)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if 6 <= h <= 30 and w <= 40 and a >= 12: keep[lab == i] = 1
    if keep.sum() < 25: return None
    m = np.zeros(img.shape[:2], np.uint8)
    m[BY0:BY1, BX0:BX1] = cv2.dilate(keep, np.ones((7, 7), np.uint8))
    return m
