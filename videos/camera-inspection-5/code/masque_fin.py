# Masque précis : seulement les lettres (et leur fine ombre) ; rectangle pour les mots gris semi-transparents
import cv2, numpy as np
def masque_lettres(img, boites, marge=6, epais=5, hmin=22):
    h, w = img.shape[:2]; m = np.zeros((h, w), np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); v = hsv[..., 2]; s = hsv[..., 1]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * epais + 1, 2 * epais + 1))
    for p in boites:
        p = np.array(p)
        if p[:, 1].max() - p[:, 1].min() < hmin: continue          # petits textes = interface du téléphone
        x0, y0 = np.maximum(p.min(0).astype(int) - marge, 0); x1, y1 = np.minimum(p.max(0).astype(int) + marge, [w - 1, h - 1])
        vv = v[y0:y1, x0:x1]; ss = s[y0:y1, x0:x1]
        th = cv2.morphologyEx(vv, cv2.MORPH_TOPHAT, np.ones((21, 21), np.uint8))
        lettres = (((th > 22) & (ss < 70) & (vv > 85)) | ((vv > 200) & (ss < 45))).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(lettres, 8)
        grand = np.isin(lab, [j for j in range(1, n) if st[j, 3] >= 16]).astype(np.uint8)
        lettres = lettres & cv2.dilate(grand, np.ones((15, 15), np.uint8))   # petits morceaux seulement s'ils touchent une vraie lettre
        if lettres.sum() < 40: continue                              # rien à effacer ici à cette image
        blanc = (vv[lettres > 0] > 215).mean()
        if blanc < 0.55:                                               # mot gris / transparent : on prend tout le cadre
            m[y0:y1, x0:x1] = 255; continue
        lettres = cv2.morphologyEx(lettres, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        m[y0:y1, x0:x1] |= cv2.dilate(lettres, k) * 255
    return m
