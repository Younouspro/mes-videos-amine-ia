# Détection des sous-titres suédois + reconstruction du fond (LaMa)
import cv2, numpy as np, onnxruntime as ort
from rapidocr_onnxruntime import RapidOCR
ZX0, ZY0, ZX1, ZY1 = 30, 780, 700, 1070          # zone où vivent les sous-titres d'origine
_ocr = RapidOCR()
_lama = None
def lama(path):
    global _lama
    if _lama is None:
        so = ort.SessionOptions(); import os; so.intra_op_num_threads = int(os.environ.get("LAMA_THREADS", "4"))
        _lama = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    return _lama

def boites_texte(img):
    roi = img[ZY0:ZY1, ZX0:ZX1]
    dets, _ = _ocr.text_detector(roi)
    out = []
    if dets is None: return out
    for p in dets:
        p = np.array(p, np.float32); p[:, 0] += ZX0; p[:, 1] += ZY0
        out.append(p)
    return out

def masque(img, boites, marge=7):
    m = np.zeros(img.shape[:2], np.uint8)
    for p in boites:
        x0, y0 = p.min(0); x1, y1 = p.max(0)
        cv2.rectangle(m, (int(x0) - marge, int(y0) - marge), (int(x1) + marge, int(y1) + marge), 255, -1)
    return m

def inpaint(img, m, modele):
    """LaMa sur deux carrés 512 qui se chevauchent, à pleine résolution"""
    if not m.any(): return img
    out = img.copy()
    ys, xs = np.where(m > 0)
    cy = int(np.clip((ys.min() + ys.max()) // 2, 256, img.shape[0] - 256))
    y0 = cy - 256
    for x0 in (0, img.shape[1] - 512):
        mm = m[y0:y0 + 512, x0:x0 + 512]
        if not mm.any(): continue
        im = cv2.cvtColor(out[y0:y0 + 512, x0:x0 + 512], cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        r = lama(modele).run(None, {"image": im.transpose(2, 0, 1)[None], "mask": (mm > 0).astype(np.float32)[None, None]})[0][0]
        r = r.transpose(1, 2, 0)
        if r.max() <= 1.5: r = r * 255
        r = cv2.cvtColor(np.clip(r, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        # fondu doux sur le bord du masque
        a = cv2.GaussianBlur((mm > 0).astype(np.float32), (0, 0), 1.5)[..., None]
        zone = out[y0:y0 + 512, x0:x0 + 512].astype(np.float32)
        out[y0:y0 + 512, x0:x0 + 512] = (r * a + zone * (1 - a)).astype(np.uint8)
    return out
