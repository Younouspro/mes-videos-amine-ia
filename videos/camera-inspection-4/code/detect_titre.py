import sys, json, cv2
from rapidocr_onnxruntime import RapidOCR
o = RapidOCR(); cap = cv2.VideoCapture(sys.argv[1] + "/original.mp4"); res = []; i = 0
while True:
    ok, f = cap.read()
    if not ok: break
    if i % 3 == 0:
        d, _ = o.text_detector(f[150:290, 40:680]); res.append((i, 0 if d is None else len(d)))
    i += 1
json.dump(res, open(sys.argv[1] + "/titre.json", "w")); print("titre ok")
