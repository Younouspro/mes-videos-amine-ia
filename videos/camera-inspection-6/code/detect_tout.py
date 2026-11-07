import sys, json, cv2
sys.path.insert(0, sys.argv[1] + "/code")
from gomme import boites_texte
cap = cv2.VideoCapture(sys.argv[1] + "/original.mp4"); res = []
while True:
    ok, f = cap.read()
    if not ok: break
    res.append([p.round(1).tolist() for p in boites_texte(f)])
json.dump(res, open(sys.argv[1] + "/boites.json", "w"))
print(len(res), "images analysées")
