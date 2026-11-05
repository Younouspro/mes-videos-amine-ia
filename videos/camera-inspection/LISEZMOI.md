# Caméra d'inspection : version française VUZENA

- `VUZENA_camera_FR.mp4` : la vidéo finale (720×1280, 40,7 s), voix française, sous-titres français, bandeau et carte de fin VUZENA.
- `code/script_fr.json` : le texte de la voix off, phrase par phrase, avec l'heure de départ de chaque phrase.

## Refaire la vidéo après une modification du texte

Placer la vidéo d'origine dans ce dossier sous le nom `original.mp4`, puis lancer depuis ce dossier :

```bash
python3 code/tts.py "$PWD" 1.1
python3 code/audio.py "$PWD"
python3 code/rendu.py "$PWD" video
```

Nécessite : `elevenlabs` (accès API), `opencv-python-headless`, `pillow`, `numpy`, `requests` et ffmpeg.
