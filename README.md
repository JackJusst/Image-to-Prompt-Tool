# Image → Prompt Tool

Ein kleines lokales Windows-Tool, das aus einem Bild einen einstellbar langen Prompt und optional einen Negative Prompt erstellt.

## Funktionen

- Bild hochladen: PNG, JPG/JPEG, WEBP, GIF
- Prompt-Laenge: Kurz, Mittel, Lang, Sehr lang
- Zielformat: Stable Diffusion / SDXL, FLUX, Midjourney, Allgemein
- Look: Amateurfoto, professionelles Foto, dokumentarisch, Studio
- Optionaler fixer Charakter-/Triggerblock
- Zusatzregeln fuer Szene, Kamera, Licht, Pose usw.
- Ganzkoerper-/Distanz-Fix gegen Close-ups
- Optionaler Fingernagel-Fix: natuerlich, kurz, keine kuenstlichen Naegel
- OpenAI API oder lokales Ollama-Vision-Modell

## Installation Windows

1. Ordner entpacken.
2. `install.bat` starten.
3. Danach `run.bat` starten.
4. Browser oeffnet Streamlit lokal.

## OpenAI API nutzen

In Windows CMD:

```bat
setx OPENAI_API_KEY "dein_api_key"
```

Danach CMD/PowerShell neu oeffnen und `run.bat` starten.

## Ollama lokal nutzen

1. Ollama installieren.
2. Ein Vision-Modell installieren.
3. Im Tool `Ollama lokal` auswaehlen und Modellnamen eintragen.

Beispiel:

```bat
ollama pull llava
```

Danach im Tool Modell `llava:latest` verwenden.

## Tipp fuer Fitnessstudio-Prompts

Aktiviere:

- Ganzkoerper staerker erzwingen
- Close-up/Portrait vermeiden

Und schreibe in Zusatzregeln z. B.:

```text
realistic gym training, standing rear three-quarter view using a cable machine, full body visible, camera 5m away, normal gym lighting, amateur smartphone photo
```
