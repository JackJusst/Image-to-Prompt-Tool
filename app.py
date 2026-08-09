import base64
import json
import os
import re
import shutil
import subprocess
import time
from io import BytesIO
from typing import Dict, Tuple
from urllib.parse import urlparse

import requests
import streamlit as st
from PIL import Image

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


APP_TITLE = "Image → Prompt Tool"
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llava:latest")
OLLAMA_START_TIMEOUT = 20

LENGTHS = {
    "Kurz": "45-70 words, compact, only important visual facts",
    "Mittel": "90-140 words, balanced detail, good for SDXL/Flux",
    "Lang": "170-260 words, detailed scene, camera, light, mood, texture",
    "Sehr lang": "320-480 words, very detailed but no filler, highly controlled composition",
}

TARGETS = {
    "Stable Diffusion / SDXL": "comma-separated Stable Diffusion style prompt, practical tokens, avoid full sentences when possible",
    "FLUX / realistisch": "natural but prompt-friendly English, strong realism, scene, camera and lighting terms",
    "Midjourney": "descriptive cinematic prompt, no technical SD weights, no negative prompt syntax inside positive prompt",
    "Allgemein": "clean reusable image generation prompt, descriptive but not overloaded",
}

REALISM_PRESETS = {
    "Natürlich / Amateurfoto": "realistic amateur photo, normal ambient light, candid, slight imperfections, no cinematic look",
    "Professionelles Foto": "professional photo, polished lighting, high detail, clean composition",
    "Roh / Dokumentarisch": "documentary snapshot, unposed, practical lighting, realistic textures, natural imperfections",
    "Studio / Clean": "clean studio look, controlled light, simple background, polished commercial style",
}

BASE_NEGATIVE = (
    "low quality, worst quality, blurry, out of focus, overexposed, underexposed, jpeg artifacts, "
    "bad anatomy, bad hands, extra fingers, missing fingers, deformed fingers, extra limbs, distorted body, "
    "plastic skin, CGI, 3d render, doll-like face, fake skin, unnatural proportions, duplicate person, text, watermark, logo"
)


def image_to_data_url(uploaded_file) -> Tuple[str, str, bytes]:
    image_bytes = uploaded_file.getvalue()
    mime = uploaded_file.type or "image/jpeg"
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{encoded}", mime, image_bytes


def clean_json(text: str) -> Dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # fallback: extract the first JSON object
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return {
        "positive_prompt": text,
        "negative_prompt": BASE_NEGATIVE,
        "short_summary": "Model returned non-JSON text.",
        "detected_elements": [],
        "warnings": ["JSON parsing failed; raw text was used as positive prompt."],
    }


def build_instruction(
    length_label: str,
    target: str,
    realism: str,
    language: str,
    fixed_character: str,
    custom_rules: str,
    full_body_fix: bool,
    avoid_closeup: bool,
    include_negative: bool,
    nail_rule: bool,
) -> str:
    length_rule = LENGTHS[length_label]
    target_rule = TARGETS[target]
    realism_rule = REALISM_PRESETS[realism]

    extra_rules = []
    if full_body_fix:
        extra_rules.append(
            "If a person is visible, strongly control framing: full body visible from head to shoes, wide medium-long shot, subject fills about 35-50% of the image."
        )
    if avoid_closeup:
        extra_rules.append(
            "Avoid close-up or portrait framing; prefer camera distance around 4-6 meters, clear recognizable subject but not cropped."
        )
    if nail_rule:
        extra_rules.append(
            "For visible hands include: natural short fingernails, neatly trimmed, optional subtle colored nail polish, no artificial nails."
        )

    fixed = fixed_character.strip()
    custom = custom_rules.strip()

    return f"""
You are an expert prompt engineer for realistic AI image generation.
Analyze the uploaded image and create a reusable image-generation prompt.

Output language: {language}
Target format: {target_rule}
Length: {length_rule}
Realism preset: {realism_rule}

Rules:
- Describe only visible or strongly inferable visual elements: subject, action, pose, environment, camera distance, lens/perspective, lighting, colors, textures, mood.
- Do not identify real people or guess private identity.
- If a person appears under 18, keep the description neutral and non-sexual.
- No filler, no contradictions, no camera impossibilities.
- Make the prompt useful for recreating the image style and scene, not a plain caption.
- Keep clothing and body descriptions neutral and image-generation focused.
- If fixed character text is provided, prepend and preserve it, then adapt the scene from the image.
- Return valid JSON only.

Fixed character / trigger text to preserve:
{fixed if fixed else "(none)"}

Extra user rules:
{custom if custom else "(none)"}

Additional framing rules:
{chr(10).join(extra_rules) if extra_rules else "(none)"}

JSON schema:
{{
  "positive_prompt": "final positive prompt only",
  "negative_prompt": "negative prompt only" if {str(include_negative).lower()} else "",
  "short_summary": "one short German summary of what the prompt describes",
  "detected_elements": ["important visible elements"],
  "warnings": ["only real issues, otherwise empty list"]
}}
""".strip()


def call_openai(data_url: str, instruction: str, model: str) -> Dict:
    if OpenAI is None:
        raise RuntimeError("OpenAI package is not installed. Run: pip install openai")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Set it in Windows environment variables or .env handling.")

    client = OpenAI()
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": instruction},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
    )
    return clean_json(response.output_text)


def is_local_ollama_host(host: str) -> bool:
    parsed = urlparse(host if "://" in host else f"http://{host}")
    return parsed.hostname in {"localhost", "127.0.0.1", "::1"}


def get_ollama_models(host: str) -> Dict:
    resp = requests.get(host.rstrip("/") + "/api/tags", timeout=3)
    resp.raise_for_status()
    return resp.json()


def ollama_model_is_installed(models: Dict, model: str) -> bool:
    wanted = model.strip()
    wanted_with_tag = wanted if ":" in wanted else f"{wanted}:latest"
    installed = {
        item.get("name") or item.get("model")
        for item in models.get("models", [])
    }
    return wanted in installed or wanted_with_tag in installed


def start_local_ollama(host: str) -> None:
    if not is_local_ollama_host(host):
        raise RuntimeError(
            f"Ollama unter {host} ist nicht erreichbar. Ein entfernter Ollama-Server kann nicht automatisch gestartet werden."
        )

    ollama_exe = shutil.which("ollama")
    if not ollama_exe:
        raise RuntimeError(
            "Ollama ist nicht installiert oder nicht im Windows-PATH. Installiere Ollama einmalig und starte danach dieses Tool erneut."
        )

    popen_options = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        popen_options["creationflags"] = subprocess.CREATE_NO_WINDOW

    subprocess.Popen([ollama_exe, "serve"], **popen_options)

    deadline = time.monotonic() + OLLAMA_START_TIMEOUT
    while time.monotonic() < deadline:
        try:
            get_ollama_models(host)
            return
        except requests.RequestException:
            time.sleep(0.5)

    raise RuntimeError(
        "Ollama wurde gestartet, antwortet aber noch nicht. Warte kurz und versuche es erneut."
    )


def pull_ollama_model(host: str, model: str) -> None:
    resp = requests.post(
        host.rstrip("/") + "/api/pull",
        json={"name": model, "stream": False},
        timeout=(10, 3600),
    )
    resp.raise_for_status()


def ensure_ollama_ready(host: str, model: str) -> Tuple[bool, bool]:
    started = False
    pulled = False

    try:
        models = get_ollama_models(host)
    except requests.RequestException:
        start_local_ollama(host)
        started = True
        models = get_ollama_models(host)

    if not ollama_model_is_installed(models, model):
        if not is_local_ollama_host(host):
            raise RuntimeError(
                f"Das Modell {model} fehlt auf dem entfernten Ollama-Server."
            )
        pull_ollama_model(host, model)
        pulled = True

    return started, pulled


def call_ollama(image_bytes: bytes, instruction: str, model: str, host: str) -> Dict:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    url = host.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": instruction,
        "images": [encoded],
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": 8192,
        },
    }
    resp = requests.post(url, json=payload, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    return clean_json(data.get("response", ""))


def postprocess_negative(negative: str, full_body_fix: bool, avoid_closeup: bool, nail_rule: bool) -> str:
    parts = []
    if negative:
        parts.append(negative.strip())
    parts.append(BASE_NEGATIVE)
    if full_body_fix or avoid_closeup:
        parts.append("close-up, portrait, headshot, upper body only, cropped body, cropped legs, cropped feet, cut off shoes, too close camera, tiny person")
    if nail_rule:
        parts.append("long fingernails, fake nails, acrylic nails, gel nails, nail extensions, claw nails, oversized nails, unnatural nails")
    # de-duplicate comma tokens while preserving order
    seen = set()
    out = []
    for token in ",".join(parts).split(","):
        t = token.strip()
        key = t.lower()
        if t and key not in seen:
            out.append(t)
            seen.add(key)
    return ", ".join(out)


def show_copy_box(label: str, text: str, height: int = 180):
    st.text_area(label, text, height=height)


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🧩", layout="wide")
    st.title(APP_TITLE)
    st.caption("Lokales Prompt-Tool: Bild hochladen → Länge/Format wählen → fertiger Prompt + Negative Prompt.")

    with st.sidebar:
        st.header("Einstellungen")
        provider = st.selectbox("KI-Anbindung", ["Ollama lokal", "OpenAI API"])
        if provider == "OpenAI API":
            model = st.text_input("OpenAI Modell", value=DEFAULT_OPENAI_MODEL)
            st.caption("API-Key wird aus OPENAI_API_KEY gelesen.")
            ollama_host = ""
        else:
            model = st.text_input("Ollama Vision-Modell", value=DEFAULT_OLLAMA_MODEL)
            ollama_host = st.text_input("Ollama Host", value="http://localhost:11434")
            try:
                models = get_ollama_models(ollama_host)
                if ollama_model_is_installed(models, model):
                    st.success("Ollama und Modell sind bereit.")
                else:
                    st.info("Das Modell wird beim ersten Prompt automatisch heruntergeladen.")
            except requests.RequestException:
                st.info("Ollama wird beim ersten Prompt automatisch gestartet.")

        target = st.selectbox("Zielformat", list(TARGETS.keys()), index=0)
        length_label = st.select_slider("Prompt-Länge", options=list(LENGTHS.keys()), value="Mittel")
        realism = st.selectbox("Look", list(REALISM_PRESETS.keys()), index=0)
        language = st.selectbox("Ausgabesprache", ["English", "Deutsch"], index=0)

        st.divider()
        full_body_fix = st.checkbox("Ganzkörper stärker erzwingen", value=True)
        avoid_closeup = st.checkbox("Close-up/Portrait vermeiden", value=True)
        nail_rule = st.checkbox("Natürliche kurze Fingernägel erzwingen", value=False)
        include_negative = st.checkbox("Negative Prompt erzeugen", value=True)

    uploaded = st.file_uploader("Bild hochladen", type=["png", "jpg", "jpeg", "webp", "gif"])

    col_left, col_right = st.columns([0.45, 0.55], gap="large")

    with col_left:
        if uploaded:
            image = Image.open(uploaded)
            st.image(image, caption="Eingabebild", use_container_width=True)
        else:
            st.info("Bild reinziehen oder auswählen.")

        fixed_character = st.text_area(
            "Fixe Personenbeschreibung / Trigger / Outfit (optional)",
            placeholder="z.B. jussman, adult woman, wet braided hair, white tank top...",
            height=120,
        )
        custom_rules = st.text_area(
            "Zusatzregeln (optional)",
            placeholder="z.B. realistisch, keine Studiooptik, normales Fitnessstudio-Licht, Kamera 5m entfernt...",
            height=100,
        )

    with col_right:
        if st.button("Prompt erzeugen", type="primary", disabled=uploaded is None):
            data_url, mime, image_bytes = image_to_data_url(uploaded)
            instruction = build_instruction(
                length_label=length_label,
                target=target,
                realism=realism,
                language=language,
                fixed_character=fixed_character,
                custom_rules=custom_rules,
                full_body_fix=full_body_fix,
                avoid_closeup=avoid_closeup,
                include_negative=include_negative,
                nail_rule=nail_rule,
            )

            with st.spinner("Analysiere Bild und baue Prompt..."):
                try:
                    if provider == "OpenAI API":
                        result = call_openai(data_url, instruction, model)
                    else:
                        started, pulled = ensure_ollama_ready(ollama_host, model)
                        if started:
                            st.toast("Ollama wurde automatisch gestartet.")
                        if pulled:
                            st.toast(f"Ollama-Modell {model} wurde installiert.")
                        result = call_ollama(image_bytes, instruction, model, ollama_host)

                    positive = result.get("positive_prompt", "").strip()
                    negative = result.get("negative_prompt", "").strip()
                    if include_negative:
                        negative = postprocess_negative(negative, full_body_fix, avoid_closeup, nail_rule)

                    st.success("Fertig")
                    show_copy_box("Positive Prompt", positive, height=220)
                    if include_negative:
                        show_copy_box("Negative Prompt", negative, height=180)

                    with st.expander("Analyse / erkannte Elemente"):
                        st.write(result.get("short_summary", ""))
                        st.write(result.get("detected_elements", []))
                        warnings = result.get("warnings", [])
                        if warnings:
                            st.warning("\n".join(warnings))
                except Exception as e:
                    st.error(str(e))

    st.divider()
    st.caption("Tipp: Für dein Gym-Problem 'Ganzkörper stärker erzwingen' + 'Close-up vermeiden' aktiv lassen und im Zusatzfeld konkrete Trainingsaktion eintragen.")


if __name__ == "__main__":
    main()
