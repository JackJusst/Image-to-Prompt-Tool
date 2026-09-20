import csv
import shutil
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parent
ESR = ROOT / "tools" / "realesrgan" / "realesrgan-ncnn-vulkan.exe"
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def metrics(img):
    arr = np.asarray(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


def auto_correct(img, strong=False):
    changed = []
    m = metrics(img)
    if m["brightness"] < 85:
        img = ImageEnhance.Brightness(img).enhance(1.20 if strong else 1.12)
        changed.append("Belichtung+")
    elif m["brightness"] > 205:
        img = ImageEnhance.Brightness(img).enhance(0.90 if strong else 0.95)
        changed.append("Belichtung-")
    if m["contrast"] < 42:
        img = ImageOps.autocontrast(img, cutoff=0.5)
        changed.append("Kontrast")
    m2 = metrics(img)
    if m2["sharpness"] < 100:
        img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=110 if strong else 75, threshold=3))
        changed.append("Schaerfe")
    return img, changed


def remove_red_eye(img):
    rgb = np.asarray(img.convert("RGB")).copy()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.12, 5, minSize=(60, 60))
    changed = False

    for fx, fy, fw, fh in faces:
        top = gray[fy:fy + int(fh * 0.65), fx:fx + fw]
        eyes = eye_cascade.detectMultiScale(top, 1.10, 6, minSize=(15, 10))

        for ex, ey, ew, eh in eyes:
            x = fx + ex
            y = fy + ey
            roi = rgb[y:y + eh, x:x + ew]
            if roi.size == 0:
                continue

            r = roi[:, :, 0].astype(np.int16)
            g = roi[:, :, 1].astype(np.int16)
            b = roi[:, :, 2].astype(np.int16)
            yy, xx = np.ogrid[:eh, :ew]
            cx, cy = ew / 2.0, eh / 2.0
            central = ((xx - cx) ** 2 / (0.48 * ew + 1) ** 2 + (yy - cy) ** 2 / (0.55 * eh + 1) ** 2) <= 1
            mask = (r > 90) & (r > g * 1.35 + 10) & (r > b * 1.35 + 10) & central
            if mask.sum() < 3:
                continue

            replacement = ((g + b) / 2).astype(np.uint8)
            roi[:, :, 0][mask] = replacement[mask]
            changed = True

    return Image.fromarray(rgb), changed


def resize_long_edge(img, target):
    if target <= 0:
        return img
    w, h = img.size
    edge = max(w, h)
    if edge == target:
        return img
    scale = target / edge
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)


def ai_upscale(img, temp):
    if not ESR.exists():
        return img, False

    inp = temp / "input.png"
    out = temp / "output.png"
    img.save(inp)

    try:
        subprocess.run(
            [str(ESR), "-i", str(inp), "-o", str(out), "-n", "realesrgan-x4plus", "-s", "4"],
            cwd=str(ESR.parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=600,
        )
        if out.exists():
            with Image.open(out) as im:
                return im.convert("RGB").copy(), True
    except Exception:
        pass

    return img, False


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Jusst BildTool")
        self.geometry("820x620")
        self.files = []

        ttk.Label(self, text="Jusst BildTool", font=("Segoe UI", 20, "bold")).pack(pady=(12, 4))
        ttk.Label(self, text="Stapelverarbeitung · PNG · Metadaten löschen · Rote Augen · KI").pack()

        buttons = ttk.Frame(self)
        buttons.pack(pady=10)
        ttk.Button(buttons, text="Bilder auswählen", command=self.choose_files).pack(side="left", padx=4)
        ttk.Button(buttons, text="Ordner auswählen", command=self.choose_folder).pack(side="left", padx=4)
        ttk.Button(buttons, text="Liste löschen", command=self.clear).pack(side="left", padx=4)

        self.listbox = tk.Listbox(self, height=14)
        self.listbox.pack(fill="both", expand=True, padx=12)

        frame = ttk.Frame(self)
        frame.pack(fill="x", padx=12, pady=10)

        ttk.Label(frame, text="Profil").grid(row=0, column=0)
        self.profile = tk.StringVar(value="AUTO")
        ttk.Combobox(frame, textvariable=self.profile, values=["NORMAL", "AUTO", "MAX"], state="readonly", width=12).grid(row=0, column=1, padx=5)

        ttk.Label(frame, text="Zielkante").grid(row=0, column=2)
        self.size = tk.StringVar(value="3840")
        ttk.Combobox(frame, textvariable=self.size, values=["0", "1920", "2560", "3000", "3840", "4096"], width=10).grid(row=0, column=3, padx=5)
        ttk.Label(frame, text="px").grid(row=0, column=4)

        self.red = tk.BooleanVar(value=True)
        self.ai = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Rote Augen automatisch", variable=self.red).grid(row=1, column=0, columnspan=2, pady=8)
        ttk.Checkbutton(frame, text="KI-Upscaling wenn nötig", variable=self.ai).grid(row=1, column=2, columnspan=3)

        self.status = ttk.Label(self, text="Bereit")
        self.status.pack(pady=5)

        self.progress = ttk.Progressbar(self)
        self.progress.pack(fill="x", padx=12)

        ttk.Button(self, text="ALLE BILDER VERARBEITEN", command=self.process_all).pack(fill="x", padx=12, pady=12, ipady=8)

    def choose_files(self):
        files = filedialog.askopenfilenames(filetypes=[("Bilder", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp")])
        for f in files:
            self.add(Path(f))

    def choose_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        for f in Path(folder).rglob("*"):
            if f.suffix.lower() in SUPPORTED:
                self.add(f)

    def add(self, p):
        if p not in self.files:
            self.files.append(p)
            self.listbox.insert("end", str(p))

    def clear(self):
        self.files = []
        self.listbox.delete(0, "end")

    def process_all(self):
        if not self.files:
            messagebox.showwarning("Keine Bilder", "Bitte zuerst Bilder auswählen.")
            return

        try:
            target = int(self.size.get())
        except ValueError:
            messagebox.showerror("Fehler", "Zielkante muss eine Zahl sein.")
            return

        output = Path.home() / "Pictures" / "JusstBildTool_Fertig"
        changed = output / "Geaendert"
        temp = output / "_temp"
        output.mkdir(parents=True, exist_ok=True)
        changed.mkdir(exist_ok=True)
        temp.mkdir(exist_ok=True)

        logs = []
        total = len(self.files)
        self.progress["maximum"] = total

        for i, src in enumerate(self.files, 1):
            self.status.config(text=f"{i}/{total}: {src.name}")
            self.update()
            notes = []

            try:
                with Image.open(src) as im:
                    img = ImageOps.exif_transpose(im).convert("RGB")

                if self.red.get() and self.profile.get() != "NORMAL":
                    img, did = remove_red_eye(img)
                    if did:
                        notes.append("Rote Augen")

                if self.profile.get() != "NORMAL":
                    img, auto_notes = auto_correct(img, self.profile.get() == "MAX")
                    notes += auto_notes

                edge = max(img.size)
                need_ai = False
                if self.ai.get() and target > 0:
                    if self.profile.get() == "MAX":
                        need_ai = edge < target
                    elif self.profile.get() == "AUTO":
                        need_ai = edge < target * 0.70

                if need_ai:
                    img, did = ai_upscale(img, temp)
                    if did:
                        notes.append("KI-Upscaling")

                img = resize_long_edge(img, target)

                out = output / (src.stem + ".png")
                count = 1
                while out.exists():
                    out = output / f"{src.stem}_{count}.png"
                    count += 1

                # Frisches PNG ohne übernommene EXIF-/GPS-/IPTC-/XMP-Daten speichern.
                img.save(out, "PNG", optimize=True)

                if notes:
                    shutil.copy2(out, changed / out.name)

                logs.append([str(src), str(out), "; ".join(notes)])
            except Exception as e:
                logs.append([str(src), "FEHLER", str(e)])

            self.progress["value"] = i
            self.update()

        with open(output / "Verarbeitung.csv", "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["Original", "Ausgabe", "Aenderungen"])
            writer.writerows(logs)

        shutil.rmtree(temp, ignore_errors=True)
        self.status.config(text=f"Fertig: {output}")
        messagebox.showinfo("Fertig", f"Verarbeitung abgeschlossen.\n\n{output}")


if __name__ == "__main__":
    App().mainloop()
