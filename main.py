import os, uuid, tempfile, traceback, shutil
from pathlib import Path
from typing import List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx

app = FastAPI(title="MixTok API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

OUTPUTS_DIR = Path("/tmp/mixtok_outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

class MontageRequest(BaseModel):
    video_urls: List[str]
    style: str = "fun"
    target_duration: int = 30
    clip_duration: int = 4

STYLES = {
    "drama": {"speed": 1.0,  "color": 1.4, "fade": 0.4},
    "fun":   {"speed": 1.5,  "color": 1.0, "fade": 0.1},
    "hype":  {"speed": 1.25, "color": 1.6, "fade": 0.15},
}

W, H = 720, 1280

async def download_video(url, dest):
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        r = await client.get(url)
        r.raise_for_status()
        dest.write_bytes(r.content)

def crop_916(clip):
    ratio = W / H
    if clip.w / clip.h > ratio:
        clip = clip.crop(x_center=clip.w/2, width=int(clip.h * ratio))
    else:
        clip = clip.crop(y_center=clip.h/2, height=int(clip.w / ratio))
    return clip.resize((W, H))

def best_start(clip, clip_len):
    if clip.audio is None or clip.duration <= clip_len:
        return 0.0
    import numpy as np
    best_e, best_t = -1, 0.0
    for t in range(0, int(clip.duration - clip_len)):
        try:
            s = clip.audio.subclip(t, t+1).to_soundarray()
            e = float(np.sqrt(np.mean(s**2)))
            if e > best_e:
                best_e, best_t = e, float(t)
        except:
            pass
    return best_t

def apply_style(clip, style_name):
    cfg = STYLES.get(style_name, STYLES["fun"])
    if cfg["speed"] != 1.0:
        clip = clip.fx(vfx.speedx, cfg["speed"])
    if cfg["color"] != 1.0:
        clip = clip.fx(vfx.colorx, cfg["color"])
    return clip.fadein(cfg["fade"]).fadeout(cfg["fade"])

@app.get("/")
def root():
    return {"service": "MixTok API", "status": "ok"}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/montage")
async def create_montage(req: MontageRequest):
    """Montage synchrone — attend le résultat et retourne l'URL directement."""
    if len(req.video_urls) < 2:
        raise HTTPException(400, "Minimum 2 vidéos")

    job_id = str(uuid.uuid4())[:8]
    tmpdir = Path(tempfile.mkdtemp())

    try:
        # 1. Téléchargement
        paths = []
        for i, url in enumerate(req.video_urls):
            dest = tmpdir / f"v{i}.mp4"
            await download_video(url, dest)
            paths.append(dest)

        # 2. Extraction + style
        clips = []
        for path in paths:
            try:
                clip = VideoFileClip(str(path))
                clip = crop_916(clip)
                start = best_start(clip, req.clip_duration)
                end = min(start + req.clip_duration, clip.duration)
                sub = clip.subclip(start, end)
                sub = apply_style(sub, req.style)
                clips.append(sub)
            except Exception as e:
                print(f"Erreur clip: {e}")
                continue

        if not clips:
            raise Exception("Aucun clip valide")

        if req.style in ["fun", "hype"]:
            import random
            random.shuffle(clips)

        # 3. Concat
        total, final = 0, []
        for c in clips:
            if total + c.duration > req.target_duration:
                break
            final.append(c)
            total += c.duration

        video = concatenate_videoclips(final, method="compose")

        # 4. Export
        out = OUTPUTS_DIR / f"{job_id}.mp4"
        video.write_videofile(
            str(out), fps=24, codec="libx264",
            audio_codec="aac",
            temp_audiofile=str(tmpdir / "tmp.m4a"),
            remove_temp=True, verbose=False, logger=None,
        )
        video.close()

        return {
            "job_id": job_id,
            "status": "done",
            "progress": 100,
            "result_url": f"https://mixtok.onrender.com/download/{job_id}"
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

@app.get("/download/{job_id}")
def download(job_id: str):
    path = OUTPUTS_DIR / f"{job_id}.mp4"
    if not path.exists():
        raise HTTPException(404, "Fichier introuvable")
    return FileResponse(str(path), media_type="video/mp4",
                        filename=f"mixtok_{job_id}.mp4")
