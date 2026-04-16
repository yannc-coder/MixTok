import os, uuid, tempfile, traceback, shutil
from pathlib import Path
from typing import List, Optional

import httpx
from fastapi import FastAPI, BackgroundTasks, HTTPException
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

jobs = {}
OUTPUTS_DIR = Path("/tmp/mixtok_outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

class MontageRequest(BaseModel):
    video_urls: List[str]
    style: str = "fun"
    target_duration: int = 30
    clip_duration: int = 4

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: int = 0
    step: str = ""
    result_url: Optional[str] = None
    error: Optional[str] = None

STYLES = {
    "drama": {"speed": 1.0,  "color": 1.4, "fade": 0.4},
    "fun":   {"speed": 1.5,  "color": 1.0, "fade": 0.1},
    "hype":  {"speed": 1.25, "color": 1.6, "fade": 0.15},
}

W, H = 720, 1280

def upd(job_id, **kw):
    jobs[job_id].update(kw)

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

async def run_montage(job_id, req):
    tmpdir = Path(tempfile.mkdtemp())
    try:
        upd(job_id, status="processing", step="Téléchargement des vidéos", progress=5)
        paths = []
        for i, url in enumerate(req.video_urls):
            dest = tmpdir / f"v{i}.mp4"
            await download_video(url, dest)
            paths.append(dest)
            upd(job_id, progress=5 + int((i+1)/len(req.video_urls)*30),
                step=f"Vidéo {i+1}/{len(req.video_urls)} téléchargée")

        upd(job_id, progress=40, step="Analyse et découpage des clips")
        clips = []
        for i, path in enumerate(paths):
            try:
                clip = VideoFileClip(str(path))
                clip = crop_916(clip)
                start = best_start(clip, req.clip_duration)
                end = min(start + req.clip_duration, clip.duration)
                sub = clip.subclip(start, end)
                sub = apply_style(sub, req.style)
                clips.append(sub)
                upd(job_id, progress=40 + int((i+1)/len(paths)*25),
                    step=f"Clip {i+1} prêt")
            except Exception as e:
                print(f"Erreur clip {i}: {e}")
                continue

        if not clips:
            raise Exception("Aucun clip valide extrait")

        if req.style in ["fun", "hype"]:
            import random
            random.shuffle(clips)

        upd(job_id, progress=70, step="Assemblage du montage")
        total, final = 0, []
        for c in clips:
            if total + c.duration > req.target_duration:
                break
            final.append(c)
            total += c.duration

        video = concatenate_videoclips(final, method="compose")
        upd(job_id, progress=82, step="Export MP4...")

        out = OUTPUTS_DIR / f"{job_id}.mp4"
        video.write_videofile(
            str(out), fps=24, codec="libx264",
            audio_codec="aac",
            temp_audiofile=str(tmpdir / "tmp.m4a"),
            remove_temp=True, verbose=False, logger=None,
        )
        video.close()

        upd(job_id, status="done", progress=100,
            step="Terminé !", result_url=f"/download/{job_id}")

    except Exception as e:
        upd(job_id, status="error", error=str(e))
        traceback.print_exc()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

@app.get("/")
def root():
    return {"service": "MixTok API", "status": "ok"}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/montage", response_model=JobStatus)
async def create_montage(req: MontageRequest, bg: BackgroundTasks):
    if len(req.video_urls) < 2:
        raise HTTPException(400, "Minimum 2 vidéos")
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "job_id": job_id, "status": "pending",
        "progress": 0, "step": "En attente...",
        "result_url": None, "error": None,
    }
    bg.add_task(run_montage, job_id, req)
    return jobs[job_id]

@app.get("/status/{job_id}", response_model=JobStatus)
def get_status(job_id):
    if job_id not in jobs:
        raise HTTPException(404, "Job introuvable")
    return jobs[job_id]

@app.get("/download/{job_id}")
def download(job_id):
    path = OUTPUTS_DIR / f"{job_id}.mp4"
    if not path.exists():
        raise HTTPException(404, "Fichier introuvable")
    return FileResponse(str(path), media_type="video/mp4",
                        filename=f"mixtok_{job_id}.mp4")
