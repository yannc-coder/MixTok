"""
MixTok Backend — Railway Worker
FastAPI + FFmpeg + MoviePy → MP4 TikTok
"""
import os, uuid, tempfile, asyncio, traceback
from pathlib import Path
from typing import List, Optional

import httpx
import numpy as np
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── MoviePy (v1.x sur Railway) ───────────────────────────
from moviepy.editor import (
    VideoFileClip, concatenate_videoclips,
    AudioFileClip, CompositeVideoClip, vfx
)

# ─── Cloudflare R2 (optionnel) ────────────────────────────
import boto3
R2_ENDPOINT   = os.getenv("R2_ENDPOINT", "")
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY", "")
R2_SECRET_KEY = os.getenv("R2_SECRET_KEY", "")
R2_BUCKET     = os.getenv("R2_BUCKET", "mixtok")
R2_PUBLIC_URL = os.getenv("R2_PUBLIC_URL", "")

# ─── App ──────────────────────────────────────────────────
app = FastAPI(title="MixTok API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Restreindre en prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── State jobs en mémoire (Redis en prod) ────────────────
jobs: dict = {}

# ─── Modèles ──────────────────────────────────────────────
class MontageRequest(BaseModel):
    video_urls: List[str]           # URLs Pexels ou autres
    style: str = "fun"              # drama | fun | hype
    target_duration: int = 30       # secondes
    clip_duration: int = 4          # secondes par extrait
    music: Optional[str] = None     # URL d'une musique (optionnel)

class JobStatus(BaseModel):
    job_id: str
    status: str                     # pending | processing | done | error
    progress: int = 0               # 0-100
    step: str = ""
    result_url: Optional[str] = None
    error: Optional[str] = None

# ─── Style configs ────────────────────────────────────────
STYLES = {
    "drama": {"speed": 1.0,  "color": 1.4, "fade": 0.4},
    "fun":   {"speed": 1.5,  "color": 1.0, "fade": 0.1},
    "hype":  {"speed": 1.25, "color": 1.6, "fade": 0.15},
}

W, H = 1080, 1920  # TikTok 9:16

# ─── Helpers ──────────────────────────────────────────────
def update_job(job_id, **kwargs):
    jobs[job_id].update(kwargs)

async def download_video(url: str, dest: Path):
    """Télécharge une vidéo depuis une URL Pexels."""
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        r = await client.get(url)
        r.raise_for_status()
        dest.write_bytes(r.content)

def crop_916(clip):
    """Centre + recadre en 9:16."""
    ratio = W / H
    if clip.w / clip.h > ratio:
        new_w = int(clip.h * ratio)
        clip = clip.crop(x_center=clip.w/2, width=new_w)
    else:
        new_h = int(clip.w / ratio)
        clip = clip.crop(y_center=clip.h/2, height=new_h)
    return clip.resize((W, H))

def find_best_moment(clip, clip_len=4):
    """Repère le moment avec le plus d'énergie audio."""
    if clip.audio is None or clip.duration <= clip_len:
        return 0.0
    step, best_e, best_t = 0.5, -1, 0.0
    import numpy as np
    for t in np.arange(0, clip.duration - clip_len, step):
        try:
            chunk = clip.audio.subclip(t, min(t + step, clip.duration))
            samples = chunk.to_soundarray()
            e = float(np.sqrt(np.mean(samples**2)))
            if e > best_e:
                best_e, best_t = e, t
        except:
            pass
    return best_t

def apply_style(clip, style_name: str):
    cfg = STYLES.get(style_name, STYLES["fun"])
    if cfg["speed"] != 1.0:
        clip = clip.fx(vfx.speedx, cfg["speed"])
    if cfg["color"] != 1.0:
        clip = clip.fx(vfx.colorx, cfg["color"])
    clip = clip.fadein(cfg["fade"]).fadeout(cfg["fade"])
    return clip

def upload_to_r2(local_path: Path, key: str) -> str:
    """Upload vers Cloudflare R2, retourne l'URL publique."""
    if not R2_ENDPOINT:
        return ""  # Pas de R2 configuré
    s3 = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
    )
    s3.upload_file(str(local_path), R2_BUCKET, key,
                   ExtraArgs={"ContentType": "video/mp4"})
    return f"{R2_PUBLIC_URL}/{key}"

# ─── Worker principal ─────────────────────────────────────
async def run_montage(job_id: str, req: MontageRequest):
    tmpdir = Path(tempfile.mkdtemp())
    try:
        update_job(job_id, status="processing", step="Téléchargement des vidéos", progress=5)

        # 1. Téléchargement
        local_paths = []
        for i, url in enumerate(req.video_urls):
            dest = tmpdir / f"src_{i}.mp4"
            await download_video(url, dest)
            local_paths.append(dest)
            pct = 5 + int((i+1) / len(req.video_urls) * 25)
            update_job(job_id, progress=pct, step=f"Vidéo {i+1}/{len(req.video_urls)} téléchargée")

        update_job(job_id, progress=30, step="Analyse audio & sélection des clips")

        # 2. Extraction des clips
        clips_per_video = max(1, req.target_duration // (len(local_paths) * req.clip_duration))
        all_clips = []

        for i, path in enumerate(local_paths):
            clip = VideoFileClip(str(path))
            clip = crop_916(clip)
            start = find_best_moment(clip, req.clip_duration)
            end = min(start + req.clip_duration, clip.duration)
            sub = clip.subclip(start, end)
            sub = apply_style(sub, req.style)
            all_clips.append(sub)
            pct = 30 + int((i+1) / len(local_paths) * 30)
            update_job(job_id, progress=pct, step=f"Clip {i+1} extrait")

        # Ordre aléatoire pour fun/hype
        if req.style in ["fun", "hype"]:
            import random
            random.shuffle(all_clips)

        update_job(job_id, progress=65, step="Assemblage du montage")

        # 3. Concaténation
        total, final_clips = 0, []
        for c in all_clips:
            if total + c.duration > req.target_duration:
                break
            final_clips.append(c)
            total += c.duration

        final = concatenate_videoclips(final_clips, method="compose")

        update_job(job_id, progress=75, step="Export MP4")

        # 4. Export
        out_path = tmpdir / f"{job_id}.mp4"
        final.write_videofile(
            str(out_path),
            fps=30,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=str(tmpdir / "tmp_audio.m4a"),
            remove_temp=True,
            verbose=False,
            logger=None,
        )
        final.close()

        update_job(job_id, progress=90, step="Upload vers le cloud")

        # 5. Upload R2 ou URL locale
        key = f"montages/{job_id}.mp4"
        result_url = upload_to_r2(out_path, key)
        if not result_url:
            # Fallback : retourne le fichier en local (dev)
            result_url = f"/download/{job_id}"

        update_job(job_id, status="done", progress=100,
                   step="Terminé !", result_url=result_url)

    except Exception as e:
        update_job(job_id, status="error", error=str(e),
                   step=f"Erreur : {e}")
        traceback.print_exc()
    finally:
        # Nettoyage
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

# ─── Routes ───────────────────────────────────────────────
@app.get("/")
def root():
    return {"service": "MixTok API", "status": "ok"}

@app.post("/montage", response_model=JobStatus)
async def create_montage(req: MontageRequest, bg: BackgroundTasks):
    if len(req.video_urls) < 2:
        raise HTTPException(400, "Minimum 2 vidéos requises")
    if len(req.video_urls) > 6:
        raise HTTPException(400, "Maximum 6 vidéos")

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "job_id": job_id, "status": "pending",
        "progress": 0, "step": "En attente...",
        "result_url": None, "error": None,
    }
    bg.add_task(run_montage, job_id, req)
    return jobs[job_id]

@app.get("/status/{job_id}", response_model=JobStatus)
def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job introuvable")
    return jobs[job_id]

@app.get("/health")
def health():
    return {"ok": True}
