import os, uuid, tempfile, traceback, shutil, asyncio
from pathlib import Path
from typing import List
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"], expose_headers=["*"])

OUTPUTS_DIR = Path("/tmp/mixtok")
OUTPUTS_DIR.mkdir(exist_ok=True)

class Req(BaseModel):
    video_urls: List[str]
    style: str = "fun"
    clip_duration: int = 2

STYLES = {
    "drama": {"speed": 1.0, "fade": 0.2},
    "fun":   {"speed": 1.5, "fade": 0.1},
    "hype":  {"speed": 1.3, "fade": 0.1},
}

async def dl(url, dest):
    """Télécharge avec timeout court."""
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as c:
        r = await c.get(url)
        r.raise_for_status()
        dest.write_bytes(r.content)

def process(path, style, clip_dur):
    """Traitement ultra-léger : resize petit + subclip court."""
    clip = VideoFileClip(str(path))
    # Très petite résolution pour rapidité max
    W, H = 360, 640
    ratio = W / H
    if clip.w / clip.h > ratio:
        clip = clip.crop(x_center=clip.w/2, width=int(clip.h*ratio))
    else:
        clip = clip.crop(y_center=clip.h/2, height=int(clip.w/ratio))
    clip = clip.resize((W, H))
    dur = min(clip_dur, clip.duration)
    clip = clip.subclip(0, dur)
    cfg = STYLES.get(style, STYLES["fun"])
    if cfg["speed"] != 1.0:
        clip = clip.fx(vfx.speedx, cfg["speed"])
    return clip.fadein(cfg["fade"]).fadeout(cfg["fade"])

@app.get("/health")
def health(): return {"ok": True}

@app.post("/montage")
async def montage(req: Req):
    if not req.video_urls:
        raise HTTPException(400, "Pas de vidéos")
    job = str(uuid.uuid4())[:8]
    tmp = Path(tempfile.mkdtemp())
    try:
        # Téléchargements en parallèle
        paths = [tmp / f"v{i}.mp4" for i in range(len(req.video_urls[:3]))]
        await asyncio.gather(*[dl(url, dest) for url, dest in zip(req.video_urls[:3], paths)])

        # Traitement séquentiel
        clips = []
        for path in paths:
            try:
                c = process(path, req.style, req.clip_duration)
                clips.append(c)
            except Exception as e:
                print(f"Clip error: {e}")

        if not clips:
            raise Exception("Aucun clip valide")

        final = concatenate_videoclips(clips, method="compose")
        out = OUTPUTS_DIR / f"{job}.mp4"
        final.write_videofile(str(out), fps=24, codec="libx264",
            audio_codec="aac", temp_audiofile=str(tmp/"a.m4a"),
            remove_temp=True, verbose=False, logger=None,
            preset="ultrafast")  # ultrafast = export 3x plus rapide
        final.close()
        return {"status": "done", "url": f"/download/{job}"}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

@app.get("/download/{job}")
def download(job: str):
    p = OUTPUTS_DIR / f"{job}.mp4"
    if not p.exists(): raise HTTPException(404)
    return FileResponse(str(p), media_type="video/mp4",
                        filename=f"mixtok_{job}.mp4")
