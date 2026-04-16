import os, uuid, tempfile, traceback, shutil, asyncio
from pathlib import Path
from typing import List
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx

app = FastAPI()

# CORS très permissif
@app.middleware("http")
async def add_cors(request: Request, call_next):
    if request.method == "OPTIONS":
        return JSONResponse({}, headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "86400",
        })
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

app.add_middleware(CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"], expose_headers=["*"])

OUTPUTS_DIR = Path("/tmp/mixtok")
OUTPUTS_DIR.mkdir(exist_ok=True)
jobs = {}

class Req(BaseModel):
    video_urls: List[str]
    style: str = "fun"

STYLES = {
    "drama": {"speed": 1.0, "fade": 0.2},
    "fun":   {"speed": 1.5, "fade": 0.1},
    "hype":  {"speed": 1.3, "fade": 0.1},
}

async def dl(url, dest):
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as c:
        r = await c.get(url)
        r.raise_for_status()
        dest.write_bytes(r.content)

def process(path, style):
    clip = VideoFileClip(str(path))
    W, H = 360, 640
    ratio = W / H
    if clip.w / clip.h > ratio:
        clip = clip.crop(x_center=clip.w/2, width=int(clip.h*ratio))
    else:
        clip = clip.crop(y_center=clip.h/2, height=int(clip.w/ratio))
    clip = clip.resize((W, H))
    clip = clip.subclip(0, min(2.5, clip.duration))
    cfg = STYLES.get(style, STYLES["fun"])
    if cfg["speed"] != 1.0:
        clip = clip.fx(vfx.speedx, cfg["speed"])
    return clip.fadein(cfg["fade"]).fadeout(cfg["fade"])

async def run_job(job_id, urls, style):
    tmp = Path(tempfile.mkdtemp())
    try:
        jobs[job_id] = {"status": "processing", "progress": 10}
        paths = [tmp / f"v{i}.mp4" for i in range(len(urls))]
        await asyncio.gather(*[dl(u, p) for u, p in zip(urls, paths)])
        jobs[job_id]["progress"] = 50
        clips = []
        for path in paths:
            try:
                clips.append(process(path, style))
            except: pass
        if not clips: raise Exception("Aucun clip valide")
        jobs[job_id]["progress"] = 70
        final = concatenate_videoclips(clips, method="compose")
        out = OUTPUTS_DIR / f"{job_id}.mp4"
        final.write_videofile(str(out), fps=24, codec="libx264",
            audio_codec="aac", temp_audiofile=str(tmp/"a.m4a"),
            remove_temp=True, verbose=False, logger=None, preset="ultrafast")
        final.close()
        jobs[job_id] = {"status": "done", "progress": 100,
                        "url": f"https://mixtok.onrender.com/download/{job_id}"}
    except Exception as e:
        jobs[job_id] = {"status": "error", "error": str(e)}
        traceback.print_exc()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

@app.get("/health")
def health(): return {"ok": True}

@app.post("/montage")
async def montage(req: Req):
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "pending", "progress": 0}
    asyncio.create_task(run_job(job_id, req.video_urls[:3], req.style))
    return {"job_id": job_id, "status": "pending"}

@app.get("/status/{job_id}")
def status(job_id: str):
    if job_id not in jobs: raise HTTPException(404)
    return jobs[job_id]

@app.get("/download/{job_id}")
def download(job_id: str):
    p = OUTPUTS_DIR / f"{job_id}.mp4"
    if not p.exists(): raise HTTPException(404)
    return FileResponse(str(p), media_type="video/mp4",
                        filename=f"mixtok_{job_id}.mp4")
