import uvicorn
import cv2
import tempfile
import os
import numpy as np
from fastapi import FastAPI, File, UploadFile
from detection import video_detection

app = FastAPI()

@app.get('/')
def root():
    return {'SIZ detection FastAPI': 'Для начала работы перейдите по ссылке http://127.0.0.1:8000/docs'}

@app.post("/video_detection")
async def predict_video(file: UploadFile = File(...), 
                        frame_step: int = 600, 
                        start_time: str = "00:00:00", 
                        stop_time: str = None):
    video_bytes = await file.read()
    tmp_file = tempfile.NamedTemporaryFile(delete=False)
    try:
        tmp_file.write(video_bytes)
        tmp_file.flush()
        p = video_detection(tmp_file.name, frame_step, start_time, stop_time)
        
        if start_time == "00:00:00":
          start_time = 'начала'
        if stop_time == None:
          stop_time = 'конца'
        
        k = f'{file.filename}, от {start_time} до {stop_time} c шагом кадра {frame_step}.'
        return {k: p}
    
    finally:
        tmp_file.close()
        os.unlink(tmp_file.name)
