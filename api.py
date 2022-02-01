import logging
import time
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from io import BytesIO
from scipy.io.wavfile import read, write
from src.model import PoetFormer

name = 'GPT2-fa-ganjoor-conditional'
model = PoetFormer(pretrained_name="HooshvareLab/gpt2-fa")
# model = PoetFormer.load_from_checkpoint(f'weights/{name}/last.ckpt', pretrained="HooshvareLab/gpt2-fa")
model.eval()


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def index():
    return "Welcom to ZLab Poem API. Please use /docs route to get started."

@app.get("/generate/")
def serve(poet : str = 'حافظ', prompt : str = ''):
    start = time.time()
    text = model.generate(
        prompt,
        poet, 
        max_length=48, 
        num_return_sequences=1, 
        topk=100, 
        top_p=0.9, 
        n_beam=10, 
        no_repeat_ngram=3,
        temperature=0.8
    )[0]
    latency = time.time() - start
    return {'text' : text, 'latency' : latency}


## running the web server with ssl (https)
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=5000)