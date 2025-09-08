# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.concurrency import run_in_threadpool
import google.generativeai as genai
import os
from fastapi import File, UploadFile
from fastapi.responses import JSONResponse
import base64

# 1) Load environment variables from .env
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

if not API_KEY:
    # Fail fast if key missing — saves head-scratching later
    raise RuntimeError(
        "GEMINI_API_KEY not found. Put it in .env like: GEMINI_API_KEY=your_key_here"
    )

# 2) Configure Gemini SDK with your key
genai.configure(api_key=API_KEY)

# 3) Pydantic model: defines the JSON shape we accept from the client
class PromptIn(BaseModel):
    prompt: str

# 4) Optional output schema (handy if you later add usage, latency, etc.)
class OutputOut(BaseModel):
    output: str

# 5) Create FastAPI app
app = FastAPI(title="Gemini Relay API", version="1.0")

# 6) CORS middleware — lets your Flutter app call this server in dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # in production, list specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 7) Health check endpoint — easy ping for “is server alive?”
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}

# 8) Main endpoint: accepts a prompt, forwards to Gemini, returns text

@app.post("/generate", response_model=OutputOut)
async def generate(body: PromptIn = None, file: UploadFile = File(None)):
    prompt = (body.prompt or "").strip() if body else ""

    if not prompt and not file:
        raise HTTPException(status_code=400, detail="`prompt` or image required.")

    try:
        model = genai.GenerativeModel(MODEL_NAME)

        inputs = []
        if prompt:
            inputs.append(prompt)

        if file:
            # Convert file to bytes
            img_bytes = await file.read()
            inputs.append({"mime_type": file.content_type, "data": img_bytes})

        # Run Gemini multimodal generation
        response = await run_in_threadpool(model.generate_content, inputs)

        output_text = getattr(response, "text", None)
        if not output_text:
            raise RuntimeError("No text returned from Gemini.")

        return {"output": output_text}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


# 9) Root — optional friendly message
@app.get("/")
def root():
    return {"message": "Gemini Relay API. POST /generate with { 'prompt': '...'}"}
