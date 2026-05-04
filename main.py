from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from supabase import create_client, Client
from dotenv import load_dotenv
import base64
import os

load_dotenv()

app = FastAPI(
    title="Task 3 HD OCR Backend",
    docs_url="/docs",
    openapi_url="/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

class SaveRequest(BaseModel):
    filename: str
    extracted_text: str

@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "HD OCR backend is running on Render"
    }

@app.get("/config")
def config_check():
    return {
        "groq_key_exists": bool(GROQ_API_KEY),
        "supabase_url_exists": bool(SUPABASE_URL),
        "supabase_key_exists": bool(SUPABASE_KEY)
    }

@app.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    encoded = base64.b64encode(content).decode("utf-8")
    mime_type = file.content_type or "image/png"

    try:
        client = Groq(api_key=GROQ_API_KEY)

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all visible text from this image. Return only the extracted text."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{encoded}"
                            }
                        }
                    ]
                }
            ]
        )

        extracted_text = response.choices[0].message.content.strip()

        return {
            "filename": file.filename,
            "message": extracted_text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save")
def save_result(data: SaveRequest):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase is not configured")

    try:
        result = supabase.table("ocr_history").insert({
            "filename": data.filename,
            "extracted_text": data.extracted_text
        }).execute()

        return {
            "status": "saved",
            "data": result.data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def get_history():
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase is not configured")

    try:
        result = (
            supabase.table("ocr_history")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )

        return {
            "count": len(result.data),
            "items": result.data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))