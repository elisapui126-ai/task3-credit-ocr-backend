from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
import base64
import os

app = FastAPI(
    title="Task 3 Credit OCR Backend",
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

@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "Credit OCR backend is running successfully on Render"
    }

@app.get("/config")
def config_check():
    return {
        "groq_key_exists": bool(os.getenv("GROQ_API_KEY"))
    }

@app.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    encoded = base64.b64encode(content).decode("utf-8")
    mime_type = file.content_type or "image/png"

    try:
        client = Groq(api_key=api_key)

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