from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import os
import uvicorn
from pydantic import BaseModel, ValidationError
from typing import Literal, List, Union
from store_models import models

"""
main server for patient management, image upload/processing, model inference, and record storage
in a multi-tier architecture.

Endpoints:
  POST   /register          - register a new user
  POST   /login             - authenticate a user
  POST   /analyze           - upload an image and run lab or symptom model inference
  GET    /patients/search   - autocomplete patients by name or code
  POST   /patients          - fetch or create a patient record
  POST   /records           - save original/processed images and lab/symptom entries

"""

from base import (
    get_or_create_patient,
    find_patient_by_name_or_code,
    save_original_image,
    save_processed_image,
    save_symptom_entry,
    save_lab_test_entry,
    login_user,
    register_user
)

from model_inference import predict as predict_symptoms, load_models_symptom
from model_inference_lab import predict as predict_labs, load_models_lab

UPLOAD_FOLDER = "raw_files"
OUTPUT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")
app.mount("/results", StaticFiles(directory=OUTPUT_FOLDER), name="results")

LABEL_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Effusion',
    'Emphysema', 'Fibrosis', 'Infiltration', 'Pneumonia', 'Pneumothorax'
]

class LabEntry(BaseModel):
    test: str
    value: Union[int, float] 

class AnalyzePayload(BaseModel):
    model: Literal['lab', 'symptoms']
    region: str
    labs: List[LabEntry]     = []
    symptoms: List[str]      = []
class UserRegisterPayload(BaseModel):
    user_name: str
    password: str

class UserLoginPayload(BaseModel):
    user_name: str
    password: str

@app.post("/register")
async def register(payload: UserRegisterPayload):
    try:
        user = register_user(payload.user_name, payload.password)
        return {"status": "ok", "user_id": user["id"], "user_name": user["user_name"]}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/login")
async def login(payload: UserLoginPayload):
    user = login_user(payload.user_name, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return {"status": "ok", "user_id": user["id"], "user_name": user["user_name"]}

@app.on_event("startup")
async def startup_event():
    print("Loading models into memory…")
    models['symptoms'] = load_models_symptom()  
    models['lab']     = load_models_lab() 
    print("Models loaded.")

@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    payload_raw: str   = Form(...)
):
    filename   = os.path.basename(image.filename)
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(input_path, "wb") as buf:
        shutil.copyfileobj(image.file, buf)

    try:
        payload = AnalyzePayload.parse_raw(payload_raw)
    except ValidationError as e:
        raise HTTPException(400, f"Invalid payload: {e}")

        # Model selection
    if payload.model == 'lab':
        labs   = {entry.test: entry.value for entry in payload.labs}
        result = predict_labs(input_path, labs, models['lab'])
    else:
        symptoms_text = ", ".join(payload.symptoms)
        result = predict_symptoms(
            input_path,
            symptoms_text,
            LABEL_NAMES,
            model_dict=models['symptoms']
        )


    # Get results
    boxed_fn = os.path.basename(result["boxed_path"])
    response = {
        "original_url":     f"/uploads/{filename}",
        "boxed_url":        f"/results/{boxed_fn}",
        "predicted_labels": result["predicted_labels"],
        "predicted_probs":  result["predicted_probs"],
    }

    if payload.model == 'lab':
        response["filtered_labs"] = result.get("filtered_labs", [])
    else:
        response["filtered_symptoms"] = result.get("filtered_symptoms", "")

    return JSONResponse(response)

@app.get("/patients/search")
async def search_patients(q: str = Query(..., min_length=2), user_id: int = Query(...)):
    return find_patient_by_name_or_code(q, user_id)

@app.post("/patients")
async def create_patient(body: dict = Body(...)):
    firstname    = body.get("firstname")
    surname      = body.get("surname")
    personal_code = body.get("personal_code")
    user_id       = body.get("user_id")
    if not (firstname and surname and personal_code):
        raise HTTPException(400, "firstname, surname и personal_code обязательны")
    patient = get_or_create_patient(firstname, surname, personal_code, user_id) 
    return patient

@app.post("/records")
async def save_record(data: dict = Body(...)):
    """
    We expect to get something like:
    {
      "patient_id": 42,
      "model": "symptoms"|"lab",
      "region": "Lungs",
      "symptoms": [...],      // in case if model == "symptoms"
      "labs": [...],          // in case if model == "lab"
      "original_url": "/uploads/xyz.png",
      "heatmap_url": "/results/xyz_boxed.png",
      "notes": "..."         
    }
    """
    pid      = data.get("patient_id")
    orig_url = data.get("original_url")
    heat_url = data.get("heatmap_url")

    if not (pid and orig_url and heat_url):
        raise HTTPException(400, "patient_id, original_url и heatmap_url обязательны")


    orig_rec = save_original_image(
        patient_id=pid,
        file_url=orig_url,
        notes=data.get("notes", "")
    )


    proc_rec = save_processed_image(
        original_id=orig_rec["id"],
        file_url=heat_url,
        notes=f"model={data.get('model')} region={data.get('region')}"
    )

    if data.get("model") == "symptoms":
        for s in data.get("symptoms", []):
            save_symptom_entry(patient_id=pid, symptom=s)
    else:
        for lab in data.get("labs", []):
            save_lab_test_entry(
                patient_id=pid,
                test_name=lab["test"],
                test_value=float(lab["value"])
            )

    return {"status": "ok", "original_id": orig_rec["id"], "processed_id": proc_rec["id"]}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
