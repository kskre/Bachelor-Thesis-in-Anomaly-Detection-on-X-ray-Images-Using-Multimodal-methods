import datetime
from supabase import create_client
import bcrypt

"""
This file represents data access layer for patients, images, symptoms, lab tests, and user authentication 
in a multi-tier architecture.

get_or_create_patient(...): fetch or create a patient by personal_code and user_id.
find_patient_by_name_or_code(...): search patients by name or code for a given user.
save_original_image(...): record an original patient image with upload date.
save_processed_image(...): record a processed image linked to the original.
save_symptom_entry(...): log a patient symptom with the current date.
save_lab_test_entry(...): log a lab test result with the date.
register_user(...): create a new user with a bcrypt-hashed password.
login_user(...): authenticate a user by verifying a bcrypt-hashed password.

"""

url = "https://uvbloyfklkhemhwzrgwe.supabase.co" 
key = " " 

supabase = create_client(url, key)

def get_or_create_patient(firstname, surname, personal_code, user_id):
    existing = (
        supabase
        .table("patients")
        .select("*")
        .eq("personal_code", personal_code)
        .eq("user_id", user_id)
        .execute()
    )
    if existing.data:
        return existing.data[0]
    created = (
        supabase
        .table("patients")
        .insert({
            "firstname": firstname,
            "surname": surname,
            "personal_code": personal_code,
            "user_id": user_id
        })
        .execute()
    )
    return created.data[0]

def find_patient_by_name_or_code(query: str, user_id: int):
    q = query.strip().lower()
    resp = supabase.table("patients").select("*").eq("user_id", user_id).execute()
    all_patients = resp.data

    matches = []
    for p in all_patients:
        fn = (p.get("firstname") or "").lower()
        ln = (p.get("surname")   or "").lower()
        pc = str(p.get("personal_code") or "").lower()
        if q in fn or q in ln or q in pc:
            matches.append(p)

    return matches

def save_original_image(patient_id: int, file_url: str, notes: str = "") -> dict:
    rec = {
        "patient_id":  patient_id,
        "file_url":    file_url,
        "upload_date": datetime.date.today().isoformat(),
        "notes":       notes
    }
    return supabase.table("original_images").insert(rec).execute().data[0]

def save_processed_image(original_id: int, file_url: str, notes: str = "") -> dict:
    rec = {
        "original_id":  original_id,
        "file_url":     file_url,
        "created_date": datetime.date.today().isoformat(),
        "notes":        notes
    }
    return supabase.table("processed_images").insert(rec).execute().data[0]

def save_symptom_entry(patient_id: int, symptom: str) -> None:
    supabase.table("symptoms").insert({
        "patient_id":  patient_id,
        "symptom":     symptom,
        "create_date": datetime.date.today().isoformat()
    }).execute()

def save_lab_test_entry(patient_id: int, test_name: str, test_value: float) -> None:
    supabase.table("lab_tests").insert({
        "patient_id":  patient_id,
        "test_name":   test_name,
        "test_value":  test_value,
        "result_date": datetime.date.today().isoformat()
    }).execute()


def register_user(user_name: str, password: str):
    existing = supabase.table("user_data").select("*").eq("user_name", user_name).execute()
    if existing.data:
        raise ValueError("This user already exists!")
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    user = (
        supabase.table("user_data")
        .insert({"user_name": user_name, "user_password": password_hash})
        .execute()
    )
    return user.data[0]

def login_user(user_name: str, password: str):
    user = supabase.table("user_data").select("*").eq("user_name", user_name).execute()
    if not user.data:
        return None
    user = user.data[0]
    if bcrypt.checkpw(password.encode(), user["user_password"].encode()):
        return user
    return None

