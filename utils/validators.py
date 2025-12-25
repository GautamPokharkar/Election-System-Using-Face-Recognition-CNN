 # utils/validators.py
import re

def valid_aadhaar(aadhaar: str) -> bool:
    return bool(re.fullmatch(r'\d{12}', aadhaar))

def valid_voterid(voter: str) -> bool:
    # 3 uppercase letters + 5 digits
    return bool(re.fullmatch(r'[A-Z]{3}\d{5}', voter))

def valid_phone(phone: str) -> bool:
    return bool(re.fullmatch(r'\d{10}', phone))

def valid_email(email: str) -> bool:
    # simple check; use email-validator for stronger checks
    return bool(re.fullmatch(r"[^@]+@[^@]+\.[^@]+", email))

def is_eligible(age: int) -> bool:
    return age >= 18

