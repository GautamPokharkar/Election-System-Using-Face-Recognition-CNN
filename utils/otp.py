 # utils/otp.py
import random
import smtplib
from email.message import EmailMessage
from config import SMTP_CONFIG

def generate_otp(length=6):
    return ''.join([str(random.randint(0,9)) for _ in range(length)])

def send_otp(email_to: str, otp: str):
    msg = EmailMessage()
    msg['Subject'] = 'Your Election System OTP'
    msg['From'] = SMTP_CONFIG['smtp_user']
    msg['To'] = email_to
    msg.set_content(f'Your OTP is: {otp}\nDo not share this with anyone.')

    server = smtplib.SMTP(SMTP_CONFIG['smtp_server'], SMTP_CONFIG['smtp_port'])
    server.starttls()
    server.login(SMTP_CONFIG['smtp_user'], SMTP_CONFIG['smtp_password'])
    server.send_message(msg)
    server.quit()

