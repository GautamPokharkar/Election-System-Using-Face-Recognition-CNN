# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, session
import mysql.connector
from config import DB_CONFIG, SMTP_CONFIG, MODEL_PATH
from utils.validators import valid_aadhaar, valid_voterid, valid_phone, valid_email, is_eligible
from utils.otp import generate_otp, send_otp
from models.face_utils import capture_faces_for_aadhaar
from models.train_model import train_and_save
import os, pickle, datetime
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from werkzeug.security import check_password_hash

app = Flask(__name__)
app.secret_key = 'replace_with_a_secure_random_key'

# ---------------- DB helper ----------------
def get_db():
    return mysql.connector.connect(**DB_CONFIG)

# ---------------- Home ----------------
@app.route('/')
def home():
    return render_template('home.html')

# ---------------- Register ----------------
@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        data = request.form
        aadhaar = data.get('aadhaar','').strip()
        voter_id = data.get('voter_id','').strip().upper()
        first_name = data.get('first_name','').strip()
        middle_name = data.get('middle_name','').strip()
        last_name = data.get('last_name','').strip()
        phone = data.get('phone','').strip()
        email = data.get('email','').strip()
        try:
            age = int(data.get('age',0))
        except:
            flash('Invalid age', 'danger'); return redirect(url_for('register'))
        state = data.get('state','').strip()
        city = data.get('city','').strip()

        # Validations
        if not valid_aadhaar(aadhaar):
            flash('Aadhaar must be 12 digits', 'danger'); return redirect(url_for('register'))
        if not valid_voterid(voter_id):
            flash('Voter ID must be 3 letters + 5 digits', 'danger'); return redirect(url_for('register'))
        if not valid_phone(phone):
            flash('Phone must be 10 digits', 'danger'); return redirect(url_for('register'))
        if not valid_email(email):
            flash('Invalid email', 'danger'); return redirect(url_for('register'))
        if not is_eligible(age):
            flash('Not eligible to vote (age < 18)', 'danger'); return redirect(url_for('register'))

        # Check existing
        db = get_db(); cur = db.cursor(dictionary=True)
        cur.execute("SELECT * FROM voters WHERE aadhaar=%s OR voter_id=%s", (aadhaar, voter_id))
        if cur.fetchone():
            flash('Already registered', 'warning'); db.close(); return redirect(url_for('home'))

        # Insert new voter
        cur.execute("""INSERT INTO voters 
            (aadhaar,voter_id,first_name,middle_name,last_name,phone,email,age,state,city)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (aadhaar,voter_id,first_name,middle_name,last_name,phone,email,age,state,city))
        db.commit(); db.close()

        # Generate OTP
        otp = generate_otp()
        session['otp'] = otp
        session['pending_aadhaar'] = aadhaar

        try:
            send_otp(email, otp)
            flash('OTP sent to your email. Verify to proceed.', 'info')
            return redirect(url_for('verify_email'))
        except Exception as e:
            flash('Failed to send OTP: ' + str(e), 'danger')
            return redirect(url_for('home'))

    return render_template('register.html')

# ---------------- Email OTP verification ----------------
@app.route('/verify_email', methods=['GET','POST'])
def verify_email():
    if request.method == 'POST':
        otp_sub = request.form.get('otp','').strip()
        if 'otp' in session and otp_sub == session['otp'] and 'pending_aadhaar' in session:
            aadhaar = session['pending_aadhaar']
            db = get_db(); cur = db.cursor()
            cur.execute("UPDATE voters SET email_verified=1 WHERE aadhaar=%s", (aadhaar,))
            db.commit(); db.close()
            session.pop('otp', None)
            flash('Email verified. Proceed to face registration.', 'success')
            return redirect(url_for('capture_faces', aadhaar=aadhaar))
        else:
            flash('Invalid OTP', 'danger')
    return render_template('verify_email.html')

# ---------------- Capture faces ----------------
@app.route('/capture_faces/<aadhaar>')
def capture_faces(aadhaar):
    db = get_db(); cur = db.cursor(dictionary=True)
    cur.execute("SELECT email_verified FROM voters WHERE aadhaar=%s", (aadhaar,))
    row = cur.fetchone(); db.close()
    if not row or not row['email_verified']:
        flash('Email not verified', 'danger'); return redirect(url_for('home'))

    success = capture_faces_for_aadhaar(aadhaar)
    if success:
        db = get_db(); cur = db.cursor()
        cur.execute("UPDATE voters SET face_registered=1 WHERE aadhaar=%s", (aadhaar,))
        db.commit(); db.close()
        flash('Face registered successfully.', 'success')
    else:
        flash('Face capture incomplete.', 'warning')
    return redirect(url_for('home'))

# ---------------- Train model (admin) ----------------
@app.route('/admin/train', methods=['POST'])
def admin_train_post():
    try:
        train_and_save()
        flash('Model trained successfully.', 'success')
    except Exception as e:
        flash('Training failed: ' + str(e), 'danger')
    return redirect(url_for('home'))


# ---------------- Login ----------------
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        aadhaar = request.form.get('aadhaar','').strip()
        if not valid_aadhaar(aadhaar):
            flash('Invalid Aadhaar', 'danger'); return redirect(url_for('login'))
        db = get_db(); cur = db.cursor(dictionary=True)
        cur.execute("SELECT * FROM voters WHERE aadhaar=%s", (aadhaar,))
        row = cur.fetchone(); db.close()
        if not row:
            flash('Not registered', 'danger'); return redirect(url_for('register'))
        if not row['email_verified']:
            flash('Email not verified', 'warning'); return redirect(url_for('verify_email'))
        if not row['face_registered']:
            flash('Face not registered', 'warning'); return redirect(url_for('capture_faces', aadhaar=aadhaar))
        session['aadhaar'] = aadhaar
        flash('Proceed to facial verification.', 'info')
        return redirect(url_for('face_verify'))
    return render_template('login.html')

# ---------------- Load model helper ----------------
def load_model_and_encoder():
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = load_model(MODEL_PATH)
    le_path = os.path.join(os.path.dirname(MODEL_PATH), 'le.pkl')
    le = None
    if os.path.exists(le_path):
        with open(le_path, 'rb') as f:
            le = pickle.load(f)
    return model, le

# ---------------- Face verification ----------------
@app.route('/face_verify')
def face_verify():
    if 'aadhaar' not in session:
        flash('Login first', 'danger'); return redirect(url_for('login'))

    model, le = load_model_and_encoder()
    if model is None:
        flash('Model not trained', 'danger'); return redirect(url_for('home'))

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cam = cv2.VideoCapture(0)
    predicted = None
    tries = 0

    while True:
        ret, frame = cam.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (50,50)).astype('float32')/255.0
            arr = face.reshape(1,50,50,1)
            preds = model.predict(arr)
            idx = np.argmax(preds, axis=1)[0]
            label = le.inverse_transform([idx])[0] if le else None
            predicted = label
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, str(label), (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0),2)

        cv2.imshow('Face Verification - Press q to confirm', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        tries += 1
        if tries > 1000: break

    cam.release(); cv2.destroyAllWindows()
    if predicted == session['aadhaar']:
        flash('Identity verified. Proceed to voting.', 'success')
        return redirect(url_for('vote_page'))
    else:
        flash('Face not recognized. Contact help desk.', 'danger')
        return redirect(url_for('home'))

# ---------------- Vote page ----------------
@app.route('/vote', methods=['GET','POST'])
def vote_page():
    if 'aadhaar' not in session:
        flash('Login first', 'danger'); return redirect(url_for('login'))
    aadhaar = session['aadhaar']

    db = get_db(); cur = db.cursor()
    cur.execute("SELECT * FROM votes WHERE aadhaar=%s", (aadhaar,))
    if cur.fetchone():
        db.close(); flash('Already voted', 'warning'); session.pop('aadhaar',None); return redirect(url_for('home'))
    db.close()

    if request.method == 'POST':
        party = request.form.get('party')
        if not party:
            flash('Select a party', 'danger'); return redirect(url_for('vote_page'))
        now = datetime.datetime.now()
        db = get_db(); cur = db.cursor()
        cur.execute("INSERT INTO votes (aadhaar, party, vote_date, vote_time) VALUES (%s,%s,%s,%s)",
                    (aadhaar, party, now.date(), now.time()))
        db.commit(); db.close()
        flash('Vote cast successfully', 'success')
        session.pop('aadhaar', None)
        return redirect(url_for('home'))

    parties = [
        {'id':'BJP','img':'/static/images/party_symbols/bjp.png'},
        {'id':'CONGRESS','img':'/static/images/party_symbols/congress.png'},
        {'id':'AAP','img':'/static/images/party_symbols/aap.png'},
        {'id':'NOTA','img':'/static/images/party_symbols/nota.png'}
    ]
    return render_template('vote.html', parties=parties)

# ---------------- Update details ----------------
@app.route('/update', methods=['GET','POST'])
def update_details():
    if request.method == 'POST':
        aadhaar = request.form.get('aadhaar','').strip()
        db = get_db(); cur = db.cursor(dictionary=True)
        cur.execute("SELECT * FROM voters WHERE aadhaar=%s", (aadhaar,))
        row = cur.fetchone()
        if not row:
            flash('No such user', 'danger'); db.close(); return redirect(url_for('update_details'))

        fields = ['first_name','middle_name','last_name','phone','email','state','city']
        updates, vals = [], []
        for f in fields:
            v = request.form.get(f)
            if v:
                updates.append(f"{f}=%s")
                vals.append(v)
        if 'age' in request.form:
            age = int(request.form.get('age'))
            if not is_eligible(age):
                flash('Not eligible', 'danger'); db.close(); return redirect(url_for('update_details'))
            updates.append("age=%s"); vals.append(age)

        if updates:
            vals.append(aadhaar)
            sql = "UPDATE voters SET " + ", ".join(updates) + " WHERE aadhaar=%s"
            cur.execute(sql, tuple(vals)); db.commit()
        db.close()
        flash('Updated. Re-verify email if changed.', 'success')
        return redirect(url_for('home'))

    return render_template('update.html')


# ---------------- Admin auth and dashboard ----------------
from werkzeug.security import check_password_hash
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        db = get_db(); cur = db.cursor(dictionary=True)
        cur.execute("SELECT * FROM admins WHERE username=%s", (username,))
        admin = cur.fetchone(); cur.close(); db.close()
        if admin and check_password_hash(admin.get('password_hash',''), password):
            session['role'] = 'admin'
            session['admin_id'] = admin.get('id')
            session['username'] = admin.get('username')
            flash('Admin logged in', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid admin credentials', 'danger')
    return render_template('admin_login.html')

@app.route('/admin/dashboard')
def admin_dashboard():
    if session.get('role') != 'admin':
        flash('Admin access required', 'warning'); return redirect(url_for('home'))
    db = get_db(); cur = db.cursor(dictionary=True)
    cur.execute("SELECT * FROM voters ORDER BY created_at DESC")
    voters = cur.fetchall(); cur.close(); db.close()
    return render_template('admin_dashboard.html', voters=voters)

@app.route('/admin/train')
def admin_train():
    if session.get('role') != 'admin':
        flash('Admin access required', 'warning'); return redirect(url_for('home'))
    try:
        # call existing training function
        train_and_save()
        flash('Model retraining finished', 'success')
    except Exception as e:
        flash('Training failed: ' + str(e), 'danger')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/add', methods=['GET', 'POST'])
def admin_add():
    if session.get('role') != 'admin':
        flash('Admin access required', 'warning'); return redirect(url_for('home'))
    if request.method == 'POST':
        aadhaar = request.form.get('aadhaar','').strip()
        voter_id = request.form.get('voter_id','').strip().upper()
        first_name = request.form.get('first_name','').strip()
        middle_name = request.form.get('middle_name','').strip()
        last_name = request.form.get('last_name','').strip()
        phone = request.form.get('phone','').strip()
        email = request.form.get('email','').strip()
        age = request.form.get('age','').strip()
        state = request.form.get('state','').strip()
        city = request.form.get('city','').strip()

        db = get_db(); cur = db.cursor()
        cur.execute("""INSERT INTO voters (aadhaar, voter_id, first_name, middle_name, last_name, phone, email, age, state, city)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (aadhaar, voter_id, first_name, middle_name, last_name, phone, email, age, state, city))
        db.commit(); cur.close(); db.close()
        flash('Voter added', 'success')
        return redirect(url_for('admin_dashboard'))
    return render_template('admin_add.html')

@app.route('/admin/edit/<int:voter_id>', methods=['GET', 'POST'])
def admin_edit(voter_id):
    if session.get('role') != 'admin':
        flash('Admin access required', 'warning'); return redirect(url_for('home'))
    db = get_db(); cur = db.cursor(dictionary=True)
    if request.method == 'POST':
        cur.execute("""UPDATE voters SET first_name=%s, middle_name=%s, last_name=%s,
                       phone=%s, email=%s, age=%s, state=%s, city=%s WHERE id=%s""",
                    (request.form.get('first_name'), request.form.get('middle_name'),
                     request.form.get('last_name'), request.form.get('phone'),
                     request.form.get('email'), request.form.get('age'),
                     request.form.get('state'), request.form.get('city'), voter_id))
        db.commit(); cur.close(); db.close()
        flash('Voter updated', 'success')
        return redirect(url_for('admin_dashboard'))

    cur.execute("SELECT * FROM voters WHERE id=%s", (voter_id,))
    voter = cur.fetchone(); cur.close(); db.close()
    return render_template('admin_edit.html', voter=voter)

@app.route('/admin/delete/<int:voter_id>', methods=['POST'])
def admin_delete(voter_id):
    if session.get('role') != 'admin':
        flash('Admin access required', 'warning'); return redirect(url_for('home'))
    db = get_db(); cur = db.cursor()
    cur.execute("DELETE FROM voters WHERE id=%s", (voter_id,))
    db.commit(); cur.close(); db.close()
    flash('Voter deleted', 'success')
    return redirect(url_for('admin_dashboard'))


@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out', 'info')
    return redirect(url_for('home'))

# ---------------- Admin results ---------------- ----------------
@app.route('/admin/results')
def admin_results():
    db = get_db(); cur = db.cursor(dictionary=True)
    cur.execute("SELECT party, COUNT(*) as votes FROM votes GROUP BY party")
    rows = cur.fetchall(); db.close()
    return render_template('results.html', results=rows)

# ---------------- Run ----------------
if __name__ == '__main__':
    app.run(debug=True)
