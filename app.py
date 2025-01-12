from flask import Flask, render_template, request, Response, jsonify, session, redirect, url_for
import os
import cv2
from werkzeug.utils import secure_filename
import json
import FaceDetect
import random
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import numpy as np
from pathlib import Path
import time

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "password123"

reqDict = {"status": 0, "name": "", "mail": "", "profile": ""}
otpValue = 0
ft = {"status": 0, "index": 0}
voted_users = set()  # Set to keep track of users who have voted
otp_sent = False
recognition_start_time = None
recognition_duration = 3  # seconds

# Initialize face detection without recognition
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Add a global variable to track the last OTP request time
last_otp_request_time = 0
otp_request_interval = 30  # Increase interval to 30 seconds between OTP requests

def sendMail(otpdata, mail):
    try:
        mail_content = f'''Hello,
Your OTP is : {otpdata}
Thank You
'''
        # The mail addresses and password
        sender_address = 'sthaprak24241@gmail.com'
        sender_pass = 'ijym okyd obyo aenx'

        receiver_address = mail
        # Setup the MIME
        message = MIMEMultipart()
        message['From'] = sender_address
        message['To'] = receiver_address
        message['Subject'] = 'OTP for voting'  # The subject line
        # The body and the attachments for the mail
        message.attach(MIMEText(mail_content, 'plain'))
        # Create SMTP session for sending the mail
        session_email = smtplib.SMTP('smtp.gmail.com', 587)  # use gmail with port
        session_email.starttls()  # enable security
        session_email.login(sender_address, sender_pass)  # login with mail_id and password
        text = message.as_string()
        session_email.sendmail(sender_address, receiver_address, text)
        session_email.quit()
        print('Mail Sent')
    except Exception as e:
        print(e)

def profileDb(name, info):
    m_path = os.getcwd()
    j_path = os.path.join(m_path, 'profileDb.json')
    data = ""
    try:
        with open(j_path, "r") as f:
            data = json.load(f)
    except:
        data = {}
    data[name] = info
    with open(j_path, 'w') as j_file:
        json.dump(data, j_file, indent=4)

def F(name, path, obj):
    f_path = os.path.join(path, name)
    if not os.path.exists(f_path):
        os.makedirs(f_path)
    for file in obj:
        filename = secure_filename(file.filename)
        f_path1 = os.path.join(f_path, filename)
        if not os.path.exists(f_path1):
            file.save(f_path1)

def load_voted_users():
    global voted_users
    try:
        if os.path.exists("voted_users.json"):
            with open("voted_users.json", "r") as f:
                voted_list = json.load(f)
                voted_users = set(voted_list)
        else:
            # Initialize the file with an empty list if it doesn't exist
            with open("voted_users.json", "w") as f:
                json.dump([], f)
    except Exception as e:
        print("Error loading voted users:", e)

def save_voted_user(name):
    global voted_users
    try:
        # Strip whitespace from name before adding
        cleaned_name = name.strip()
        voted_users.add(cleaned_name)  # Add the user to the set of voted users
        with open("voted_users.json", "w") as f:
            json.dump(list(voted_users), f, indent=4)  # Save the set as a formatted JSON
        print(f"User {cleaned_name} added to voted users list.")
    except Exception as e:
        print("Error saving voted user:", e)

app = Flask(__name__, template_folder='templates')
app.secret_key = os.urandom(24)  # Enable session support

@app.route('/', methods=["GET", "POST"])
def server_app():
    global reqDict
    global ft
    ft["status"] = 0
    if request.method == "GET":
        return render_template("index.html")
    else:
        data = request.data.decode('utf-8')
        print(data)
        return ""

@app.route('/main', methods=["GET", "POST"])
def main():
    global reqDict, recognition_start_time, otp_sent
    if request.method == "GET":
        reqDict = {"status": 0, "name": "", "mail": "", "profile": ""}
        recognition_start_time = None
        return render_template("main.html")
    else:
        return jsonify(reqDict)  # Return JSON response instead of redirect

def gen():
    global reqDict, otp_sent, recognition_start_time

    cam = cv2.VideoCapture(0)

    while True:
        try:
            ret, frame = cam.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Use FaceDetect's face detector
            faces = FaceDetect.face_detector.detectMultiScale(
                gray,
                scaleFactor=FaceDetect.scalefactor,
                minNeighbors=FaceDetect.minneighbors
            )

            if len(faces):
                for (left, top, width, height) in faces:
                    # Get face ROI in color
                    face_roi = frame[top:top+height, left:left+width]

                    # Get recognition result
                    detected_name, confidence = FaceDetect.recog(face_roi, FaceDetect.RECOGNITION_THRESHOLD)

                    # Draw rectangle
                    cv2.rectangle(frame, (left, top), (left+width, top+height), (10, 0, 255), 2)

                    if detected_name != "Unknown":
                        try:
                            # Load profile data
                            with open("profileDb.json", "r") as f:
                                data = json.load(f)

                            if detected_name in voted_users:
                                reqDict = {"status": 0, "name": "", "mail": "", "profile": ""}
                                message = f"{detected_name} has already voted!"
                                cv2.putText(frame, message, (10, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            else:
                                if detected_name == "admin":
                                    reqDict["status"] = 2
                                else:
                                    reqDict["mail"] = data[detected_name]["mail"]
                                    reqDict["profile"] = data[detected_name]["profile"]
                                    reqDict["status"] = 1
                                    reqDict["name"] = detected_name

                                # Display name and confidence
                                label = f"{detected_name} ({confidence:.2f})"
                                cv2.putText(frame, label, (left, top-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 0, 255), 2)

                        except Exception as e:
                            print(f"Error processing recognition: {e}")
                    else:
                        reqDict = {"status": 0, "name": "", "mail": "", "profile": ""}
                        cv2.putText(frame, "Unknown", (left, top-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 0, 255), 2)
            else:
                reqDict = {"status": 0, "name": "", "mail": "", "profile": ""}

            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

        except Exception as e:
            print(f"Error in gen(): {e}")
            continue

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/form', methods=["GET", "POST"])
def form():
    try:
        if request.method == 'POST':
            if 'face files[]' not in request.files:
                print('No file part')
                return render_template("form.html")

            # Save files to a temporary location
            temp_dir = os.path.join(os.getcwd(), 'temp_uploads')
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            face_files = request.files.getlist('face files[]')
            profile_files = request.files.getlist('profile files[]')

            # Save files and store paths
            face_paths = []
            profile_paths = []

            for file in face_files:
                if file.filename:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(temp_dir, filename)
                    file.save(filepath)
                    face_paths.append(filepath)

            for file in profile_files:
                if file.filename:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(temp_dir, filename)
                    file.save(filepath)
                    profile_paths.append(filepath)

            # Store paths and other data in session
            session['temp_data'] = {
                'face_paths': face_paths,
                'profile_paths': profile_paths,
                'name': request.form.get("uname"),
                'mail': request.form.get("mail")
            }

            # Generate and send OTP
            global otpValue
            otpValue = random.randint(1000, 9999)
            print(f"\nSignup OTP generated: {otpValue}\n")
            sendMail(otpValue, session['temp_data']['mail'])

            return redirect(url_for('signup_otp'))

    except Exception as e:
        print(f"Error in form(): {e}")

    return render_template("form.html", success=False)

@app.route('/signup-otp', methods=["GET", "POST"])
def signup_otp():
    if request.method == "GET":
        return render_template('signup_otp.html')

    if request.method == "POST":
        try:
            data = request.get_json()
            otp = int(data['otp'])
            print(f"Received OTP: {otp}")
            print(f"Stored OTP: {otpValue}")

            if int(otp) == int(otpValue):
                if 'temp_data' not in session:
                    print("Error: No temporary data found in session")
                    return {"data": 0}

                temp_data = session['temp_data']
                print(f"Processing data for user: {temp_data['name']}")

                try:
                    # Move files from temp location to permanent storage
                    m_path = os.getcwd()
                    f_path = os.path.join(m_path, 'faces')

                    if not os.path.exists(f_path):
                        os.makedirs(f_path)

                    user_face_dir = os.path.join(f_path, temp_data['name'])
                    if not os.path.exists(user_face_dir):
                        os.makedirs(user_face_dir)

                    # Move face files to permanent location
                    for temp_path in temp_data['face_paths']:
                        if os.path.exists(temp_path):
                            filename = os.path.basename(temp_path)
                            new_path = os.path.join(user_face_dir, filename)
                            # If file exists, remove it first
                            if os.path.exists(new_path):
                                os.remove(new_path)
                            os.rename(temp_path, new_path)
                            print(f"Moved file from {temp_path} to {new_path}")

                    # Process faces and update database
                    print("Starting face processing...")
                    FaceDetect.face_processing()

                    print("Updating profile database...")
                    profileDb(temp_data['name'], {
                        "mail": temp_data['mail'],
                        "profile": temp_data['name'],
                        "hasVoted": False
                    })

                    # Clean up temp directory
                    for path in temp_data['profile_paths']:
                        if os.path.exists(path):
                            os.remove(path)
                            print(f"Removed temporary file: {path}")

                    # Clear temporary data
                    session.pop('temp_data', None)
                    print("Successfully completed signup process")

                    return {"data": 1, "redirect": url_for('success')}

                except Exception as e:
                    print(f"Error during file processing: {str(e)}")
                    # Clean up any remaining temporary files
                    for path in temp_data['face_paths'] + temp_data['profile_paths']:
                        if os.path.exists(path):
                            try:
                                os.remove(path)
                            except:
                                pass
                    return {"data": 0}
            else:
                print(f"OTP verification failed. Received: {otp}, Expected: {otpValue}")
                return {"data": 0}
        except Exception as e:
            print(f"Error in signup_otp route: {str(e)}")
            return {"data": 0}

@app.route('/success')
def success():
    return render_template('success.html')

@app.route('/profile', methods=["GET", "POST"])
def profile():
    global reqDict
    ft["status"] = 6
    if request.method == "GET":
        # Store the name in session to persist it
        if reqDict.get("name"):
            session['voter_name'] = reqDict["name"]
        return render_template("profile.html")
    else:
        data = request.data.decode('utf-8')
        if "party" in data:
            try:
                party = data.split("=")[1]
                print(f"Vote cast for party: {party}")
                
                # Get the voter's name from session instead of reqDict
                voter_name = session.get('voter_name')
                print(f"Voter name from session: {voter_name}")
                
                if voter_name:
                    # Update results.json
                    m_path = os.getcwd()
                    j_path = os.path.join(m_path, 'result.json')
                    try:
                        with open(j_path, "r") as f:
                            data = json.load(f)
                    except:
                        data = {"result": []}
                    
                    data["result"].append(party)
                    with open(j_path, 'w') as j_file:
                        json.dump(data, j_file, indent=4)
                    
                    # Save the voter's name
                    save_voted_user(voter_name)
                    print(f"Successfully recorded vote for {voter_name}")
                    
                    # Clear the session after successful vote
                    session.pop('voter_name', None)
                else:
                    print("Error: No voter name found in session")
                    
            except Exception as e:
                print(f"Error processing vote: {e}")
                
        return reqDict

@app.route('/otp', methods=["GET", "POST"])
def otp():
    global otpValue, reqDict, otp_sent
    if request.method == "GET":
        # Store the name in session when OTP page is loaded
        if reqDict.get("name"):
            session['voter_name'] = reqDict["name"]
            
        if not otp_sent:
            otpValue = random.randint(1000, 9999)
            print(f"\n=== OTP for {reqDict['name']}: {otpValue} ===\n")
            otp_sent = True

            sendMail(otpValue, reqDict["mail"])
            
            print("OTP sent to:", reqDict["mail"])
        else:
            print("OTP already sent, not sending again.")
        return render_template('signin_otp.html')

    if request.method == "POST":
        try:
            data = request.get_json()
            otp = int(data['otp'])
            print(f"Received OTP: {otp}")
            print(f"Stored OTP: {otpValue}")

            if otp == otpValue:
                otp_sent = False
                # Make sure reqDict has the name from session
                if session.get('voter_name'):
                    reqDict["name"] = session['voter_name']
                return {"data": 1}
            else:
                return {"data": 0}
        except Exception as e:
            print(f"Error in OTP verification: {e}")
            return {"data": 0}
    return ""

@app.route('/admin', methods=["GET", "POST"])
def admin():
    if request.method == "GET":
        return render_template("admin_login.html")
    elif request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            # Authentication successful, store user session
            session['admin_logged_in'] = True
            # Redirect the admin to the result page after successful login
            return redirect(url_for('result'))
        else:
            # Authentication failed, redirect back to login page with error message
            return render_template("admin_login.html", error="Invalid username or password")

@app.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        # If user is not authenticated, redirect to admin login page
        return redirect(url_for('admin'))
    else:
        # User is authenticated, redirect to the result page
        return redirect(url_for('result'))

@app.route('/result', methods=["GET", "POST"])
def result():
    if not session.get('admin_logged_in'):
        # If user is not authenticated, redirect to admin login page
        return redirect(url_for('admin'))
    elif request.method == "GET":
        return render_template("result.html")
    else:
        data = request.data.decode("utf-8")
        print(data)
        if data == "clear":
            # Clear both result.json and voted_users.json
            with open("result.json", "w") as f:
                json.dump({"result": []}, f)
            with open("voted_users.json", "w") as f:
                json.dump([], f)
            # Clear the voted_users set
            global voted_users
            voted_users.clear()

        if data == "get":
            try:
                resList = []
                with open("result.json", "r") as f:
                    resList = json.load(f)
                resList = resList["result"]
                print(resList)
                resData = []
                names = []
                for i in resList:
                    if i not in names:
                        resData.append([i, resList.count(i)])
                        names.append(i)
                print(names)
                return jsonify(resData)
            except:
                pass
        session.pop('admin_logged_in', None)
        return jsonify([["Party Names", "Votes"], []])

if __name__ == '__main__':
    load_voted_users()  # Load existing voted users when app starts
    app.run(host='0.0.0.0', port=80, debug=True)  # Local run