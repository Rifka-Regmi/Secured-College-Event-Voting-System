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
recognition_duration = 5  # seconds

# Initialize face detection without recognition
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

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
        session = smtplib.SMTP('smtp.gmail.com', 587)  # use gmail with port
        session.starttls()  # enable security
        session.login(sender_address, sender_pass)  # login with mail_id and password
        text = message.as_string()
        session.sendmail(sender_address, receiver_address, text)
        session.quit()
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
        f_path1 = f_path
        f_path1 = os.path.join(f_path1, filename)
        if not os.path.exists(f_path1):
            file.save(f_path1)

def save_voted_user(name):
    try:
        voted_users.add(name)  # Add the user to the set of voted users
        with open("voted_users.json", "w") as f:
            json.dump(list(voted_users), f)  # Save the set to a JSON file
    except Exception as e:
        print("Error saving voted user:", e)

app = Flask(__name__, template_folder='templates')

@app.route('/', methods=["GET", "POST"])
def server_app():
    global reqDict
    global st
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
        otp_sent = False
        return render_template("main.html")
    else:
        return jsonify(reqDict)  # Return JSON response instead of redirect

def gen():
    global reqDict, otp_sent, recognition_start_time, otpValue
    
    # Initialize face recognition if training data exists
    faces_path = Path("cropped_faces.npy")
    labels_path = Path("cropped_labels.npy")
    
    if faces_path.exists() and labels_path.exists():
        print("Loading face recognition model...")
        try:
            face_recognizer = FaceDetect.train(str(faces_path), str(labels_path))
            print("Face recognition model loaded successfully")
        except Exception as e:
            print(f"Error loading face recognition model: {e}")
            face_recognizer = None
    else:
        print("No training data found")
        face_recognizer = None

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    last_detected_name = None
    
    while True:
        try:
            _, image = cam.read()
            image = cv2.flip(image, 1)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)
            
            if len(faces):
                for (left, top, width, height) in faces:
                    face_roi = gray[top:top + height, left:left + width]
                    
                    if face_recognizer is not None:
                        detected_name, confidence = FaceDetect.recog(face_roi, FaceDetect.RECOGNITION_THRESHOLD)
                    else:
                        detected_name, confidence = "Unknown", 0
                    
                    cv2.rectangle(image, (left, top), (left + width, top + height), (10, 0, 255), 2)
                    
                    if not "Unknown" in detected_name:
                        # Check if the same person is being detected
                        if last_detected_name != detected_name:
                            recognition_start_time = time.time()
                            last_detected_name = detected_name
                        
                        with open("profileDb.json", "r") as f:
                            data = json.load(f)
                            
                        if detected_name in voted_users:
                            reqDict = {"status": 0, "name": "", "mail": "", "profile": ""}
                            message = f"{detected_name} has already voted!"
                            cv2.putText(image, message, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        else:
                            if detected_name == "admin":
                                reqDict["status"] = 2
                            else:
                                time_elapsed = time.time() - recognition_start_time
                                remaining_time = max(0, recognition_duration - time_elapsed)
                                
                                # # Calculate accuracy percentage (inverse of confidence)
                                # accuracy = max(0, min(100, 100 - (confidence / FaceDetect.RECOGNITION_THRESHOLD * 100)))
                                
                                # Show name (white color)
                                name_msg = f"Name: {detected_name}"
                                cv2.putText(image, name_msg, (left, top-60), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                                
                                # # Show accuracy (green if >70%, yellow if >50%, red otherwise)
                                # accuracy_color = (0, 255, 0) if accuracy > 70 else (0, 255, 255) if accuracy > 50 else (0, 0, 255)
                                # accuracy_msg = f"Accuracy: {accuracy:.1f}%"
                                # cv2.putText(image, accuracy_msg, (left, top-35), 
                                #           cv2.FONT_HERSHEY_SIMPLEX, 0.9, accuracy_color, 2)
                                
                                # Show countdown (blue color)
                                countdown_msg = f"Verifying: {int(remaining_time)}s"
                                cv2.putText(image, countdown_msg, (left, top-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                                
                                # If recognition time is complete
                                if time_elapsed >= recognition_duration and not otp_sent:
                                    reqDict["mail"] = data[detected_name]["mail"]
                                    reqDict["profile"] = data[detected_name]["profile"]
                                    reqDict["status"] = 1
                                    reqDict["name"] = detected_name
                                    otp_sent = True
                                    # Generate and send OTP
                                    otpValue = random.randint(1000, 9999)
                                    print(f"\nLogin OTP generated: {otpValue}\n")
                                    sendMail(otpValue, reqDict["mail"])
                    else:
                        # Reset timer if face is not recognized
                        recognition_start_time = None
                        last_detected_name = None
            else:
                # Reset timer if no faces detected
                recognition_start_time = None
                last_detected_name = None
            
            ret, jpeg = cv2.imencode('.jpg', image)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        except Exception as e:
            print(f"Error : {e}")

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
        print(e)
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

# Add new success route
@app.route('/success')
def success():
    return render_template('success.html')

@app.route('/profile', methods=["GET", "POST"])
def profile():
    global reqDict
    ft["status"] = 6
    if request.method == "GET":
        return render_template("profile.html")
    else:
        data = request.data.decode('utf-8')
        if "party" in data:
            party = data.split("=")[1]
            print(party)
            m_path = os.getcwd()
            j_path = os.path.join(m_path, 'result.json')
            data = ""
            try:
                with open(j_path, "r") as f:
                    data = json.load(f)
            except:
                data = {"result": []}
            data["result"].append(party)
            with open(j_path, 'w') as j_file:
                json.dump(data, j_file, indent=4)
            save_voted_user(reqDict["name"])  # Save the name of the user who voted
        return reqDict

@app.route('/otp', methods=["GET", "POST"])
def otp():
    global otpValue, reqDict, otp_sent
    if request.method == "GET":
        if not otp_sent:
            otpValue = random.randint(1000, 9999)
            print(f"\nLogin OTP generated: {otpValue}\n")
            sendMail(otpValue, reqDict["mail"])
            otp_sent = True
        return render_template('signin_otp.html')
    if request.method == "POST":
        try:
            data = request.get_json()  # Get JSON data
            otp = int(data['otp'])  # Extract OTP from JSON
            print(f"Received OTP: {otp}")  # Debug print
            print(f"Stored OTP: {otpValue}")  # Debug print
            
            if otp == otpValue:
                otp_sent = False  # Reset for next login
                return {"data": 1}
            else:
                return {"data": 0}
        except Exception as e:
            print(f"Error in OTP verification: {e}")  # Debug print
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

# Add this line to enable sessions in your Flask app
app.secret_key = os.urandom(24)

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
            with open("result.json", "w") as f:
                json.dump({"result": []}, f)

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
    # app.run(host='127.0.0.1', port=5000, debug=True) #Cloud run
    app.run(host='0.0.0.0', port=80, debug=True)  #Local run