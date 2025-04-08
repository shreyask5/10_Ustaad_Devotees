import os
import sys
import cv2
import base64
import json
import time
import threading
import queue
import smtplib
import numpy as np
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import google.generativeai as genai
from twilio.rest import Client

app = Flask(__name__)
CORS(app)

# Add this function before loading the model
def ensure_model_exists(model_path="./models/yolo11n.pt"):
    """Check if YOLO model exists and download it if not."""
    model_dir = os.path.dirname(model_path)
    
    # Create models directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory: {model_dir}")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Downloading...")
        
        # Import YOLO and download the model
        try:
            from ultralytics import YOLO
            
            # Download the model - ultralytics will automatically download from their repo
            model = YOLO("yolov11n")  # This will download the model
            
            # Save the model to the specified path
            model.export(format="pt", path=model_path)
            print(f"Model downloaded and saved to {model_path}")
        except Exception as e:
            print(f"Error downloading model: {e}")
            sys.exit(1)
    else:
        print(f"Model found at {model_path}")

# Ensure model exists before loading
ensure_model_exists()

# Configure Gemini API
GEMINI_API_KEY = ''
genai.configure(api_key=GEMINI_API_KEY)

# Load the YOLO model
model = YOLO("./models/yolo11n.pt")

# Global variables for image and audio collection
image_queue = queue.Queue()
audio_queue = queue.Queue()
processing_lock = threading.Lock()
last_gemini_request_time = 0
GEMINI_REQUEST_INTERVAL = 60  # Minimum seconds between Gemini API requests

# File utility functions


def get_file_size(file_path):
    """Get file size in megabytes"""
    return os.path.getsize(file_path) / (1024 * 1024)

def validate_audio_file(file_path):
    """Validate if the file exists and is an audio file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    _, ext = os.path.splitext(file_path)
    supported_extensions = ['.opus', '.mp3', '.wav', '.ogg', '.m4a', '.waptt']
    
    if ext.lower() not in supported_extensions:
        raise ValueError(f"Unsupported file format. Supported formats: {', '.join(supported_extensions)}")
    
    return True

def validate_video_file(file_path):
    """Validate if the file exists and is a video file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    _, ext = os.path.splitext(file_path)
    supported_extensions = ['.mp4', '.mov', '.avi', '.webm', '.mkv', '.3gp']
    
    if ext.lower() not in supported_extensions:
        raise ValueError(f"Unsupported file format. Supported formats: {', '.join(supported_extensions)}")
    
    return True

# Image processing functions
def encode_image(image):
    """Encode image to base64 for Gemini API"""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def detect_face(image):
    """Detect if a face is present in the image using YOLO"""
    results = model(image)
    class_ids = results[0].boxes.cls.cpu().tolist()
    return 0 in class_ids or "person" in [results[0].names[int(cls)] for cls in class_ids]

def extract_frames(video_path, frames_per_second=24, max_dim=256):
    """Extract compressed frames from a video file"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame extraction intervals
    extract_interval = int(fps / frames_per_second)
    if extract_interval < 1:
        extract_interval = 1
    
    frames = []
    for frame_idx in range(0, frame_count, extract_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame to reduce size
        h, w = frame.shape[:2]
        if h > max_dim or w > max_dim:
            if h > w:
                new_h, new_w = max_dim, int(w * max_dim / h)
            else:
                new_h, new_w = int(h * max_dim / w), max_dim
            frame = cv2.resize(frame, (new_w, new_h))
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    cap.release()
    return frames

# Email notification function
def send_emergency_email(sender_email, sender_password, recipient_email, emergency_data):
    """Send an emergency alert email using Gmail SMTP server"""
    # Create message container
    msg = MIMEMultipart('alternative')
    msg['Subject'] = f"Emergency Alert: {emergency_data.get('type', 'Threat Detection')}"
    msg['From'] = sender_email
    msg['To'] = recipient_email

    # Create the email body HTML template
    html = f"""
    <p>ALERT: A potential {emergency_data.get('type', 'threat')} has been detected.</p>
    <table style="width: 100%; border-collapse: collapse; margin-bottom: 15px;">
      <tr>
        <td style="padding: 8px; border: 1px solid #d0d0d0; background-color: #f9f9f9;">
          <strong>Detection Type:</strong>
        </td>
        <td style="padding: 8px; border: 1px solid #d0d0d0;">
          {emergency_data.get('type', 'Unknown')}
        </td>
      </tr>
      <tr>
        <td style="padding: 8px; border: 1px solid #d0d0d0; background-color: #f9f9f9;">
          <strong>Assessment:</strong>
        </td>
        <td style="padding: 8px; border: 1px solid #d0d0d0;">
          {emergency_data.get('assessment', 'N/A')}
        </td>
      </tr>
      <tr>
        <td style="padding: 8px; border: 1px solid #d0d0d0; background-color: #f9f9f9;">
          <strong>Name:</strong>
        </td>
        <td style="padding: 8px; border: 1px solid #d0d0d0;">
          {emergency_data.get('name', 'N/A')}
        </td>
      </tr>
      <tr>
        <td style="padding: 8px; border: 1px solid #d0d0d0; background-color: #f9f9f9;">
          <strong>Contact Phone:</strong>
        </td>
        <td style="padding: 8px; border: 1px solid #d0d0d0;">
          {emergency_data.get('phone', 'N/A')}
        </td>
      </tr>
      <tr>
        <td style="padding: 8px; border: 1px solid #d0d0d0; background-color: #f9f9f9;">
          <strong>Emergency Contact Phone:</strong>
        </td>
        <td style="padding: 8px; border: 1px solid #d0d0d0;">
          {emergency_data.get('emergencyPhone', 'N/A')}
        </td>
      </tr>
      <tr>
        <td style="padding: 8px; border: 1px solid #d0d0d0; background-color: #f9f9f9;">
          <strong>Location:</strong>
        </td>
        <td style="padding: 8px; border: 1px solid #d0d0d0;">
          Latitude: {emergency_data.get('latitude', 'N/A')}<br>
          Longitude: {emergency_data.get('longitude', 'N/A')}<br>
          <a href="{emergency_data.get('mapsLink', '#')}" style="color: #1a73e8;">View on Google Maps</a>
        </td>
      </tr>
      <tr>
        <td style="padding: 8px; border: 1px solid #d0d0d0; background-color: #f9f9f9;">
          <strong>Timestamp:</strong>
        </td>
        <td style="padding: 8px; border: 1px solid #d0d0d0;">
          {emergency_data.get('timestamp', 'N/A')}
        </td>
      </tr>
    </table>
    <p style="color: #666; font-size: 0.9em;">
      Integrated Threat Detection System
    </p>
    """

    # Attach HTML content
    msg.attach(MIMEText(html, 'html'))

    try:
        # Setup the SMTP server
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()  # Enable security
            # Login to the server
            server.login(sender_email, sender_password)
            
            # Send email
            server.sendmail(sender_email, recipient_email, msg.as_string())
        
        print(f"Emergency alert email sent successfully to {recipient_email}")
        return True
    
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

# Analysis functions for voice and video
def analyze_audio(audio_data, mime_type):
    """Process audio data using Gemini API"""
    try:
        # System instructions for the model
        system_prompt = """
        You are a voice threat detection system. Your task is to analyze speech input and assess whether it presents a verbal threat.

        Reporting Guidelines:
        - Output strictly in the format: Intensity: <intensity>: <flag> : <explanation>
        - <flag> must be one of: "threat" or "no threat"
        - Intensity must be: Low / Medium / High — based on tone, aggression, and severity
        - Explanation must include:
            • HOW the threat or incident was expressed (specific words/phrases or tone)
            • IMPACT — what body parts, emotional state, or safety factors are affected
        - Keep the explanation clear and minimal to allow rapid understanding
        - Report only one critical flag per audio
        """
        
        # Generation configuration
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 256,
        }
        
        # Initialize model with system instructions
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash", 
            generation_config=generation_config
        )
        
        # First send the system prompt
        chat = model.start_chat(history=[
            {
                "role": "user",
                "parts": [system_prompt]
            },
            {
                "role": "model",
                "parts": ["I understand. I'll analyze audio for verbal threats using the specified format."]
            }
        ])
        
        # Then send the audio
        response = chat.send_message([
            {"text": "Analyze the audio for any verbal threats and report using the specified format."},
            {"inline_data": {
                "mime_type": mime_type,
                "data": audio_data
            }}
        ])
        
        return response.text.strip()
    
    except Exception as e:
        return f"Error processing audio: {str(e)}"

def analyze_images(images):
    """Analyze images using Gemini API"""
    try:
        # System instructions for the model
        system_prompt = """
        You are a video threat detection system. Your task is to analyze video frames and assess whether they present a visual threat.

        Reporting Guidelines:
        - Output strictly in the format: Intensity: <intensity>: <flag> : <explanation>
        - <flag> must be one of: "threat" or "no threat"
        - Intensity must be: Low / Medium / High — based on visuals, apparent danger, and severity
        - Explanation must include:
            • WHAT is visible in the frames (objects, people, environments)
            • HOW the threat or incident is presented (visual cues, positions, movements)
            • IMPACT — what physical, emotional, or safety factors are affected
        - Keep the explanation clear and minimal to allow rapid understanding
        - Report only one critical flag per segment
        """
        
        # Generation configuration
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 256,
        }
        
        # Initialize model with system instructions
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-latest", 
            generation_config=generation_config
        )
        
        # First send the system prompt
        chat = model.start_chat(history=[
            {
                "role": "user",
                "parts": [system_prompt]
            },
            {
                "role": "model",
                "parts": ["I understand. I'll analyze the video frames for threats using the specified format."]
            }
        ])
        
        # Prepare message parts
        message_parts = [
            {"text": "Analyze these video frames for any threats and report using the specified format. These are key frames from a 30-second video segment."}
        ]
        
        # Add frames (limited to 24 frames)
        max_frames = min(24, len(images))
        for i in range(max_frames):
            message_parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": encode_image(images[i])
                }
            })
        
        # Send the message with frames
        response = chat.send_message(message_parts)
        
        return response.text.strip()
    
    except Exception as e:
        return f"Error processing video frames: {str(e)}"

# Worker threads for processing
def process_image_and_audio_queue():
    """Process images from the queue in 30-second batches"""
    images_batch = []
    audio_batch = []
    last_process_time = time.time()
    batch_start_time = time.time()
    
    while True:
        try:
            # Process images
            if not image_queue.empty():
                image = image_queue.get(block=False)
                images_batch.append(image)
                image_queue.task_done()
            
            # Process audio
            if not audio_queue.empty():
                audio_item = audio_queue.get(block=False)
                audio_batch.append(audio_item)
                audio_queue.task_done()
            
            current_time = time.time()
            
            # Process batch when we have 30 seconds of data
            if (current_time - batch_start_time) >= 30:
                if images_batch and audio_batch:
                    with processing_lock:
                        # Check if we have faces in the images
                        has_faces = False
                        for image in images_batch:
                            if detect_face(image):
                                has_faces = True
                                break
                        
                        if has_faces:
                            # Check rate limiting
                            global last_gemini_request_time
                            time_since_last_request = current_time - last_gemini_request_time
                            
                            if time_since_last_request >= GEMINI_REQUEST_INTERVAL:
                                try:
                                    # Process video frames
                                    video_result = analyze_images(images_batch)
                                    print("Video Analysis Result:")
                                    print(video_result)
                                    
                                    # Process audio
                                    audio_result = analyze_audio(audio_batch[0]["data"], audio_batch[0]["mime_type"])
                                    print("Audio Analysis Result:")
                                    print(audio_result)
                                    
                                    # Update last request time
                                    last_gemini_request_time = current_time
                                    
                                    # Check for threats
                                    if ("threat" in video_result.lower() and not "no threat" in video_result.lower()) or \
                                       ("threat" in audio_result.lower() and not "no threat" in audio_result.lower()):
                                        send_threat_email("combined", f"Video: {video_result}\nAudio: {audio_result}")
                                except Exception as e:
                                    print(f"Error processing with Gemini API: {e}")
                                    # If rate limited, wait and try again in the next batch
                                    if "429" in str(e):
                                        time.sleep(GEMINI_REQUEST_INTERVAL)
                            else:
                                print(f"Rate limited. Waiting {GEMINI_REQUEST_INTERVAL - time_since_last_request:.1f} seconds")
                        
                    # Clear batches
                    images_batch = []
                    audio_batch = []
                    batch_start_time = current_time
                else:
                    # If we don't have both types of data, reset the timer
                    batch_start_time = current_time
            
            # Sleep briefly to prevent CPU overuse
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error in processing thread: {e}")
            time.sleep(1)

def send_threat_email(threat_type, assessment):
    """Send email alert for detected threats"""
    try:
        # Get email configuration from app config
        sender_email = app.config.get('SENDER_EMAIL', 'emergencyresponsesystem1@gmail.com')
        sender_password = app.config.get('SENDER_PASSWORD', 'qsdu vnit fbpt okjw')
        recipient_email = app.config.get('RECIPIENT_EMAIL', 'shreyasksh6@gmail.com')
        
        # User information
        user_info = app.config.get('USER_INFO', {})
        
        # Send the email
        send_emergency_email(
            sender_email=sender_email,
            sender_password=sender_password,
            recipient_email=recipient_email,
            emergency_data={
                'type': threat_type.capitalize() + " Threat",
                'assessment': assessment,
                'name': user_info.get('name', 'Unknown'),
                'phone': user_info.get('phone', 'N/A'),
                'emergencyPhone': user_info.get('emergency_phone', 'N/A'),
                'latitude': user_info.get('latitude', 'N/A'),
                'longitude': user_info.get('longitude', 'N/A'),
                'mapsLink': user_info.get('maps_link', '#'),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        )
        
        print(f"Threat email sent: {threat_type}")
        return True
    except Exception as e:
        print(f"Error sending threat email: {e}")
        return False

# API routes
@app.route('/api/configure', methods=['POST'])
def configure():
    """Configure the application with user and email settings"""
    try:
        data = request.json
        
        # Update app configuration
        app.config['SENDER_EMAIL'] = data.get('sender_email', app.config.get('SENDER_EMAIL', 'ustaaddevotees@gmail.com'))
        app.config['SENDER_PASSWORD'] = data.get('sender_password', app.config.get('SENDER_PASSWORD', 'xoyw swvd ycra mncl'))
        app.config['RECIPIENT_EMAIL'] = data.get('recipient_email', app.config.get('RECIPIENT_EMAIL', 'shreyasksh6@gmail.com'))
        
        # User information
        app.config['USER_INFO'] = {
            'name': data.get('name', 'Unknown'),
            'phone': data.get('phone', 'N/A'),
            'emergency_phone': data.get('emergency_phone', 'N/A'),
            'address': data.get('address', 'N/A'),
            'latitude': data.get('latitude', 'N/A'),
            'longitude': data.get('longitude', 'N/A'),
            'maps_link': data.get('maps_link', '#')
        }
        
        return jsonify({"status": "success", "message": "Configuration updated successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/api/image', methods=['POST'])
def receive_image():
    """Receive image from client and add to processing queue"""
    try:
        if 'image' not in request.files:
            return jsonify({"status": "error", "message": "No image file provided"}), 400
        
        file = request.files['image']
        image = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        
        # Add to processing queue
        image_queue.put(image)
        
        return jsonify({"status": "success", "message": "Image received for processing"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/audio', methods=['POST'])
def receive_audio():
    """Receive audio from client and add to processing queue"""
    try:
        if 'audio' not in request.files:
            return jsonify({"status": "error", "message": "No audio file provided"}), 400
        
        file = request.files['audio']
        audio_data = file.read()
        
        # Determine MIME type
        filename = file.filename
        _, ext = os.path.splitext(filename)
        
        if ext.lower() == '.opus' or ext.lower() == '.waptt':
            mime_type = 'audio/ogg'
        elif ext.lower() == '.mp3':
            mime_type = 'audio/mpeg'
        elif ext.lower() == '.wav':
            mime_type = 'audio/wav'
        elif ext.lower() == '.m4a':
            mime_type = 'audio/mp4'
        else:
            mime_type = f'audio/{ext[1:].lower()}'
        
        # Add to processing queue
        audio_queue.put({
            "data": audio_data,
            "mime_type": mime_type
        })
        
        return jsonify({"status": "success", "message": "Audio received for processing"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/status', methods=['GET'])
def status():
    """Get system status"""
    return jsonify({
        "status": "running",
        "image_queue_size": image_queue.qsize(),
        "audio_queue_size": audio_queue.qsize()
    })

@app.route('/sos', methods=['GET'])
def handle_data():
    # Extract query parameters
    name = request.args.get('name')
    emergency_contact = request.args.get('emergency_contact')
    location = request.args.get('location')
    
    # Add country code +91 if not already present
    if emergency_contact and not emergency_contact.startswith('+91'):
        emergency_contact = f'+91{emergency_contact}'
    
    # Split location into latitude and longitude
    if location:
        lat, lon = location.split(',')
        # Create Google Maps link with the coordinates
        maps_link = f"https://maps.google.com/?q={lat},{lon}"
    else:
        lat, lon = None, None
        maps_link = "Location unavailable"

    # Twilio credentials
    account_sid = ""
    auth_token = ""
    client = Client(account_sid, auth_token)
    
    # Create emergency message with Google Maps link for SMS
    sms_message = f"EMERGENCY ALERT: {name} is in need of immediate assistance. Location: {maps_link}"
    
    # Create emergency message for call (without coordinates)
    call_message = f"EMERGENCY ALERT: {name} is in need of immediate assistance. Check messages for more details immediately."
    
    # Send emergency SMS first
    try:
        message = client.messages.create(
            body=sms_message,
            from_="+19152333816",
            to=emergency_contact,
        )
        message_sid = message.sid
    except Exception as e:
        message_sid = f"Error sending message: {str(e)}"
    
    # Then make emergency call
    try:
        call = client.calls.create(
            twiml=f"<Response><Say>{call_message}</Say></Response>",
            to="+918660870744",
            from_="+19152333816",
        )
        call_sid = call.sid
    except Exception as e:
        call_sid = f"Error making call: {str(e)}"

    # Create a response
    response = {
        "name": name,
        "emergency_contact": emergency_contact,
        "latitude": lat,
        "longitude": lon,
        "maps_link": maps_link,
        "message_sid": message_sid,
        "call_sid": call_sid
    }
    
    return jsonify(response)




# Initialize the server
def initialize_server():
    """Initialize the server and start worker threads"""
    # Start image processing thread
    image_and_audio_thread = threading.Thread(target=process_image_and_audio_queue, daemon=True)
    image_and_audio_thread.start()

    
    print("Worker threads started")

if __name__ == '__main__':
    # Default configuration
    app.config['SENDER_EMAIL'] = 'ustaaddevotees@gmail.com'
    app.config['SENDER_PASSWORD'] = 'qsdu vnit fbpt okjw'
    app.config['RECIPIENT_EMAIL'] = 'shreyasksh6@gmail.com'
    app.config['USER_INFO'] = {}
    
    # Initialize server and start worker threads
    initialize_server()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)