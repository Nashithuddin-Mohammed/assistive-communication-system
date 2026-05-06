"""
Flask + WebSocket Real-time Assistive Communication System
- Video: Browser captures, server processes emotion, sends back
- Audio: Browser captures continuously, server transcribes on pause
- True async - no page refresh needed
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info/warning messages

import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from deepface import DeepFace
import speech_recognition as sr
import base64
import io
import wave
import time
import pyttsx3
from scipy import signal
import tempfile
import json
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'assistive-comm-secret'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='eventlet',
    ping_timeout=60,
    ping_interval=25,
    max_http_buffer_size=10 * 1024 * 1024  # 10MB - allows larger audio chunks
)

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# Global state
emotion_detector_lock = eventlet.semaphore.Semaphore()
transcriptions = []
current_emotion = "Neutral"  # Track current detected emotion


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'message': 'Connected to server'})


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


@socketio.on('video_frame')
def handle_video_frame(data):
    """Process video frame for emotion detection"""
    try:
        # Decode base64 image
        img_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return

        # Emotion detection
        with emotion_detector_lock:
            try:
                result = DeepFace.analyze(
                    frame,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )

                if result:
                    if isinstance(result, list):
                        result = result[0]

                    emotions = result.get('emotion', {})
                    if emotions:
                        global current_emotion
                        emotion = max(emotions, key=emotions.get)
                        confidence = float(emotions[emotion]) / 100.0
                        region = result.get('region', {})

                        # Update current emotion for transcription tagging
                        current_emotion = emotion.capitalize()

                        # Convert numpy float32 to regular floats for JSON serialization
                        emotions_serializable = {k: float(v) for k, v in emotions.items()}
                        region_serializable = {k: int(v) if isinstance(v, (np.integer, np.floating)) else v for k, v in region.items()}

                        emit('emotion_result', {
                            'emotion': emotion,
                            'confidence': confidence,
                            'box': region_serializable,
                            'all_emotions': emotions_serializable
                        })
            except Exception as e:
                print(f"Emotion detection error: {e}")

    except Exception as e:
        print(f"Frame processing error: {e}")


@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    """Process audio chunk for speech-to-text"""
    try:
        # Decode base64 audio - client sends complete WAV file
        audio_data = base64.b64decode(data['audio'])
        sample_rate = data.get('sampleRate', 48000)
        print(f"Received audio: {len(audio_data)} bytes at {sample_rate}Hz")

        # Read the WAV file
        wav_buffer = io.BytesIO(audio_data)

        # If sample rate is high (48kHz), resample to 16kHz for better recognition
        if sample_rate > 16000:
            with wave.open(wav_buffer, 'rb') as wav_in:
                n_channels = wav_in.getnchannels()
                sampwidth = wav_in.getsampwidth()
                n_frames = wav_in.getnframes()
                audio_frames = wav_in.readframes(n_frames)

            # Convert to numpy array and resample properly
            audio_array = np.frombuffer(audio_frames, dtype=np.int16).astype(np.float32)

            # Calculate target number of samples for 16kHz
            target_samples = int(len(audio_array) * 16000 / sample_rate)

            # Use scipy's resample for proper anti-aliasing
            resampled_float = signal.resample(audio_array, target_samples)

            # Convert back to int16
            resampled = np.clip(resampled_float, -32768, 32767).astype(np.int16)

            # Debug: check audio levels
            max_amplitude = np.max(np.abs(resampled))
            print(f"Audio max amplitude: {max_amplitude} (should be >1000 for speech)")

            # Create new WAV at 16kHz
            resampled_buffer = io.BytesIO()
            with wave.open(resampled_buffer, 'wb') as wav_out:
                wav_out.setnchannels(1)
                wav_out.setsampwidth(2)
                wav_out.setframerate(16000)
                wav_out.writeframes(resampled.tobytes())
            resampled_buffer.seek(0)
            wav_buffer = resampled_buffer
            print(f"Resampled to 16000Hz, {len(resampled)} samples")
        else:
            wav_buffer.seek(0)

        # Get audio duration for logging
        wav_buffer.seek(0)
        with sr.AudioFile(wav_buffer) as source:
            recognizer = sr.Recognizer()
            audio = recognizer.record(source)
            print(f"Audio duration: {len(audio.frame_data) / audio.sample_rate / audio.sample_width:.2f}s")

        # Save to temp file for subprocess
        wav_buffer.seek(0)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(wav_buffer.read())
            tmp_path = tmp.name

        try:
            # Connect to persistent transcription server
            print("Sending to Whisper server...")
            import socket as sock

            client = sock.socket(sock.AF_INET, sock.SOCK_STREAM)
            client.settimeout(30)

            try:
                client.connect(('127.0.0.1', 5050))
                client.send(tmp_path.encode('utf-8'))

                response = client.recv(4096).decode('utf-8')
                result = json.loads(response)

            except sock.timeout:
                print("Transcription server timeout")
                emit('error', {'message': 'Transcription timeout'})
                return
            except ConnectionRefusedError:
                print("Transcription server not running! Start it with: python transcribe_server.py")
                emit('error', {'message': 'Transcription server not running'})
                return
            finally:
                client.close()

            if result['success']:
                text = result['text']
                if text.strip():
                    # Combine transcription with current emotion
                    text_with_emotion = f"{text} [{current_emotion}]"
                    timestamp = time.strftime("%H:%M:%S")
                    transcription = {'text': text_with_emotion, 'time': timestamp, 'emotion': current_emotion}
                    transcriptions.append(transcription)
                    emit('transcription', transcription)
                    print(f"Transcribed: {text_with_emotion}")
            else:
                error = result['error']
                if error == 'unknown_value':
                    print("Could not understand audio - speech unclear or too quiet")
                    emit('status', {'message': 'Could not understand - speak louder/clearer'})
                else:
                    print(f"Transcription error: {error}")

        except Exception as e:
            print(f"Transcription connection error: {e}")
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
    except Exception as e:
        import traceback
        print(f"Audio processing error: {e}")
        traceback.print_exc()


@socketio.on('speak_text')
def handle_speak_text(data):
    """Text-to-speech - creates fresh engine each time to avoid async issues"""
    try:
        text = data.get('text', '')
        if text:
            # Create fresh engine for each request (fixes pyttsx3 async issues)
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()

            # Clean up engine
            try:
                engine.stop()
            except:
                pass

            emit('tts_complete', {'success': True})
    except Exception as e:
        print(f"TTS error: {e}")
        emit('error', {'message': f'TTS error: {e}'})


@socketio.on('get_transcriptions')
def handle_get_transcriptions():
    """Get all transcriptions"""
    emit('all_transcriptions', {'transcriptions': transcriptions})


@socketio.on('clear_transcriptions')
def handle_clear_transcriptions():
    """Clear transcription history"""
    global transcriptions
    transcriptions = []
    emit('transcriptions_cleared', {'success': True})


if __name__ == '__main__':
    print("Starting Flask server...")
    print("Open http://localhost:5000 in your browser")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
