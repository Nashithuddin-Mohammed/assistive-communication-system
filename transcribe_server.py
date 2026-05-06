"""
Persistent transcription server - keeps Whisper model loaded in memory
Communicates via TCP socket for fast responses
"""
import socket
import json
import os
import sys

# Suppress warnings before importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'

print("Loading Whisper model...", flush=True)
from faster_whisper import WhisperModel

# Load model once at startup
MODEL = WhisperModel("tiny", device="cpu", compute_type="int8")
print("Model loaded! Server ready.", flush=True)

HOST = '127.0.0.1'
PORT = 5050


def transcribe(wav_path):
    try:
        segments, info = MODEL.transcribe(
            wav_path,
            beam_size=1,
            language="en",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        text = " ".join(text_parts)

        if text.strip():
            return {'success': True, 'text': text}
        else:
            return {'success': False, 'error': 'unknown_value'}

    except Exception as e:
        return {'success': False, 'error': str(e)}


def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server.bind((HOST, PORT))
    except OSError as e:
        print(f"Port {PORT} already in use. Kill existing server first.", flush=True)
        sys.exit(1)

    server.listen(5)
    print(f"Transcription server listening on {HOST}:{PORT}", flush=True)

    while True:
        try:
            client, addr = server.accept()

            # Receive wav file path
            data = client.recv(4096).decode('utf-8').strip()

            if data:
                result = transcribe(data)
                client.send(json.dumps(result).encode('utf-8'))

            client.close()

        except KeyboardInterrupt:
            print("\nShutting down server...")
            break
        except Exception as e:
            print(f"Error: {e}", flush=True)

    server.close()


if __name__ == '__main__':
    main()
