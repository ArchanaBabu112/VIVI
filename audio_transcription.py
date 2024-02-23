import os
from openai import OpenAI
from dotenv import load_dotenv
import pyaudio
import wave
import keyboard
import time
import datetime

load_dotenv()

class A_t:

    def record_and_transcribe(self):
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        CHUNK = 1024

        # Use the default input device
        INPUT_DEVICE_INDEX = None

        # Generate a unique filename using the current timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        OUTPUT_FILENAME = f"recordedFile_{timestamp}.wav"

        print("Recording started")

        try:
            audio = pyaudio.PyAudio()

            # Check if the default input device is available
            if INPUT_DEVICE_INDEX is not None and INPUT_DEVICE_INDEX >= audio.get_device_count():
                raise ValueError("Input device not found")

            stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, frames_per_buffer=CHUNK, input=True, input_device_index=INPUT_DEVICE_INDEX)
            frames = []

            print("Press Space to start recording")
            keyboard.wait("space")
            print("Recording... Press space to stop")
            time.sleep(0.2)

            while True:
                try:
                    data = stream.read(CHUNK)
                    frames.append(data)
                except KeyboardInterrupt:
                    break
                if keyboard.is_pressed("space"):
                    print("Stopping recording after a delay")
                    time.sleep(0.2)
                    break

            stream.stop_stream()
            stream.close()
            audio.terminate()

            # Save recorded audio to a WAV file
            waveFile = wave.open(OUTPUT_FILENAME, "wb")
            waveFile.setnchannels(CHANNELS)
            waveFile.setsampwidth(audio.get_sample_size(FORMAT))
            waveFile.setframerate(RATE)
            waveFile.writeframes(b''.join(frames))
            waveFile.close()

            # Transcribe the recorded audio using OpenAI GPT-3
            api_key = os.environ.get("OPENAI_API_KEY")
            client = OpenAI(api_key=api_key)

            audio_file = open(OUTPUT_FILENAME, "rb")
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
            print("Transcription Result:")
            print(transcript)

            print(f"Recording and transcription completed successfully. Saved as {OUTPUT_FILENAME}")
            return transcript

        except ValueError as e:
            print("Error:", e)
            print("Oh sorry, your microphone is missing.")

        except IOError as e:
            if e.errno == -9998:
                print("Error: Invalid number of channels")
                print("Please ensure that your microphone is set to use 1 channel.")
            else:
                print(f"Error: {e}")

        except Exception as e:
            print(f"Error: {e}")

# Instantiate the class
a_t_instance = A_t()

# Call the record_and_transcribe function
transcription_result = a_t_instance.record_and_transcribe()