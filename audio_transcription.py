import pyaudio
import wave
import keyboard
import time
import whisper
import datetime

def record_and_transcribe():
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

        # Transcribe the recorded audio
        transcription_model = whisper.load_model("base")
        transcription_result = transcription_model.transcribe(OUTPUT_FILENAME)
        print("Transcription Result:")
        print(transcription_result["text"])

        print(f"Recording and transcription completed successfully. Saved as {OUTPUT_FILENAME}")
        return transcription_result["text"]

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

# Call the function
# record_and_transcribe()
  