import pyaudio
import wave
import os
import time


FORMAT = pyaudio.paInt16  
CHANNELS = 1              
RATE = 44100              
CHUNK = 1024              
OUTPUT_FOLDER = "uploads"  

os.makedirs(OUTPUT_FOLDER, exist_ok=True)  

try:
    while True:
        
        input("Press Enter to start recording...")
        
        
        audio = pyaudio.PyAudio()
        
       
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        
        print("Recording...")
        
        frames = []
        
        
        try:
            while True:
                data = stream.read(CHUNK)
                frames.append(data)
        except KeyboardInterrupt:
            print("Recording interrupted.")
        
        print("Recording finished.")
        
       
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        if frames:
            
            timestamp = time.strftime("%Y%m%d%H%M%S")
            output_filename = os.path.join(OUTPUT_FOLDER, f"output_{timestamp}.wav")
            
            
            with wave.open(output_filename, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
            
            print(f"Audio saved as {output_filename}")
        else:
            print("No audio recorded.")
except KeyboardInterrupt:
    print("\nProgram terminated.")
