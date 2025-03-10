# Subtitle Generator

This project automates **subtitle generation for videos** by:
- Extracting **audio** from a video
- Transcribing and translating **speech** using OpenAI's **Whisper**
- Generating **SRT subtitle files** with accurate timestamps
- Optionally **burning subtitles** into the video

## Features
✅ **Multilingual Support** – Detects and translates speech to English  
✅ **Automatic Time Syncing** – Generates **SRT** subtitles with timestamps  
✅ **Fast Processing** – Uses **Torch & Hugging Face Transformers** for efficient speech recognition  
✅ **GPU Support** – Runs on CUDA for **faster transcription** (if available)  

---

## Installation
### **1. Install Dependencies**
#### **For GPU Users (CUDA 12.6)**
```sh
pip install -r requirements-gpu.txt
```
#### **For CPU Users**
```sh
pip install -r requirements-cpu.txt
```

### **2. Install FFmpeg**
If FFmpeg is not installed, install it via:
- **Windows:** [Download FFmpeg](https://ffmpeg.org/download.html)
- **Linux/macOS:** Run:
  ```sh
  sudo apt install ffmpeg  # Ubuntu/Debian
  brew install ffmpeg  # macOS
  ```

---

## Usage
Run the `subtitle_generator.ipynb` Jupyter Notebook to process a video and generate subtitles.

### **1. Extract Audio from Video**
```python
import subprocess

video_path = "/path/to/video.mp4"
audio_path = "audio.wav"

subprocess.run(["ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", audio_path, "-y"])
print("Audio extracted successfully!")
```

### **2. Transcribe & Translate Speech**
```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch, librosa

# Load Whisper Model
model_name = "openai/whisper-large-v2"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

# Load & Process Audio
audio, sr = librosa.load(audio_path, sr=16000)
inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to("cuda" if torch.cuda.is_available() else "cpu")

# Transcribe & Translate
forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="translate")
with torch.no_grad():
    result = model.generate(inputs, forced_decoder_ids=forced_decoder_ids)

transcription = processor.batch_decode(result, skip_special_tokens=True)[0]
print("Transcription:", transcription)
```

### **3. Generate SRT Subtitle File**
```python
from datetime import timedelta

srt_path = "subtitles.srt"

def format_timestamp(seconds):
    return str(timedelta(seconds=int(seconds))).replace(".", ",") + ",000"

with open(srt_path, "w", encoding="utf-8") as srt_file:
    for i, segment in enumerate(transcription.split(". ")):  
        start_time = i * 5  
        end_time = start_time + 5  

        srt_file.write(f"{i+1}\n{format_timestamp(start_time)} --> {format_timestamp(end_time)}\n{segment}\n\n")

print(f"Subtitles saved to {srt_path}")
```

### **4. Burn Subtitles into Video (Optional)**
```sh
ffmpeg -y -i /path/to/video.mp4 -vf subtitles=subtitles.srt -c:a copy output.mp4
```

---

## Folder Structure
```
/subtitle_generator/
│── subtitle_generator.ipynb  # Jupyter Notebook for processing
│── requirements-gpu.txt      # Dependencies for GPU users
│── requirements-cpu.txt      # Dependencies for CPU users
│── subtitles.srt             # Generated subtitle file
│── output.mp4                # Video with embedded subtitles (optional)
```

---

## License
This project is open-source under the **MIT License**.  
Feel free to use, modify, and contribute! 🎉  

---

## Author
👨‍💻 **aakash j.v.v**  
📧 **aakashjammula6@gmail.com**  
🌐 **[linkedin](https://www.linkedin.com/in/aakashjammula/)**  