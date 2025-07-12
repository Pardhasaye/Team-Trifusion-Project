
# 🎓 Trifusion – AI Classroom Assistant

An AI-powered educational assistant to record, transcribe, summarize, answer questions, and generate quizzes from classroom lectures or uploaded audio. Built with Streamlit and powered by Whisper, BART, and Gemini.

---

## 👨‍👩‍👧‍👦 Team Members

| Role         | Name                | Email                        | Contributions                                      |
|--------------|---------------------|------------------------------|----------------------------------------------------|
| Team Leader  | A. Pardha Saye      | panapart@gitam.in            | App flow, audio logic, integration & testing       |
|              |                     |                              | and q and a gen openvino conversion                |
| Team Member  | G. Siva Manikanta   | sgudla2@gitam.in             | recording and transcription logic                  |
| Team Member  | N. Sai Siddharadha  | snarayan5@gitam.in           | Gemini integration, Q&A and quiz logic and ui      |


---

## 🚀 Features

- 🎙 Record or upload classroom audio
- 📝 Whisper-powered transcription (OpenVINO accelerated)
- 📄 Text summarization using BART
- 🤖 Question answering using Gemini API
- 🧪 Automatic multiple-choice quiz generation
- 🎛️ Dark theme Streamlit UI with multi-tab workflow

---

## 📦 Setup & Installation

1. Clone this repo:

```bash
git clone https://github.com/your-username/trifusion-ai.git
cd trifusion-ai
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your Gemini API key in `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-api-key-here"
```

4. Run the app:

```bash
streamlit run app.py
```

---

## 🖥️ How It Works

| Tab                | What It Does                                                     |
|--------------------|------------------------------------------------------------------|
| 🎙 Record & Transcribe | Records via mic or uploads `.wav`, transcribes using Whisper |
| 📝 Summarize        | Summarizes transcript using BART                               |
| ❓ Question Answering | Lets you ask anything about the transcript (Gemini-powered)  |
| 🧪 Quiz Generator   | Generates MCQs with correct answers                            |

---

## 📸 Screenshots

### 🔴 Recording Interface
![Recording]([./recordings/Screenshot%202025-07-12%20230840.png](https://drive.google.com/file/d/1I5fiVicbkXiG91xyLgcHZjjFd-v2aAOS/view?usp=sharing))

### 📄 Transcription Output
![Transcript]([./recordings/Screenshot%202025-07-12%20230857.png](https://drive.google.com/file/d/1WtozyXnY3fPJ4Z_rzCzjSqm2dbSUWGJI/view?usp=sharing))

### 💡 Q&A Tab
![Q&A]([./recordings/Screenshot%202025-07-12%20230912.png](https://drive.google.com/file/d/1o3AdoHQLG71tBjVxpSSL1_nANSbUrk_v/view?usp=sharing))

### 🧠 Quiz Results
![Quiz]([./recordings/Screenshot%202025-07-12%20230940.png](https://drive.google.com/file/d/1hCzsaXQv7LndmExXvcfxVUKMyckARlsD/view?usp=sharing))

### 📝 Summary
![Summary]([./recordings/Screenshot%202025-07-12%20231001.png](https://drive.google.com/file/d/11I2pB8g7NcqhapId-gPgJuUGkSZ6QKmB/view?usp=sharing))

---

## 🎥 Demo Video

📽️ [Click to Watch Demo]([https://your-link-here.com](https://drive.google.com/file/d/1c84dbWXYKZqSEqa6jJpSZ7481ElS2cNQ/view?usp=sharing))

```html
<iframe width="560" height="315" src="https://your-embed-link.com" frameborder="0" allowfullscreen></iframe>
```

---

## 📁 Project Structure

```
├── app.py
├── requirements.txt
├── recordings/
│   ├── *.wav
│   ├── transcript.txt
│   ├── summary.txt
│   └── quiz.txt
├── .streamlit/
│   └── secrets.toml
└── README.md
```

---

## 📎 Dependencies

```
streamlit
numpy
librosa
soundfile
torch
transformers
openvino
pvrecorder
google-generativeai
```

---

## 🛠 Future Work

-Better optimization for models
- Add user profiles
- Export full lecture packs (Transcript + Summary + Quiz)
- Upload support for `.mp3` and `.m4a`

---

## 📜 License

MIT License (or your preferred one)
