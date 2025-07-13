
# Ai powered classroom Assistant


Team Trifusion – AI-Enabled Recording Assistant for Transcription, Summarization, Q&A and Quiz Generation From classroom Lectures.


## Team

| Role         | Name                | Email                        | Contributions                                      |
|--------------|---------------------|------------------------------|----------------------------------------------------|
| Team Leader  | A. Pardha Saye      | panapart@gitam.in            | App flow, audio logic, integration , testing       |
|              |                     |                              | documentatoin                                      |
|              |                     |                              | and q and a gen openvino conversion                | 
|              |                     |                              |                                                    |
| Team Member  | G. Siva Manikanta   | sgudla2@gitam.in             | recording and transcription quiz gen logic         |
|              |                     |                              | and documentation                                  |
|              |                     |                              |                                                    |
|              |                     |                              |                                                    |
| Team Member  | N. Sai Siddharadha  | snarayan5@gitam.in           | Gemini integration, Q&A and quiz logic and ui      |
|              |                     |                              |  and documentation                                 |

## Installation and usage
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
4. Download the files in the same folder as the project `.whishper-small-hf`:

```toml
https://drive.google.com/drive/folders/1b2fVEMqlCfn6N60dMTOr6nToJdOn_k3a?usp=sharing
```
5. Download the files in the same folder as the project `.whishper-small-openvino`:

```toml
https://drive.google.com/drive/folders/1SKKNaY6h3XvobQ1_g5hRXp3pJU2NluWg?usp=sharing
```

6. Run the app:

```bash
streamlit run app.py
```

---
## Introduction
At a time when university and college classrooms are churning out more knowledge nuggets than can ever be actually learned due to the painstaking efforts to record, transcribe, compile knowledge into create study materials. The AI-enabled classroom Recording Assistant, as described, solves this by capturing (record audio) classroom instruction for converting, in real-time, into fully-interactive study material.

These and the other features offered in a Streamlit interface were developed to frustrate effort, to make engagement active and to capture classroom knowledge - that is in a format that is easy to find, search and reuse.

## Problem Statement 4
AI-Powered Interactive Learning Assistant for Classrooms:
----------------------------------------------------------

Build a Multimodal AI assistant for classrooms to dynamically answer queries using text, voice, and visuals while improving student engagement with personalized responses. 

----------------------------------------------------------------


Our Objective
-------------
The AI-enabled recording assistant was developed for one purpose, to establish a bulk writing system that gives learners an efficient and automated way to record a classroom-like lecture and combine that classroom-like lecture into usable study material.

Record Live Lectures - the intention is to facilitate utilising any notes from the lecture, that means there is no longer a need to be stuck in a recording studio or listen to a previously prepared audio which has to be stored in a separate folder from the recording to notes.
From live lecturing,. the audible speech transforms it to readable text
AI will be able to summarize, condense and shorten text into more readable summaries.
AI will assimilate the content and construct multiple-choice quizzes as part of the revision process.
Interactive Questions and Answers Chat - this will enable users to ask adaptive general questions corresponding to the lecture's topic and obtain an effective corresponding instant detailed answer.

In the end, Trifusion hopes ultimately gives learners and educators the time to not even consider the processes to record and assemble their content, and instead concentrate on and listen to what is about to be discussed during their lecture and focus on everything associated with exploring, learning, or practicing the content.

Solution Overview
-----------------
Trifusion integrates AI’s, and Automatic Speech Recognition + Summarization and automated models in a single monolithic (single) Streamlit app. This singular workflow has allowed for automated transcripts, summaries, exercises (quizzes), and Q&A workflow (from the classes) or to turn raw audio into structured, interactive study material with little manual effort.

## Methodology
Iterative and Modular Development:
Instead of developing everything, the project used an iterative and feedback loop development cycle:

Stepwise modular design:

1.Audio Capture & Input Processing

2.Transcription

3.Summarization

4.Generative AI (Q & A and Quiz)

5.Streamlit User Interface

Build and iterate through each module:

1.Begin by prototyping each module at a time (almost exclusively for example, transcribe alone).

2.Test each module for accuracy, speed, and usability.Collect team feedback after each phase.

Sequential integration:

1.Once each module is stable on its own, integrate the modules into one Streamlit interface.

2.Introduce state management (i.e. st.session_state) to allow data to flow across tabs.
Test for seamless integration and regular usability testing.


Feedback loop:

1.Real world testing each time allowed real world analytic usability problems to come to the surface.

2.The  team created their models and selections, chunk sizes, and caching and the user interface as best as possible in order to maximize performance.

Outcome:

A lightweight and usable classroom assistant based on good practices of iterative approaches of design and real world testing 
Architecture

## Architecture
Audio Capture & Input Processing: 
Live Recording: utilizes the PvRecorder library that will capture the lecture audio as live via the microphone used by the user. The files are .wav files are timestamped (e. g. recording_20250712_123456.wav) when saved in the recordings subfolder. 
# Code path: record_audio() function in the main app
```http
  def record_audio(device_index):
    recorder = PvRecorder(device_index=device_index, frame_length=512)
    recorder.start()
    st.info("Recording... click Stop to finish.")
    frames = []
    stop = st.button("Stop Recording")
    if stop:
        recorder.stop()
        frames = recorder.read_all()
        audio_data = np.array(frames, dtype=np.int16)
        filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        filepath = os.path.join(SAVE_DIR, filename)
        sf.write(filepath, audio_data, samplerate=16000)
        return filepath
    return None
```




File Upload: 

allows the user to upload a pre-existing recording of the lecture which must necessarily be in .wav or .mp3 formats and gives the user an added feature to gain consistency in the preauditory processing regardless of pre-recorded or live. 

# Code path: file upload handling section in "Record & Transcribe" page
```http
uploaded = st.file_uploader("select or upload audio file", type=["wav", "mp3"])
if uploaded:
    filename = f"uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    path = os.path.join(SAVE_DIR, filename)
    with open(path, "wb") as f:
        f.write(uploaded.read())
    st.session_state.audio_path = path
    st.success("Uploaded successfully!")

```
The process includes dual input, providing a way for the user to capture the live lectures or offline lectures that were pre-recorded lectures. 
Both methods save files in:
recordings
    
    recording_20250712_123456.wav
    
    uploaded_20250712_124501.wav



Transcription
--------------
Feature Extraction: Audio files are sliced into ~30-second segments and converted into input features using WhisperProcessor (from Hugging Face).
# Code path: transcribe_with_encoder_decoder()
```html
input_features = transcriber_processor(
    chunk, sampling_rate=16000, return_tensors="pt"
).input_features

Encoding(Openvino Optimized): The Whisper encoder takes the features and provides hidden    states (vector representations) after being accelerated with OpenVINO to reduce latency and therefore achieve faster real-time transcription.
# Code path: transcribe_with_encoder_decoder()
encoder_output_np = encoder_model(
    inputs={"input_features": input_features.numpy()}
)[encoder_model.outputs[0]]
encoder_hidden_states = torch.tensor(encoder_output_np, dtype=torch.float32)
```


Decoding: The WhisperForConditionalGeneration decoder takes the hidden states and forced decoder IDs (for language and task) and produces text tokens.
# Code path: transcribe_with_encoder_decoder()
```html
forced_decoder_ids = transcriber_processor.get_decoder_prompt_ids(language="en", task="transcribe")
generated_ids = decoder_model.generate(
    encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden_states),
    forced_decoder_ids=forced_decoder_ids,
    max_length=448,
    do_sample=False
)
```

Reconstruction: The text tokens are decoded into human-readable text using batch_decode. Then, the chunks are combined in order to create the final transcript.
# Code path: transcribe_with_encoder_decoder()
```
transcription = transcriber_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
transcriptions.append(transcription)
```

This split encoder–decoder architecture may create a scalable solution for long lectures while providing reasonable accuracy.
The final transcript is:
```
return " ".join(transcriptions)
```

Summarization
------------------
The transcript will be broken up into chunks (≤1024 tokens) to fit in model input size.
# Code path: summarize_text() function
```
chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
```


Each chunk will be summarized independently with facebook/bart-large-cnn, a transformer model that has had success in the task of abstractive summarization.
# Code path: summarize_text() function
```
input_ids = summary_tokenizer(chunk, return_tensors="pt", truncation=True, max_length=1024).input_ids
output = summary_model.generate(input_ids, max_length=150)
summary = summary_tokenizer.decode(output[0], skip_special_tokens=True)
```

The individual partial summaries will be put together for a summary of the lecture.

This chunking process allows us to be sure that the summary is still meaningful and coherent even for lectures that are long.

Partial summaries will be combined for a single summary for the whole lecture that is coherent. This chunking makes the final summary meaningful and coherent even for long lectures.
# Code path: summarize_text() function
```
return " ".join(summaries)
```
```
def summarize_text(text, max_chunk_length=1024):
    chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    summaries = []
    for chunk in chunks:
        input_ids = summary_tokenizer(chunk, return_tensors="pt", truncation=True, max_length=1024).input_ids
        output = summary_model.generate(input_ids, max_length=150)
        summary = summary_tokenizer.decode(output[0], skip_special_tokens=True)
        summaries.append(summary)
    return " ".join(summaries)

```
Generative AI with Gemini (Q&A + Quiz)
----------------------------
Question Answering:

In the "❓ Question Answering" tab, users can ask any question related to the lecture content.

Under the hood:

Combines the full lecture transcript and the user's question into a natural language prompt

Sends the prompt to Gemini 1.5 Flash via Google’s official google.generativeai Python client

Returns a context-specific response that is intelligent, concise, and grounded in the transcript
Here’s how it works in code:

```
def gemini_answer(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gemini error: {str(e)}"
```

Example prompt sent to Gemini:

```
Based on the transcript below, answer this question:

Transcript:
[...full transcript here...]

Question: What were the key takeaways from the second half of the lecture?
```
This logic keeps the user interface simple and lightweight, while delivering high-quality results powered by Gemini’s large language model.

Quiz Generation:

In the "🧪 Quiz Generator" tab, automatically creates quizzes from the transcript.

Behind the scenes:

The full transcript is sent to Gemini with a specially designed prompt

Gemini returns 5 multiple-choice questions, each with 4 answer options and the correct one clearly marked

Prompt example:

```
Generate 5 multiple-choice quiz questions with 4 options each (A–D) based on the following transcript.
Mention the correct answer after each question.

Transcript:
[...lecture text...]
```
Code to fetch quiz from Gemini:

```
def gemini_answer(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gemini error: {str(e)}"
```
✅ Why Gemini Works Great Here:

1.No heavy ML models run locally – all LLM processing is done via the cloud.

2.Minimal latency – fast and reliable output.

3.Context-aware – responses are tailored to the provided transcript.

4.Effortless scalability – no need to manage infrastructure or retrain models.


Streamlit User Interface
------------------
Users navigate through four distinct tabs:

Record & Transcribe - Users can record live audio or upload – then transcribe and download corresponding transcript. 

Summarize - Users can summarize the transcript and download it.
Contextual Questions - Users can ask custom questions and receive corresponding answers.

 Quiz Builder - Users can create multiple choice quizzes and 
export them.

Model loading & caching
----
To ensure system responsiveness and not create any delays with each user interaction, all of the models load once (on startup), and the models are stored in cache for re-use.

Why cache?

Loading large transformer models (for example, Whisper, BART...) is   time-consuming.
Accessing the loaded models with each user action (transcription, summarization...) is a lot quicker.
The application will not download or reinitialize once for every button click
It uses Streamlit’s @st.cache_resource decorator, which guarantees that each model is only loaded once per session and stored in memory.
The cache will be automatically refreshed when the code or environment changes.

Load Whisper encoder (OpenVINO):
```
@st.cache_resource
def load_openvino_encoder():
    core = Core()
    encoder = core.compile_model("whisper-small-openvino/openvino_encoder_model.xml", "CPU")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    return encoder, processor
```



Load Whisper decoder (Hugging Face):
```
@st.cache_resource
def load_hf_decoder():
    return WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

```
Load BART summarizer:
```
@st.cache_resource
def setup_summarizer():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    return tokenizer, model
```

State management - Uses st.session_state to retain transcript, summary and quiz data across tabs for easy transitions between them.

Local + cloud processing - Transcription and summarization occur locally (OpenVINO + Hugging Face) while Quiz and Question Answering leverage cloud API (Aikipedia + Gemini).

File management - All recordings and produced text documents are archived in a dedicated folder (recordings) for ease of access.

Development Approach - Step-by-step wayfinder delivers audio content with ability to record, transcribe, summarize, ask contextual questions, generate quizzes (including multiple choice) without need for technical software expertise.



## 📁 Project Structure

```
├── app.py
├── requirements.txt
├── whisper-small-openvino
├── whisper-small-hf
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
## Demo

https://drive.google.com/file/d/1c84dbWXYKZqSEqa6jJpSZ7481ElS2cNQ/view?usp=sharing


## Screenshots

![Recording](https://github.com/Pardhasaye/Team-Trifusion-Project/blob/0d584b9ef4939121d980cdf52619fa653899e682/screenshots/record.png)
![Transcription](https://github.com/Pardhasaye/Team-Trifusion-Project/blob/0d584b9ef4939121d980cdf52619fa653899e682/screenshots/transcript.png)
![Quiz gen](https://github.com/Pardhasaye/Team-Trifusion-Project/blob/0d584b9ef4939121d980cdf52619fa653899e682/screenshots/quiz%20gen.png)
![Q and a](https://github.com/Pardhasaye/Team-Trifusion-Project/blob/0d584b9ef4939121d980cdf52619fa653899e682/screenshots/qanda.png)
![Summarization](https://github.com/Pardhasaye/Team-Trifusion-Project/blob/0d584b9ef4939121d980cdf52619fa653899e682/screenshots/summary.png)


## OPENVINO optimization
## ⚙️ Whisper Optimization with OpenVINO

To improve transcription performance on CPUs, Trifusion integrates **OpenVINO** for running Whisper's encoder model. This dramatically reduced latency, especially for longer audio files.

### 🧪 Benchmark: 10-Minute Audio Transcription

| Configuration             | Time Taken     |
|---------------------------|----------------|
| Full PyTorch Whisper      | ~6 minutes     |
| Whisper + OpenVINO Encoder| ~2 minutes     |

By using OpenVINO just for the encoder part, we achieved a **3x speedup** on CPU-only systems.

### 🔧 Implementation Details

- Whisper encoder converted to OpenVINO IR format (`.xml` / `.bin`)
- Encoder runs via `openvino.runtime.Core().compile_model(...)`
- Decoder remains in PyTorch for compatibility

```python
encoder = core.compile_model("whisper-small-openvino/openvino_encoder_model.xml", "CPU")
```

### ❌ Why BART Was Not Optimized with OpenVINO

We attempted to convert **BART-Large-CNN** (used for summarization) but encountered:

- **Conversion failures** using Optimum and ONNX
- **Unsupported operations** in decoder layers
- **Dynamic shapes** that OpenVINO couldn't resolve

As a result, the summarizer continues to run in PyTorch.

### ✅ Summary

| Component        | Runtime     | Reason                            |
|------------------|-------------|-----------------------------------|
| Whisper Encoder  | OpenVINO    | Optimized and fully supported     |
| Whisper Decoder  | PyTorch     | More complex, not exportable yet  |
| BART Summarizer  | PyTorch     | Could not convert to OpenVINO     |
| Gemini API (Q&A) | Cloud-hosted| Fast, off-device, no setup needed |

This hybrid strategy allows Trifusion to remain **fast and lightweight**, even on systems without GPUs.

---

## Future scalability


Trifusion lays the foundation for an effective AI-powered classroom assistant. In future versions, we aim to improve its performance, flexibility, and personalization through the following upgrades:

###  Model Optimization
- Full Whisper model conversion to OpenVINO (including decoder) once supported
- Quantization or pruning of Whisper and BART to reduce memory and improve inference time
- Integration with ONNX Runtime as a fallback for devices not supporting OpenVINO
- Explore Whisper-Tiny or Medium for size-performance trade-offs

### Fine-Tuning & Custom AI
- Train/fine-tune Whisper on academic datasets (e.g., lectures, tutorials) for better transcription quality
- Fine-tune BART or switch to T5 with curriculum-specific datasets for more accurate summarization
- Use Gemini’s Function Calling API (when available) to dynamically fetch references or diagrams

### Multi-User System
- Implement user login/signup with role-based dashboards (e.g., Student / Teacher)
- Add user-specific transcript history and personal quiz tracking
- Store data in a backend (Firebase, Supabase, or SQLite)

### Analytics & Feedback
- Track user interaction (e.g., most asked questions, topics)
- Collect feedback on Gemini-generated answers and quiz quality
- Use ratings to improve prompt tuning

### Export & Sharing
- Allow users to export lecture packs (Transcript + Summary + Quiz) as:
  - PDF
  - Email
  - Google Drive uploads
- Add dark/light theme toggle for accessibility

### Language & Accessibility
- Add multi-language support for transcription, summarization, and translation
- Enable text-to-speech playback of transcripts and answers

### Modular Plugin Design
- Design the system so educators can plug in their own AI models or data sources (e.g., textbooks)
- Create a plugin marketplace for summarizers, quiz formats, or voice styles

---
