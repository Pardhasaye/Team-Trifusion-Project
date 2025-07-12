import streamlit as st
st.set_page_config(page_title="AI Classroom Assistant", layout="wide")

import os
from datetime import datetime
import soundfile as sf
import numpy as np
import librosa
import torch
from pvrecorder import PvRecorder
from transformers import WhisperProcessor, WhisperForConditionalGeneration, BartTokenizer, BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from openvino.runtime import Core
import google.generativeai as genai

# --- Config ---
SAVE_DIR = "recordings"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Google Gemini API Key ---
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")

# --- Load Models ---
@st.cache_resource
def load_encoder_decoder():
    core = Core()
    encoder = core.compile_model("whisper-small-openvino/openvino_encoder_model.xml", "CPU")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    decoder = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    return encoder, processor, decoder

@st.cache_resource
def setup_summarizer():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    return tokenizer, model

encoder_model, transcriber_processor, decoder_model = load_encoder_decoder()
summary_tokenizer, summary_model = setup_summarizer()

# --- Gemini Helper ---
def gemini_answer(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gemini error: {str(e)}"

# --- Audio Recorder (Stable Version) ---
def micThing(index):
    if "is_recording" not in st.session_state:
        st.session_state.is_recording = False
    if "audio_frames" not in st.session_state:
        st.session_state.audio_frames = []

    if not st.session_state.is_recording:
        if st.button("🎙 Start Recording"):
            try:
                st.session_state.recorder = PvRecorder(device_index=index, frame_length=512)
                st.session_state.recorder.start()
                st.session_state.audio_frames = []
                st.session_state.is_recording = True
                st.session_state.just_started = True
                st.success("🎤 Recording started...")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error starting recorder: {e}")
        return None

    # skip frame collection immediately after rerun
    if st.session_state.get("just_started"):
        st.session_state.just_started = False
    else:
        try:
            frame = st.session_state.recorder.read()
            st.session_state.audio_frames.append(frame)
        except Exception as e:
            st.warning(f"Reading error: {e}")

    st.info("Recording... press Stop when done.")
    if st.button("⏹ Stop Recording"):
        try:
            st.session_state.recorder.stop()
            st.session_state.recorder.delete()
        except:
            pass
        st.session_state.is_recording = False

        audio_np = np.array(st.session_state.audio_frames, dtype=np.int16).flatten()
        fname = f"recorded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        path = os.path.join(SAVE_DIR, fname)
        sf.write(path, audio_np, samplerate=16000)
        st.success(f"✅ Recording saved at {path}")
        return path

    return None

# --- Transcription ---
@st.cache_data
def transcribe_with_openvino(filepath, chunk_length_sec=30):
    audio, sr = librosa.load(filepath, sr=16000)
    chunk_samples = chunk_length_sec * sr
    total_samples = len(audio)
    transcriptions = []

    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        chunk = audio[start:end]
        input_features = transcriber_processor(chunk, sampling_rate=16000, return_tensors="pt").input_features
        encoder_out = encoder_model(inputs={"input_features": input_features.numpy()})[encoder_model.outputs[0]]
        encoder_hidden_states = torch.tensor(encoder_out, dtype=torch.float32)
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)

        forced_decoder_ids = transcriber_processor.get_decoder_prompt_ids(language="en", task="transcribe")
        generated_ids = decoder_model.generate(
            encoder_outputs=encoder_outputs,
            forced_decoder_ids=forced_decoder_ids,
            max_length=448,
            do_sample=False
        )
        transcription = transcriber_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        transcriptions.append(transcription)

    return " ".join(transcriptions)

# --- Summarization ---
def summarize_text(text, max_chunk_length=1024):
    summaries = []
    chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]

    for chunk in chunks:
        inputs = summary_tokenizer(chunk, return_tensors="pt", truncation=True, max_length=1024)
        summary_ids = summary_model.generate(
            **inputs,
            max_length=130,
            min_length=30,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        summary = summary_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    return " ".join(summaries)

# --- UI ---
st.title("🎓 AI-Powered Classroom Assistant")

page = st.sidebar.radio("Navigation", ["🎙 Record & Transcribe", "📝 Summarize", "❓ Question Answering", "🧪 Quiz Generator"])

# --- Record & Transcribe ---
if page == "🎙 Record & Transcribe":
    devices = PvRecorder.get_available_devices()
    mic_choice = st.selectbox("🎤 Select Microphone", devices, index=0)
    mic_index = devices.index(mic_choice)

    audio_path = micThing(mic_index)
    if audio_path:
        st.session_state.audio_path = audio_path
        st.audio(audio_path, format="audio/wav")

    st.markdown("### 📂 Saved Recordings")
    files = [f for f in os.listdir(SAVE_DIR) if f.endswith(".wav")]
    if files:
        selected = st.selectbox("Select file:", files)
        if selected and st.button("🗕 Load Selected"):
            st.session_state.audio_path = os.path.join(SAVE_DIR, selected)
            st.audio(st.session_state.audio_path, format="audio/wav")
    else:
        st.info("No recordings found.")

    uploaded = st.file_uploader("📤 Upload Audio", type=["wav"])
    if uploaded:
        filename = f"uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        filepath = os.path.join(SAVE_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(uploaded.read())
        st.session_state.audio_path = filepath
        st.success("Uploaded successfully!")
        st.audio(filepath, format="audio/wav")

    if "audio_path" in st.session_state and st.button("📝 Transcribe"):
        with st.spinner("Transcribing..."):
            text = transcribe_with_openvino(st.session_state.audio_path)
            st.session_state.transcribed_text = text
            with open(st.session_state.audio_path.replace(".wav", ".txt"), "w") as f:
                f.write(text)
        st.subheader("📄 Transcript")
        st.write(text)
        st.download_button("⬇ Download Transcript", text, file_name="transcript.txt")

# --- Summarize ---
if page == "📝 Summarize" and "transcribed_text" in st.session_state:
    st.write("### 📄 Transcript")
    st.write(st.session_state.transcribed_text)

    if st.button("🔍 Summarize"):
        with st.spinner("Summarizing..."):
            summary = summarize_text(st.session_state.transcribed_text)
            st.session_state.summary = summary
        st.subheader("📝 Summary")
        st.write(summary)
        st.download_button("⬇ Download Summary", summary, file_name="summary.txt")

# --- Q&A ---
if page == "❓ Question Answering" and "transcribed_text" in st.session_state:
    st.write("### 📄 Transcript")
    st.write(st.session_state.transcribed_text)

    q = st.text_input("Ask a question:")
    if st.button("💬 Get Answer") and q:
        with st.spinner("Gemini thinking..."):
            prompt = f"""Based on the transcript below, answer this question:

Transcript:
{st.session_state.transcribed_text}

Question: {q}"""
            answer = gemini_answer(prompt)
            st.subheader("💡 Answer")
            st.write(answer)

# --- Quiz ---
if page == "🧪 Quiz Generator" and "transcribed_text" in st.session_state:
    st.write("### 📄 Transcript")
    st.write(st.session_state.transcribed_text)

    if st.button("📋 Generate Quiz"):
        with st.spinner("Generating quiz with Gemini..."):
            prompt = f"""Generate 5 multiple-choice quiz questions with 4 options each (A–D) based on the following transcript.
Mention the correct answer after each question.

Transcript:
{st.session_state.transcribed_text}
"""
            quiz = gemini_answer(prompt)
            st.session_state.quiz = quiz
        st.subheader("🧠 Quiz")
        st.text(quiz)
        st.download_button("⬇ Export Quiz", quiz, file_name="quiz.txt")
