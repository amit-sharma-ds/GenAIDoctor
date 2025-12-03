# MediFusionAI — AI Voice Doctor

**MediFusion AI** is an AI-powered voice doctor that combines ElevenLabs, Google Cloud Vertex AI, and LLaMA 3 Meta Vision to provide fast, natural, voice-driven medical consultations.

---

## Inspiration
In many regions, patients face long waiting times and limited access to healthcare. MediFusion AI bridges this gap by enabling real-time, voice-driven interactions between patients and an AI doctor, providing medical reasoning, symptom analysis, and disease prediction.

---

## What It Does
MediFusion AI allows users to talk naturally and get intelligent, medically relevant responses. It:

- Listens to patient speech using **Whisper** + **ElevenLabs STT** with noise reduction  
- Analyzes symptoms and medical images using **Google Cloud Vertex AI** + **LLaMA 3 Meta Vision (Groq API)**  
- Responds like a human doctor using **ElevenLabs TTS** for natural medical voice output  
- Predicts diseases using ML models for **Diabetes**, **Tumor Detection**, and **Heart Disease**  
- Provides a fully voice-based consultation through a **Gradio conversational UI**

---

## Key Features
- Voice-driven healthcare consultation  
- Real-time multimodal AI (Vision + Voice + Text)  
- Medical image analysis using LLaMA Vision  
- ML-based disease prediction (Diabetes, Tumor, Heart)  
- Natural conversational doctor voice using ElevenLabs TTS  
- Lightweight and responsive UI with Gradio  

---

## Tech Stack

| Category      | Tools                                                                 |
|---------------|----------------------------------------------------------------------|
| Cloud AI      | Google Cloud Vertex AI, Gemini, Groq API                             |
| Voice AI      | ElevenLabs STT & TTS, Whisper, gTTS, Google Speech-to-Text          |
| LLM           | LLaMA 3 Meta Vision                                                  |
| Frontend      | Gradio                                                                |
| ML Models     | Diabetes, Heart Disease, Tumor CNN                                   |
| Deployment    | Streamlit / Hugging Face / Google Cloud Run                          |
| Languages     | Python, NumPy, OpenCV, ONNX                                         |

---

## How We Built It
1. **AI Brain** — Integrated Google Cloud Vertex AI + Groq LLaMA 3 Vision for reasoning and image analysis  
2. **Voice of the Patient** — Noise-filtered voice recording and transcription using Whisper + ElevenLabs STT + Google Speech-to-Text  
3. **AI Doctor Voice** — Response synthesis using ElevenLabs TTS and gTTS for natural voice output  
4. **Disease Prediction Engine** — Integrated ML & DL models: Diabetes, Tumor (CNN), and Heart Disease  
5. **Real-Time UI** — Built a Gradio VoiceBot interface for full speech interaction  

---

## Challenges
- Managing latency between multimodal model and speech generation  
- Synchronizing STT, reasoning, TTS, and UI in real time  
- Handling noisy audio input and transcription accuracy  

---

## Accomplishments
- Real-time, voice-based intelligent medical consultation system  
- Successfully integrated ElevenLabs + Google Cloud + Groq multimodal AI  
- Achieved smooth, natural conversational experience  

---

## What We Learned
- Multimodal AI systems (Vision + Voice + ML)  
- Cloud deployment and real-time inference pipelines  
- Building scalable conversational AI healthcare solutions  

---

## What’s Next
- Multilingual voice consultation  
- Medical reports + symptom history tracking  
- Mobile app version and Firebase integration  
- End-to-end patient analytics dashboard  

---

## Why It Fits ElevenLabs Challenge
MediFusion AI delivers a fully conversational healthcare experience using ElevenLabs voice technologies combined with Google Cloud Vertex AI and multimodal reasoning. Users interact entirely through speech, making medical assistance fast, accessible, and natural.

---

## Submission Link
🔗 **GitHub Repository (Source Code):**  
[https://github.com/amit-sharma-ds/GenAIHeathcare](https://github.com/amit-sharma-ds/GenAIHeathcare)

---

## Built With
- deep-learning  
- docker  
- elevenlabs  
- ffmpeg  
- gradio  
- groq  
- groq-cloud  
- gtts  
- huggingface  
- llama-3-vision  
- machine-learning  
- openai-whisper  
- pyaudio  
- python  
- speech-to-text  
- vs-code  

---

## Try It Out
[Hugging Face App]([https://huggingface.co](https://huggingface.co/spaces/AmitSharma99/MediFusionAI))
