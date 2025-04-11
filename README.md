# Stadium-Sound

**Stadium Sound** is an AI-powered system with a **3D avatar** designed to make football commentary accessible for deaf and hard-of-hearing fans. **The system features two main capabilities**:

- **Real-Time Sign Language Translation** – Converts live football commentary into sign language using a 3D avatar.
- **Dynamic Backgrounds** – Visual elements that reflect match intensity based on crowd noise and emotional atmosphere.
- **The solution is designed** to be displayed either on stadium screens or through AR glasses with dynamic lens color adjustments, enhancing emotional and immersive engagement.

# Key Features

- **Real-Time Translation**
Translates live football commentary into sign language using Whisper and Neural Machine Translation (NMT).
- **3D Avatar**
A virtual character that signs commentary in real time for improved accessibility and fan engagement.
- **Dynamic Backgrounds**
Background visuals adapt in real time to reflect the excitement level of the match, driven by crowd noise and match events.
- **Multi-Platform Display**
Can be deployed on both stadium screens and AR glasses for flexible and inclusive presentation.

# Technologies Used

- **Whisper** (OpenAI’s speech-to-text model)
- **Transformer-based Neural Machine Translation (NMT)**
- **Unity**
- **ML Classification Model** (for background emotion mapping)
- **MediaPipe** (for gesture recognition and avatar control)
  
# Datasets Used

- **Football Match Commentary** – Collected manually from real football matches
- **General Words Dataset** – Public datasets used to train everyday vocabulary
- **Sign Language Gloss Dataset** – Combination of online resources and custom-created glosses specific to football
- **Emotion & Crowd Noise Data** – Audio datasets from online sources and manually extracted match recordings
# Methodology
![image](https://github.com/user-attachments/assets/42b8b3d3-41cb-4971-b687-3a74fde1cd1b)

# Future Work

- Extend translation to full live commentary with broader vocabulary and context understanding
- Integrate emotion-aware facial expressions and signing speed to reflect urgency
- Add support for multiple regional sign languages (ASL, BSL, Arabic SL, etc.)
- Optimize for AR glasses with real-time processing and user-focused display features
