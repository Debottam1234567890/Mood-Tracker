# Mood-Tracker

**Mood-Tracker** (MoodTrack) is a web-based application that helps users track, analyze, and reflect on their emotions. Using AI-driven sentiment analysis and interactive mood history, the app provides insights into emotional wellbeing and helps users manage their moods effectively.

---

## Features

* **Mood Input & Analysis** – Enter your thoughts or feelings, and the app uses AI to analyze your mood.
* **Sentiment Scoring** – Displays polarity (positive/negative/neutral) and subjectivity.
* **Mood Feedback** – Shows emojis and helpful feedback based on your emotions.
* **Mood History** – Saves and visualizes your last 10 mood entries locally.
* **MoodMate Chatbot** – A supportive AI companion that offers emotional support and coping strategies.
* **Responsive UI** – Clean, modern design that works beautifully on mobile and desktop.

---

## Tech Stack

* **Backend**: Python, Flask
* **Frontend**: HTML, CSS, JavaScript
* **AI Sentiment Analysis**: TextBlob
* **Chatbot AI**: Google Gemini API
* **Storage**: Browser LocalStorage

---

## Installation

1. **Clone the repository:**

```
git clone https://github.com/yourusername/mood-tracker.git
cd mood-tracker
```

2. **Install dependencies:**

```
pip install -r requirements.txt
```

3. **Set Gemini API key:**

```
export GEMINI_API_KEY="your_api_key_here"
```

4. **Run the app:**

```
python3 endpoints.py
```

5. **Open in browser:**

```
http://127.0.0.1:5000/tracker
```

---

## Usage

* Go to the **Tracker** page.
* Write about your mood or describe your day.
* Click **Analyze My Mood** to view results.
* View your **Mood History** to reflect on your emotional trends.
* Optional: Chat with **MoodMate** for emotional support.

---

## Dependencies

* Flask
* TextBlob
* Requests

Install them all with:

```
pip install -r requirements.txt
```

---

**🌱 Mood-Tracker** – Track your feelings, understand your emotions, and grow emotionally.
Available at: 
