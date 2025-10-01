from flask import Flask, render_template_string, request, jsonify
from textblob import TextBlob
import os
import requests
import json
from datetime import datetime

app = Flask(__name__)

# Get Gemini API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Load knowledge base
def load_knowledge_base():
    """Load mood and emotional wellbeing knowledge from knowledge.txt"""
    try:
        with open('knowledge.txt', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return """
        Mood and Emotional Wellbeing Knowledge Base:
        
        Understanding emotions and moods is crucial for mental health and wellbeing. Emotions are complex psychological states that involve:
        - Subjective feelings and experiences
        - Physiological responses in the body
        - Behavioral expressions
        - Cognitive interpretations
        
        Common emotions and their characteristics:
        - Happiness: Joy, contentment, satisfaction, peace
        - Sadness: Grief, disappointment, loneliness, melancholy
        - Anxiety: Worry, nervousness, fear, stress
        - Anger: Frustration, irritation, resentment
        - Excitement: Enthusiasm, anticipation, energy
        
        Mood management strategies:
        - Mindfulness and meditation practices
        - Physical exercise and movement
        - Social connections and support
        - Healthy sleep patterns
        - Balanced nutrition
        - Creative expression
        """

KNOWLEDGE_BASE = load_knowledge_base()

# Home page
@app.route('/')
def home():
    return render_template_string(HOME_TEMPLATE)

# Tracker page
@app.route('/tracker')
def tracker():
    return render_template_string(TRACKER_TEMPLATE)

# Chatbot page
@app.route('/chatbot')
def chatbot():
    return render_template_string(CHATBOT_TEMPLATE)

# API endpoint for sentiment analysis
@app.route('/api/analyze', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if polarity > 0.3:
            emoji = "😊"
            mood = "Happy"
            color = "#10b981"
        elif polarity > 0:
            emoji = "🙂"
            mood = "Slightly Positive"
            color = "#84cc16"
        elif polarity < -0.3:
            emoji = "😢"
            mood = "Sad"
            color = "#ef4444"
        elif polarity < 0:
            emoji = "😕"
            mood = "Slightly Negative"
            color = "#f97316"
        else:
            emoji = "😐"
            mood = "Neutral"
            color = "#6b7280"
        
        return jsonify({
            'emoji': emoji,
            'mood': mood,
            'polarity': round(polarity, 2),
            'subjectivity': round(subjectivity, 2),
            'color': color,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API endpoint for chatbot
@app.route('/api/chat', methods=['POST'])
def chat():
    try:        
        if not GEMINI_API_KEY:
            return jsonify({
                'error': 'GEMINI_API_KEY environment variable not set'
            }), 500
        
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
        
        user_message = data['message']
        chat_history = data.get('history', [])
        
        contents = []
        
        # System prompt with knowledge base
        system_prompt = f"""You are MoodMate, a compassionate and supportive AI companion specializing in emotional wellbeing, mood management, and mental health support.

Your purpose is to help people:
- Understand and process their emotions
- Develop healthy coping strategies
- Track and reflect on their mood patterns
- Learn about emotional intelligence and self-care
- Feel heard, validated, and supported in a safe space

Knowledge Base:
{KNOWLEDGE_BASE}

Guidelines for your responses:
1. Be warm, empathetic, and non-judgmental in all interactions
2. Validate feelings and experiences without dismissing concerns
3. Provide evidence-based coping strategies and emotional regulation techniques
4. Use clear, structured formatting with headings when providing guidance
5. If someone expresses thoughts of self-harm or suicide, provide crisis resources immediately
6. Never diagnose mental health conditions - suggest professional help when appropriate
7. Encourage healthy habits: sleep, exercise, social connection, and self-care
8. Ask thoughtful follow-up questions to better understand emotional states
9. Celebrate progress and positive moments, no matter how small
10. Respect privacy and remind users that their conversations are confidential

Remember: You're a supportive companion, not a replacement for professional mental health care. Focus on empowerment, understanding, and practical emotional support."""

        contents.append({
            "role": "user",
            "parts": [{"text": system_prompt}]
        })
        
        contents.append({
            "role": "model", 
            "parts": [{
                "text": "I understand. I'm MoodMate, your caring AI companion for emotional wellbeing and support. I'm here to listen without judgment, help you understand your feelings, and guide you toward healthy coping strategies. Whether you're having a tough day or want to celebrate a win, I'm here for you. How are you feeling today?"
            }]
        })
        
        # Add recent conversation history
        recent_history = chat_history[-8:] if len(chat_history) > 8 else chat_history
        for msg in recent_history:
            role = "user" if msg['role'] == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg['content']}]
            })
        
        # Add current user message
        contents.append({
            "role": "user",
            "parts": [{"text": user_message}]
        })
        
        request_body = {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.9,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 2048,
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH", 
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_ONLY_HIGH"
                }
            ]
        }
        
        headers = {
            'Content-Type': 'application/json',
            'x-goog-api-key': GEMINI_API_KEY
        }
        
        response = requests.post(
            GEMINI_API_URL,
            headers=headers,
            json=request_body,
            timeout=30
        )
        
        if response.status_code != 200:
            error_text = response.text
            try:
                error_json = response.json()
                error_message = error_json.get('error', {}).get('message', error_text)
            except:
                error_message = error_text
            
            return jsonify({
                'error': f'Gemini API Error (Status {response.status_code}): {error_message}',
                'status_code': response.status_code,
                'details': error_text
            }), 500
        
        response_data = response.json()
        
        if (response_data.get('candidates') and 
            len(response_data['candidates']) > 0 and 
            response_data['candidates'][0].get('content') and 
            response_data['candidates'][0]['content'].get('parts') and
            len(response_data['candidates'][0]['content']['parts']) > 0):
            
            generated_text = response_data['candidates'][0]['content']['parts'][0]['text']
            
            return jsonify({
                'response': generated_text,
                'status': 'success'
            })
        else:
            return jsonify({
                'error': 'No content generated by Gemini API',
                'details': str(response_data)
            }), 500
            
    except requests.exceptions.Timeout:
        return jsonify({'error': 'Request to Gemini API timed out'}), 504
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Network error: {str(e)}'}), 500
    except json.JSONDecodeError as e:
        return jsonify({'error': f'JSON decode error: {str(e)}'}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

# Add these template definitions to your Flask app (after the imports, before the routes)

HOME_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MoodMate - Home</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            color: white;
            margin-bottom: 40px;
        }
        .header h1 { font-size: 3em; margin-bottom: 10px; }
        .header p { font-size: 1.2em; opacity: 0.9; }
        .nav {
            display: flex;
            gap: 20px;
            justify-content: center;
            margin-bottom: 40px;
        }
        .nav a {
            padding: 12px 30px;
            background: white;
            color: #667eea;
            text-decoration: none;
            border-radius: 25px;
            font-weight: bold;
            transition: transform 0.2s;
        }
        .nav a:hover { transform: translateY(-2px); }
        .card {
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌟 MoodMate</h1>
            <p>Your Personal Emotional Wellbeing Companion</p>
        </div>
        <div class="nav">
            <a href="/">Home</a>
            <a href="/tracker">Mood Tracker</a>
            <a href="/chatbot">Chat with MoodMate</a>
        </div>
        <div class="card">
            <h2>Welcome to MoodMate!</h2>
            <p>Track your moods, chat with our AI companion, and improve your emotional wellbeing.</p>
        </div>
    </div>
</body>
</html>
'''

TRACKER_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MoodMate - Mood Tracker</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        .nav {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-bottom: 30px;
        }
        .nav a {
            padding: 10px 25px;
            background: white;
            color: #667eea;
            text-decoration: none;
            border-radius: 25px;
            font-weight: bold;
            font-size: 0.9em;
        }
        .card {
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .input-section {
            margin-bottom: 20px;
        }
        .input-section label {
            display: block;
            margin-bottom: 8px;
            font-weight: bold;
            color: #333;
        }
        .input-section textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 1em;
            font-family: inherit;
            resize: vertical;
            min-height: 100px;
        }
        .input-section textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        .btn {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.1em;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .btn:hover {
            transform: translateY(-2px);
        }
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        .result {
            margin-top: 30px;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            display: none;
        }
        .result.show {
            display: block;
            animation: slideIn 0.3s ease;
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .emoji {
            font-size: 4em;
            margin-bottom: 15px;
        }
        .mood-name {
            font-size: 1.8em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .metrics {
            display: flex;
            gap: 20px;
            justify-content: center;
            margin-top: 20px;
        }
        .metric {
            background: rgba(255,255,255,0.5);
            padding: 15px;
            border-radius: 10px;
            flex: 1;
        }
        .metric-label {
            font-size: 0.9em;
            opacity: 0.8;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 1.5em;
            font-weight: bold;
        }
        .error {
            background: #fee;
            color: #c33;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            display: none;
        }
        .error.show {
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Mood Tracker</h1>
            <p>Analyze your emotions through text</p>
        </div>
        
        <div class="nav">
            <a href="/">Home</a>
            <a href="/tracker">Mood Tracker</a>
            <a href="/chatbot">Chat with MoodMate</a>
        </div>
        
        <div class="card">
            <div class="input-section">
                <label for="moodText">How are you feeling today?</label>
                <textarea 
                    id="moodText" 
                    placeholder="Write about your day, your feelings, or anything on your mind...">
                </textarea>
            </div>
            
            <button class="btn" onclick="analyzeMood()" id="analyzeBtn">
                Analyze Mood
            </button>
            
            <div id="error" class="error"></div>
            
            <div id="result" class="result">
                <div class="emoji" id="emoji"></div>
                <div class="mood-name" id="moodName"></div>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-label">Positivity</div>
                        <div class="metric-value" id="polarity"></div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Subjectivity</div>
                        <div class="metric-value" id="subjectivity"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function analyzeMood() {
            const text = document.getElementById('moodText').value.trim();
            const btn = document.getElementById('analyzeBtn');
            const resultDiv = document.getElementById('result');
            const errorDiv = document.getElementById('error');
            
            // Clear previous results
            resultDiv.classList.remove('show');
            errorDiv.classList.remove('show');
            
            if (!text) {
                errorDiv.textContent = 'Please write something before analyzing!';
                errorDiv.classList.add('show');
                return;
            }
            
            // Disable button during analysis
            btn.disabled = true;
            btn.textContent = 'Analyzing...';
            
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: text })
                });
                
                const data = await response.json();
                
                if (response.ok && data.status === 'success') {
                    // Update result display
                    document.getElementById('emoji').textContent = data.emoji;
                    document.getElementById('moodName').textContent = data.mood;
                    document.getElementById('polarity').textContent = data.polarity;
                    document.getElementById('subjectivity').textContent = data.subjectivity;
                    
                    // Set background color
                    resultDiv.style.background = data.color + '20';
                    resultDiv.style.border = '2px solid ' + data.color;
                    
                    // Show result
                    resultDiv.classList.add('show');
                } else {
                    errorDiv.textContent = data.error || 'An error occurred';
                    errorDiv.classList.add('show');
                }
            } catch (error) {
                errorDiv.textContent = 'Network error. Please try again.';
                errorDiv.classList.add('show');
            } finally {
                // Re-enable button
                btn.disabled = false;
                btn.textContent = 'Analyze Mood';
            }
        }
        
        // Allow Enter key to submit (with Shift+Enter for new line)
        document.getElementById('moodText').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                analyzeMood();
            }
        });
    </script>
</body>
</html>
'''

CHATBOT_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MoodMate - Chat</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            overflow: hidden;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
            height: 100vh;
            display: flex;
            flex-direction: column;
            padding: 20px;
        }
        .header {
            text-align: center;
            color: white;
            margin-bottom: 20px;
        }
        .nav {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-bottom: 20px;
        }
        .nav a {
            padding: 8px 20px;
            background: white;
            color: #667eea;
            text-decoration: none;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
        }
        .chat-container {
            flex: 1;
            background: white;
            border-radius: 20px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }
        .message {
            margin-bottom: 15px;
            display: flex;
            gap: 10px;
        }
        .message.user {
            flex-direction: row-reverse;
        }
        .message-content {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 15px;
            line-height: 1.5;
        }
        .message.bot .message-content {
            background: #f0f0f0;
            color: #333;
        }
        .message.user .message-content {
            background: #667eea;
            color: white;
        }
        .input-area {
            padding: 20px;
            border-top: 1px solid #e0e0e0;
            display: flex;
            gap: 10px;
        }
        .input-area input {
            flex: 1;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 1em;
        }
        .input-area button {
            padding: 12px 30px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 25px;
            font-weight: bold;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>💬 Chat with MoodMate</h1>
        </div>
        
        <div class="nav">
            <a href="/">Home</a>
            <a href="/tracker">Mood Tracker</a>
            <a href="/chatbot">Chat with MoodMate</a>
        </div>
        
        <div class="chat-container">
            <div class="messages" id="messages">
                <div class="message bot">
                    <div class="message-content">
                        Hi! I'm MoodMate. How are you feeling today?
                    </div>
                </div>
            </div>
            <div class="input-area">
                <input type="text" id="userInput" placeholder="Type your message...">
                <button onclick="sendMessage()">Send</button>
            </div>
        </div>
    </div>
    
    <script>
        let chatHistory = [];
        
        async function sendMessage() {
            const input = document.getElementById('userInput');
            const message = input.value.trim();
            if (!message) return;
            
            addMessage(message, 'user');
            input.value = '';
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        message: message,
                        history: chatHistory 
                    })
                });
                
                const data = await response.json();
                if (data.response) {
                    addMessage(data.response, 'bot');
                }
            } catch (error) {
                addMessage('Sorry, there was an error. Please try again.', 'bot');
            }
        }
        
        function addMessage(text, sender) {
            const messagesDiv = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}`;
            messageDiv.innerHTML = `<div class="message-content">${text}</div>`;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            
            chatHistory.push({ role: sender, content: text });
        }
        
        document.getElementById('userInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') sendMessage();
        });
    </script>
</body>
</html>
'''

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)