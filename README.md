# Askew - A Personal Portfolio Chatbot

Askew is a custom-built, machine learning-powered chatbot designed to act as an intelligent assistant for a personal portfolio. It's built from scratch using Python, Flask, and Scikit-learn, featuring a clean, modern web interface for interaction.

This chatbot, named Askew, is trained on a custom dataset to answer questions about "Achinthya's" projects, skills, and professional background. It uses a classic machine learning pipeline (TF-IDF and Logistic Regression) to understand user intent and provide relevant responses.

## ✨ Features

- **Intent-Based Conversations:** Powered by a custom-trained Scikit-learn model to understand user queries.
- **Personalized Knowledge Base:** Trained on a `intents.json` file to discuss specific projects, skills, and contact information.
- **Conversational Guardrails:** Includes fallback responses and handles inappropriate or personal questions gracefully.
- **Sleek Web Interface:** A responsive, dark-themed chat UI built with HTML, CSS, and vanilla JavaScript.
- **REST API:** A simple Flask backend exposes a `/chat` endpoint for easy integration.
- **Lightweight & Fast:** Avoids large language models for a quick and deterministic user experience.

## 🚀 Technologies Used

- **Backend:** Python, Flask, Scikit-learn
- **Frontend:** HTML, CSS, JavaScript
- **Dataset:** A custom `intents.json` file for training data.

## ⚙️ Getting Started

Follow these instructions to get the chatbot up and running on your local machine.

### 1. Prerequisites

Make sure you have Python 3 installed on your system.

### 2. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-directory>
```

### 3. Install Dependencies

Install the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Run the Application

Start the Flask server by running the `model.py` file.

```bash
python model.py
```

The application will start in debug mode on `http://127.0.0.1:5000`.

### 5. Open in Browser

Open your web browser and navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000) to start chatting with Askew!

### Optional : If you want to improve the dataset for training

1. Open your browser and navigate to [https://aistudio.google.com/](https://aistudio.google.com/) and get a free gemini api key.

2. Import that into your environment using the command
```bash
export GEMINI_API_KEY='YOUR_GEMINI_API_KEY'
```
3. Use gemini to finally generate the patterns and responses using
```bash
python askew_trainer.py -q 'your-question-here' -t 'tag-for-the-list-of-responses'
```
