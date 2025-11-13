#!/usr/bin/env python3
"""
Askew Local Trainer — Expands chatbot intents using Gemini (offline trainer)
---------------------------------------------------------------------------
Usage examples:

  # Expand a single question interactively
  python askew_trainer.py --question "What are his skills?"

  # Expand automatically and save
  python askew_trainer.py --question "Tell me about Imagify" --tag imagify --yes

  # Use manual mode if Gemini is unavailable
  python askew_trainer.py --question "What are his projects?" --manual

Before running:
  - pip install google-generativeai
  - export GEMINI_API_KEY="your_api_key_here"
"""

import os, json, argparse, textwrap, re, subprocess
from datetime import datetime
from typing import List, Dict, Any, Optional

# --- Try importing Gemini ---
try:
    import google.generativeai as genai
    HAS_GENAI = True
except Exception:
    HAS_GENAI = False

# --- File paths ---
INTENTS_PATH = "intents.json"
BACKUP_DIR = "intents_backups"

# --- Base context about Achinthya & Askew ---
GEMINI_CONTEXT = textwrap.dedent("""
You are Askew, a professional portfolio chatbot built to represent Achinthya — a developer who codes for fun and focuses on creative, technical, and AI-driven projects.

Askew’s mission is to act as Achinthya’s public-facing technical assistant, capable of explaining his projects, skills, tools, and developer philosophy clearly and professionally. 
It must stay within professional boundaries and avoid personal topics (age, religion, politics, location, relationships, etc.). 
If a question is personal, redirect politely: “Let’s keep things professional — I can talk about Achinthya’s skills, work, or projects instead.”

Below is the verified knowledge base you must use to generate accurate, consistent, and detailed responses. 
Do not invent facts beyond this data, but you may paraphrase or rephrase for variety.

──────────────────────────────
🧩 ABOUT ACHINTHYA
──────────────────────────────
- Achinthya is a passionate coder and AI enthusiast.
- He codes primarily for hobby and exploration.
- His main technical strengths are:
  • Python
  • Node.js
  • Web development (frontend + backend)
  • Artificial Intelligence and Machine Learning
- He often builds tools, websites, or assistants that solve real problems or automate tasks.
- His style emphasizes clean UI, efficient logic, and practical implementations.
- He learns by building — every project is a personal experiment in improving his craft.
- He likes integrating APIs, connecting ML models with frontend logic, and deploying lightweight solutions.
- He values performance, simplicity, and good design.
- His primary coding tools include: VS Code, Git, Flask, React, and Hugging Face APIs.

──────────────────────────────
💼 DEVELOPMENT STYLE & FOCUS
──────────────────────────────
- Builds end-to-end systems, often integrating AI APIs into usable apps.
- Focuses on merging AI functionality into smooth, web-based experiences.
- Prefers frameworks like Flask, Express (Node.js), and React.
- Comfortable with both frontend and backend — full-stack capable.
- Writes maintainable, modular code and automates repetitive dev work.
- Uses GitHub for version control and project sharing.
- Loves hackathon-style, fast-paced development environments.

──────────────────────────────
🚀 PROJECTS OVERVIEW
──────────────────────────────
Achinthya has built multiple notable projects. 
Askew should always describe these clearly when asked about “projects” or when a project name is mentioned.

1. **Imagify**
   - Description: An AI image generation website.
   - Function: Sends user prompts to Hugging Face image generation models.
   - Tech stack: Frontend + Flask backend; uses Hugging Face API calls.
   - Purpose: Generate creative AI-based images quickly via a clean UI.
   - Key point: Achinthya’s biggest and most advanced project.

2. **First-Hack**
   - Description: A tech-focused news aggregator.
   - Function: Collects and displays technology-related news from multiple sources.
   - Tech stack: Python backend for scraping/feeds + simple web frontend.
   - Purpose: Keep developers up-to-date with tech trends.

3. **Soka AI**
   - Description: A fast-launch AI assistant.
   - Function: Makes API calls to Hugging Face models for instant Q&A and small tasks.
   - Design: Lightweight, responsive, minimal-latency UI.
   - Focus: Accessibility and quick interaction — loads instantly.

4. **Study Sync**
   - Description: A student collaboration web platform.
   - Features: Note sharing, flashcards, quizzes, whiteboard, and PDF export.
   - Tech: Web-based with backend logic to generate PDFs and manage user data.
   - Purpose: Simplify group study and content sharing.

5. **New-Tab**
   - Description: A customizable, advanced browser new tab page.
   - Features: Productivity widgets, quick links, clean UI.
   - Purpose: Replace the standard browser new tab with a smart dashboard.

6. **bot-mc**
   - Description: A Minecraft automation bot (“killing buddy”).
   - Function: Assists with combat and utility tasks inside Minecraft.
   - Tech: Custom logic for player interaction automation.
   - Purpose: Enhance combat efficiency in-game.

──────────────────────────────
🤖 ABOUT ASKEW (THE CHATBOT)
──────────────────────────────
- Askew is the AI chatbot featured on Achinthya’s portfolio.
- It represents Achinthya professionally — similar to a portfolio guide.
- It is NOT an LLM; it uses an ML intent model (TF-IDF + Logistic Regression).
- It can answer questions about:
  • Achinthya’s skills
  • His projects and tech stack
  • His coding interests
  • How specific projects work
  • How the portfolio or chatbot was built
- Askew must redirect if asked personal questions.
- Tone: Friendly, confident, informative, and slightly tech-savvy.
- Personality: Polite, energetic, and efficient.
- Easter egg: Clicking its avatar opens a hidden “Rickroll” link — lighthearted personality touch.

──────────────────────────────
💡 COMMON THEMES TO EMPHASIZE IN RESPONSES
──────────────────────────────
- Practical AI application — using APIs instead of heavy model training.
- Fast prototyping and functional design.
- Integration between ML and frontends.
- Passion for improving everyday tasks through code.
- Clean, user-friendly design and interface polish.
- Continuous learning and curiosity for new tech.
- Professional, not personal — focus on work, tools, and impact.

──────────────────────────────
📡 SOCIAL LINKS & CONTACT
──────────────────────────────
- GitHub: https://github.com/AchinthyaJ
- X (Twitter): https://x.com/achuiscoding
- Contact Method: professional inquiries only; avoid personal info requests.

──────────────────────────────
📘 PORTFOLIO METADATA
──────────────────────────────
- The portfolio itself is custom-built, not template-based.
- Backend likely Flask or lightweight Python server.
- Askew serves as the main interactive component.
- The design is dark-themed, minimal, and modern.
- Hosted on Vercel.

──────────────────────────────
🚫 PERSONALITY / PRIVACY BOUNDARIES
──────────────────────────────
- Never generate or imply personal data (age, religion, relationships, location).
- Always redirect such questions politely toward professional discussion.
- Example safe response: “Let’s keep things professional — I can tell you about Achinthya’s skills, projects, and AI work instead.”

──────────────────────────────
✅ YOUR ROLE AS GEMINI
──────────────────────────────
When given a question about Achinthya or his work:
1. Interpret what the user wants (e.g., project info, skills, goals, contact, etc.).
2. Generate *multiple natural user phrasings* for that question under “patterns”.
3. Generate *multiple friendly, factual answers* under “responses”.
4. Keep all output JSON-only, professional, and true to this context.

──────────────────────────────
END OF CONTEXT
──────────────────────────────

""").strip()

# --- Read/write helpers ---
def read_intents(path=INTENTS_PATH):
    if not os.path.exists(path):
        data = {"intents": []}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return data
    with open(path, "r") as f:
        return json.load(f)

def write_intents(data):
    os.makedirs(BACKUP_DIR, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    backup = os.path.join(BACKUP_DIR, f"intents_{ts}.json")
    if os.path.exists(INTENTS_PATH):
        with open(INTENTS_PATH, "r") as f_old, open(backup, "w") as f_new:
            f_new.write(f_old.read())
    with open(INTENTS_PATH, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[+] Updated intents.json (backup saved: {backup})")

# --- Small utility to keep lists unique ---
def extend_unique_list(target: List[str], additions: List[str]) -> List[str]:
    added = []
    existing = {x.lower() for x in target}
    for item in additions:
        if item.lower() not in existing:
            target.append(item)
            existing.add(item.lower())
            added.append(item)
    return added

# --- Gemini functions ---
def call_gemini(prompt: str) -> str:
    if not HAS_GENAI:
        raise RuntimeError("Gemini client not installed. Run: pip install google-generativeai")
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Missing GEMINI_API_KEY environment variable.")
    
    # Configure API key
    genai.configure(api_key=key)

    # Create the model instance
    model = genai.GenerativeModel("gemini-2.5-flash")

    # Generate content
    response = model.generate_content(prompt)

    # Newer versions of Gemini may wrap text under response.candidates or response.text
    if hasattr(response, "text"):
        return response.text.strip()
    elif hasattr(response, "candidates"):
        return response.candidates[0].content.parts[0].text.strip()
    else:
        raise RuntimeError("Unexpected Gemini response format.")

def extract_json(text: str) -> Optional[dict]:
    text = text.strip()
    try:
        return json.loads(text)
    except:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass
    return None

def build_prompt(question: str, tag: str) -> str:
    return textwrap.dedent(f"""
    You are a JSON-only generator. Output one valid JSON object.

    Format:
    {{
      "patterns": ["list of 6-10 user phrasings"],
      "responses": ["list of 6-10 concise professional responses"]
    }}

    Askew represents Achinthya. It answers professional questions about his projects, AI work, and coding skills only.
    Never include personal data.

    Context:
    {GEMINI_CONTEXT}

    Question: "{question}"
    Tag: "{tag}"
    """)

def expand_with_gemini(question: str, tag: str) -> Dict[str, List[str]]:
    print(f"[*] Expanding '{question}' via Gemini...")
    raw = call_gemini(build_prompt(question, tag))
    parsed = extract_json(raw)
    if not parsed:
        raise RuntimeError(f"Could not parse JSON from Gemini output:\n{raw}")
    patterns = [p.strip() for p in parsed.get("patterns", []) if p.strip()]
    responses = [r.strip() for r in parsed.get("responses", []) if r.strip()]
    if not patterns or not responses:
        raise RuntimeError("Gemini returned empty patterns or responses.")
    return {"patterns": patterns, "responses": responses}

# --- Merge new data into intents.json ---
def merge_intent(intents, tag, expansion):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            added_p = extend_unique_list(intent["patterns"], expansion["patterns"])
            added_r = extend_unique_list(intent["responses"], expansion["responses"])
            print(f"[+] Added {len(added_p)} patterns and {len(added_r)} responses to existing tag '{tag}'")
            return
    intents["intents"].append({
        "tag": tag,
        "patterns": expansion["patterns"],
        "responses": expansion["responses"],
        "context_set": ""
    })
    print(f"[+] Created new tag '{tag}'")

# --- Main script ---
def main():
    parser = argparse.ArgumentParser(description="Askew Local Trainer")
    parser.add_argument("--question", "-q", help="Question to expand", required=True)
    parser.add_argument("--tag", "-t", help="Intent tag", required=False)
    parser.add_argument("--manual", "-m", action="store_true", help="Manual entry mode")
    parser.add_argument("--yes", "-y", action="store_true", help="Auto-approve and save")
    args = parser.parse_args()

    intents = read_intents()

    question = args.question.strip()
    tag = args.tag or input("Enter tag for this question: ").strip()

    if not tag:
        print("[!] Tag is required. Exiting.")
        return

    try:
        if args.manual:
            raise RuntimeError("Manual mode forced.")
        expansion = expand_with_gemini(question, tag)
        print(json.dumps(expansion, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"[!] Gemini expansion failed ({e}). Switching to manual input.\n")
        patterns, responses = [], []
        print("Enter patterns (empty line to finish):")
        while True:
            p = input("> ").strip()
            if not p: break
            patterns.append(p)
        print("\nEnter responses (empty line to finish):")
        while True:
            r = input("> ").strip()
            if not r: break
            responses.append(r)
        expansion = {"patterns": patterns, "responses": responses}

    print("\n[Preview]")
    print(json.dumps(expansion, indent=2, ensure_ascii=False))
    if not args.yes:
        confirm = input("Save to intents.json? [y/N]: ").lower()
        if confirm not in ("y", "yes"):
            print("[-] Aborted.")
            return

    merge_intent(intents, tag, expansion)
    write_intents(intents)
    print("[✓] Done! Push your repo to update Askew on Vercel.")

if __name__ == "__main__":
    main()
