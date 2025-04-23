# main.py (FastAPI Backend)
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from chatbot.model import ask_llama3
from chatbot.prompt_builder import build_prompt
import sqlite3

app = FastAPI()

# SQLite DB setup
conn = sqlite3.connect("chat_history.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute(
    """CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY,
                    question TEXT,
                    answer TEXT
                 )"""
)
conn.commit()


class Question(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(data: Question):
    try:
        prompt = build_prompt(data.question)
        answer = ask_llama3(prompt)
        cursor.execute(
            "INSERT INTO chat_history (question, answer) VALUES (?, ?)",
            (data.question, answer),
        )
        conn.commit()
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# chatbot/model.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
HF_API_URL = (
    "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
)

headers = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}


def ask_llama3(prompt: str) -> str:
    payload = {"inputs": prompt}
    response = requests.post(HF_API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(
            f"Failed to call LLaMA 3 API: {response.status_code}, {response.text}"
        )

    result = response.json()
    if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
        return result[0]["generated_text"]
    elif isinstance(result, dict) and "generated_text" in result:
        return result["generated_text"]
    else:
        return "Sorry, I couldn't generate an answer."


# main.py (FastAPI Backend)
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from chatbot.model import ask_llama3
from chatbot.prompt_builder import build_prompt
import sqlite3

app = FastAPI()

# SQLite DB setup
conn = sqlite3.connect("chat_history.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute(
    """CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY,
                    question TEXT,
                    answer TEXT
                 )"""
)
conn.commit()


class Question(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(data: Question):
    try:
        prompt = build_prompt(data.question)
        answer = ask_llama3(prompt)
        cursor.execute(
            "INSERT INTO chat_history (question, answer) VALUES (?, ?)",
            (data.question, answer),
        )
        conn.commit()
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# chatbot/model.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
HF_API_URL = (
    "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
)

headers = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}


def ask_llama3(prompt: str) -> str:
    payload = {"inputs": prompt}
    response = requests.post(HF_API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(
            f"Failed to call LLaMA 3 API: {response.status_code}, {response.text}"
        )

    result = response.json()
    if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
        return result[0]["generated_text"]
    elif isinstance(result, dict) and "generated_text" in result:
        return result["generated_text"]
    else:
        return "Sorry, I couldn't generate an answer."
