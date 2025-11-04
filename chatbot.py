from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from openai import OpenAI
import httpx
from flask import Blueprint

chatbot = Blueprint('chatbot', __name__)


load_dotenv()

app = Flask(__name__)

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found. Add to your environment variables or .env file.")

httpx_client = httpx.Client(verify=False)

client = OpenAI(
    api_key=HF_TOKEN,
    base_url="https://router.huggingface.co/v1",
    http_client=httpx_client
)

@app.route("/")
def home():
    return render_template("chatbot.html")

@app.route("/ask", methods=["POST"])
def ask():
    # For form-urlencoded content
    user_question = request.form.get("question")
    if not user_question:
        return jsonify({"answer": "No question provided."}), 400

    try:
        completion = client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-7B-Instruct:hyperbolic",
            messages=[{"role": "user", "content": user_question}]
        )
        bot_reply = completion.choices[0].message.content
    except Exception as e:
        bot_reply = f"Error: {e}"

    return jsonify({"answer": bot_reply})

@chatbot.route('/chat', methods=['POST'])

def reset():
    # If you have any session or conversation context to clear, do it here.
    # For now, just return success
    return jsonify({"status": "reset successful"})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)