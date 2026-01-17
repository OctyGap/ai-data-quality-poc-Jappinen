# web.py
# Minimal Flask chat UI that calls app.handle_prompt() and supports XLSX downloads.

import os
import re
import uuid
from pathlib import Path

from flask import Flask, request, jsonify, render_template_string, send_from_directory
from dotenv import load_dotenv

from src import app as core

load_dotenv()

app = Flask(__name__)

FILE_PATH = os.getenv("FILE_PATH", "data.xlsx")

EXPORT_DIR = Path("exports")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Chat</title>
  <style>
    body{font-family:system-ui;background:#0b0f19;color:#e8eefc;margin:0}
    .wrap{max-width:900px;margin:0 auto;padding:20px}
    .box{background:#121a2a;border:1px solid #22304d;border-radius:14px;padding:16px}
    .msg{white-space:pre-wrap;margin:10px 0;line-height:1.35}
    .me{color:#b9c9ff}
    .bot{color:#e8eefc}
    .row{display:flex;gap:10px;margin-top:12px}
    input{flex:1;padding:12px;border-radius:10px;border:1px solid #22304d;background:#0e1524;color:#e8eefc}
    button{padding:12px 14px;border-radius:10px;border:0;background:#4f7cff;color:white;cursor:pointer}
    a{color:#7aa2ff;text-decoration:underline}
  </style>
</head>
<body>
<div class="wrap">
  <div class="box">
    <div id="chat"></div>
    <div class="row">
      <input
        id="q"
        placeholder="Ask something..."
        autocomplete="off"
        autocapitalize="off"
        autocorrect="off"
        spellcheck="false"
      />
      <button id="send">Send</button>
    </div>
  </div>
</div>
<script>
  const sid = localStorage.getItem("sid") || crypto.randomUUID();
  localStorage.setItem("sid", sid);

  const chat = document.getElementById("chat");
  const q = document.getElementById("q");
  const send = document.getElementById("send");

  function add(role, text){
    const el=document.createElement("div");
    el.className="msg " + (role==="user"?"me":"bot");
    el.textContent=(role==="user"?"You: ":"AI: ")+text;
    chat.appendChild(el);
    chat.scrollTop=chat.scrollHeight;
  }


  async function go(){
    const text=q.value.trim();
    if(!text) return;
    q.value="";
    add("user", text);
    send.disabled=true;

    const res=await fetch("/chat", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ session_id: sid, prompt: text })
    });

    const data=await res.json();
    const a = data.answer;
    add("assistant", (typeof a === "string") ? a : JSON.stringify(a, null, 2));


    send.disabled=false;
    q.focus();
  }

  send.onclick=go;
  q.addEventListener("keydown", (e)=>{ if(e.key==="Enter") go(); });
</script>
</body>
</html>
"""

@app.get("/")
def index():
    return render_template_string(HTML)

@app.post("/chat")
def chat():
    if not os.getenv("OPENAI_API_KEY"):
        return jsonify({"answer": "Missing OPENAI_API_KEY on server."}), 500

    payload = request.get_json(force=True)
    session_id = payload.get("session_id") or str(uuid.uuid4())
    prompt = payload.get("prompt") or ""

    answer = core.handle_prompt(prompt, session_id=session_id, file_path=FILE_PATH)

    # Detect "Created file: <name>.xlsx" (or any <name>.xlsx in the answer)
    m = re.search(r"([A-Za-z0-9_\\-]+\\.xlsx)", answer)
    fname = m.group(1) if m else None

    # If answer ever includes exports/<file>, strip folder for /download/<file>
    if fname and fname.startswith("exports/"):
        fname = fname.split("/", 1)[1]

    return jsonify({"answer": answer, "session_id": session_id, "file": fname})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=False)
