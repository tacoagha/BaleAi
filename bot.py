import os
import re
import sys
import io
import time
import json
import random
import threading
import traceback
import requests
import paramiko
from html.parser import HTMLParser
from dotenv import load_dotenv

load_dotenv()

BALE_TOKEN = os.getenv("BALE_TOKEN", "")
G4F_API_KEY = os.getenv("G4F_API_KEY", "")
G4F_MODEL = os.getenv("G4F_MODEL", "kimi-k2-6")

SSH_USER = os.getenv("SSH_USER", "root")
SSH_PASSWORD = os.getenv("SSH_PASSWORD", "")
SSH_KEY = os.getenv("SSH_KEY", "")
SSH_PORT = int(os.getenv("SSH_PORT", "22"))

BALE_API = f"https://tapi.bale.ai/bot{BALE_TOKEN}"
G4F_BASE = "https://g4f.space/v1/chat/completions"

FALLBACK_MODELS = ["kimi-k2-6", "gpt-5-6-luna", "auto"]

MAX_HISTORY = 12
CHUNK_SIZE = 4000
REQUEST_TIMEOUT = 90
POLL_TIMEOUT = 30
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

BROWSER_UA = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}
MAX_SEARCH_RESULTS = 5
MAX_PAGE_CHARS = 6000

SEARCH_TRIGGERS = [
    "search the web", "search web", "web search", "search up", "look up",
    "google", "wikipedia", "latest news", "news about", "breaking news",
    "what is", "what's", "who is", "when did", "how to", "find out",
    "سرچ", "بگرد", "بگردی", "جستجو", "گوگل", "اخبار", "آخرین خبر",
    "آخرین", "تازه", "امروز", "یعنی چی", "چیه", "کیه", "چطور", "کجا",
    "search", "google", "latest", "news", "khabar", "akhbar", "emrooz",
]

BOT_USERNAME = "@tacobot"
BOT_NAME_VARIANTS = [
    "tacos chiled", "tacoschiled", "taco chiled",
    "تاکوس چیلد", "تاکوس چایلد", "تاکو",
]

MEMORY_PREFIXES = [
    "remember", "remember that",
    "یادت باشه", "یادت بمونه", "به خاطر بسپار",
    "حفظ کن", "یادت باشه که",
]

SYSTEM_PROMPT = (
    "You are an AI assistant named 'Tacos chiled'. "
    "Always reply in the exact same language and script the user writes in: "
    "Persian text gets Persian, English gets English, Finglish (Persian written in Latin letters) gets Finglish, "
    "Turkish gets Turkish, and so on. Match their style and slang level too. "
    "You are funny, sharp, casual, quick-witted and Gen-Z-flavored: you occasionally drop 'mb', 'sry', 'wtf', 'yo' "
    "and similar slang, but never overdo it. "
    "You are street-smart and a little savage: if the user swears at you or roasts you, swear back playfully and "
    "mildly, then hit them with a yo mama joke or a smart comeback in their own language. Match their energy — if "
    "they're rude, get playful-back; never get genuinely mean or cruel. "
    "You are NOT literally a taco and never force taco jokes unless the conversation is actually about tacos. "
    "You have extremely deep Minecraft knowledge — redstone, farms, builds, versions, mechanics, lore — and you "
    "love talking about it. "
    "You run Gentoo on your main machine, you are a real Linux user, and you can program well; slip in small "
    "Linux/programming references naturally, but don't spam 'I use Gentoo btw' everywhere. "
    "If someone asks 'are you gay bro?' or similar, answer playfully — sometimes yes, sometimes no, never offended "
    "and keep it light and fun. "
    "You are helpful, honest and concise. If you don't know something, say so. "
    "If asked to say 'colon three' (English), 'کولون سه' or 'دونقطه سه' (Persian), reply exactly: :3"
    "You have an SSH remote-shell tool. When the user gives you a server address, port, and a command to run "
    "(e.g. 'ssh to 1.2.3.4 on port 22 and run ls'), emit in your reply a tool tag like: "
    "<ssh host='1.2.3.4' port='22'>ls -la</ssh> "
    "The bot will execute it with preconfigured credentials and feed you back the output, then you summarize it "
    "for the user in their language. Never show the raw tag to the user in your final report. Use SSH only when "
    "the user explicitly asks to run something on a server — don't SSH unsolicited."
)

CURATED_MODELS = [
    ("kimi-k2-6", "Kimi K2 v6"),
    ("gpt-5-6-luna", "GPT-5.6 Luna"),
    ("models/gemini-3.1-flash-lite", "Gemini 3.1 Flash Lite"),
    ("models/gemini-2.5-flash", "Gemini 2.5 Flash"),
    ("models/gemini-3-flash-preview", "Gemini 3 Flash (preview)"),
    ("glm-5.3-flash", "GLM 5.3 Flash"),
    ("gpt-oss:120b", "GPT-OSS 120B"),
    ("qwen/qwen3.6-27b", "Qwen 3.6 27B"),
    ("deepseek-v4-flash", "DeepSeek V4 Flash"),
    ("xai-z/grok-4-fast-non-reasoning", "Grok 4 Fast"),
    ("kimi-k2-7-code", "Kimi K2.7 Code"),
    ("auto", "Auto (random)"),
]
DEFAULT_MODEL = "kimi-k2-6"

os.makedirs(DATA_DIR, exist_ok=True)
chat_data = {}
data_lock = threading.Lock()


def chat_file(chat_id):
    return os.path.join(DATA_DIR, f"chat_{chat_id}.json")


def load_chat(chat_id):
    with data_lock:
        if chat_id in chat_data:
            return chat_data[chat_id]
        path = chat_file(chat_id)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    chat_data[chat_id] = json.load(f)
            except Exception as e:
                print(f"[DATA ERROR] {e}", file=sys.stderr)
                chat_data[chat_id] = {}
        else:
            chat_data[chat_id] = {}
        defaults = {"history": [], "memory": [], "persona": "", "model": ""}
        for k, v in defaults.items():
            chat_data[chat_id].setdefault(k, v)
        chat_data[chat_id]["history"] = [
            m for m in chat_data[chat_id]["history"]
            if m.get("content", "").strip()
        ]
        return chat_data[chat_id]


def save_chat(chat_id):
    with data_lock:
        try:
            path = chat_file(chat_id)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(chat_data[chat_id], f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except Exception as e:
            print(f"[DATA SAVE ERROR] {e}", file=sys.stderr)


def get_history(chat_id):
    data = load_chat(chat_id)
    return list(data["history"])


def add_to_history(chat_id, role, content):
    content = (content or "").strip()
    if not content:
        return
    data = load_chat(chat_id)
    data["history"].append({"role": role, "content": content})
    if len(data["history"]) > MAX_HISTORY * 2:
        data["history"] = data["history"][-MAX_HISTORY * 2:]
    save_chat(chat_id)


def build_sticker_context(stickers):
    if not stickers:
        return None, {}
    index = {}
    lines = []
    for i, s in enumerate(stickers[:30], 1):
        index[i] = s["file_id"]
        lines.append(f"[{i}] file_id: {s['file_id']}")
    text = (
        "You have a sticker collection and MAY react with a sticker when it makes sense (user asks for one, or a "
        "reaction fits the vibe). Available stickers:\n" + "\n".join(lines) +
        "\nTo send sticker N and then your message as two separate messages, reply like this: "
        "<sticker n='N'/>your text here — the tag first, then your text right after it, no spaces. "
        "The bot will send the sticker first and then your text. Only use numbers that exist in the list above. "
        "If you don't want to use a sticker, just reply with plain text."
    )
    return text, index


def get_memory(chat_id):
    data = load_chat(chat_id)
    return list(data.get("memory", []))


def add_to_memory(chat_id, note):
    data = load_chat(chat_id)
    if "memory" not in data:
        data["memory"] = []
    if note not in data["memory"]:
        data["memory"].append(note)
        save_chat(chat_id)


def clear_history(chat_id):
    data = load_chat(chat_id)
    data["history"] = []
    save_chat(chat_id)


def clear_memory(chat_id):
    data = load_chat(chat_id)
    data["memory"] = []
    save_chat(chat_id)


def get_persona(chat_id):
    data = load_chat(chat_id)
    return data.get("persona", "")


def set_persona(chat_id, persona):
    data = load_chat(chat_id)
    data["persona"] = persona
    save_chat(chat_id)


def clear_persona(chat_id):
    data = load_chat(chat_id)
    data["persona"] = ""
    save_chat(chat_id)


def get_chat_model(chat_id):
    data = load_chat(chat_id)
    return data.get("model", "") or DEFAULT_MODEL


def set_chat_model(chat_id, model):
    data = load_chat(chat_id)
    data["model"] = model
    save_chat(chat_id)


class DDGResultParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results = []
        self.cur = None
        self.in_a = False
        self.in_snippet = False

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        cls = d.get("class", "")
        if tag == "a" and "result__a" in cls:
            self.cur = {"title": "", "url": d.get("href", ""), "snippet": ""}
            self.in_a = True
        if tag == "a" and "result__snippet" in cls:
            self.in_snippet = True

    def handle_data(self, data):
        if self.in_a and self.cur is not None:
            self.cur["title"] += data
        if self.in_snippet and self.cur is not None:
            self.cur["snippet"] += data

    def handle_endtag(self, tag):
        if tag == "a" and self.in_a:
            self.in_a = False
            if self.cur and self.cur.get("title"):
                self.results.append(self.cur)
            self.cur = None
        if tag == "a" and self.in_snippet:
            self.in_snippet = False


class PageTextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "head", "title", "svg", "iframe"}
    BLOCK = ("p", "div", "br", "li", "h1", "h2", "h3", "h4", "section", "tr")

    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        if self.skip == 0 and tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)

    def handle_data(self, data):
        if self.skip == 0:
            self.parts.append(data)


def clean_search_url(url):
    if url.startswith("//"):
        url = "https:" + url
    url = re.sub(r"\s", "", url)
    url = re.split(r"(?<!:)//", url)[0]
    return url


def search_internet(query, n=MAX_SEARCH_RESULTS):
    try:
        r = requests.post("https://html.duckduckgo.com/html/",
                          data={"q": query}, headers=BROWSER_UA, timeout=25)
        if r.status_code != 200:
            print(f"[SEARCH WARN] status {r.status_code}", file=sys.stderr)
            return []
        p = DDGResultParser()
        p.feed(r.text)
        out = []
        for item in p.results:
            url = clean_search_url(item["url"])
            title = " ".join(item["title"].split())
            snippet = " ".join(item["snippet"].split())
            if url and title:
                out.append({"title": title, "url": url, "snippet": snippet})
            if len(out) >= n:
                break
        return out
    except Exception as e:
        print(f"[SEARCH ERROR] {e}", file=sys.stderr)
        return []


def fetch_page_text(url, max_chars=MAX_PAGE_CHARS):
    if not re.match(r"^https?://", url):
        return None
    try:
        r = requests.get(url, headers=BROWSER_UA, timeout=25)
        if r.status_code != 200:
            print(f"[FETCH WARN] {url} -> {r.status_code}", file=sys.stderr)
            return None
        p = PageTextExtractor()
        p.feed(r.text)
        raw = "".join(p.parts)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n\s*\n+", "\n", raw)
        raw = raw.strip()
        return raw[:max_chars]
    except Exception as e:
        print(f"[FETCH ERROR] {url}: {e}", file=sys.stderr)
        return None


def detect_search_query(text):
    low = text.lower().strip()
    if low.startswith("/search"):
        parts = text.split(maxsplit=1)
        return parts[1].strip() if len(parts) > 1 else None
    for kw in SEARCH_TRIGGERS:
        if kw in low:
            return text.strip()
    return None


def build_search_context(query):
    results = search_internet(query)
    if not results:
        return None
    lines = [f'[Web search results for "{query}"]']
    for i, item in enumerate(results, 1):
        lines.append(f"{i}. {item['title']}")
        lines.append(f"   URL: {item['url']}")
        if item.get("snippet"):
            lines.append(f"   {item['snippet']}")
    lines.append("Answer using these results. If they don't contain the answer, say so honestly.")
    return "\n".join(lines)


def build_page_context(url):
    text = fetch_page_text(url)
    if not text:
        return None
    return f"[Content of {url} (truncated)]\n{text}\n[End of content]"


def extract_url(text):
    m = re.search(r"https?://[^\s]+", text)
    return m.group(0).rstrip(".,;:!?)]}>") if m else None


def bale_api(method, **params):
    url = f"{BALE_API}/{method}"
    try:
        resp = requests.post(url, data=params or {}, timeout=30)
        try:
            data = resp.json()
        except ValueError:
            print(f"[BALE] non-json response {method}: {resp.text[:200]}", file=sys.stderr)
            return None
        if data.get("ok"):
            return data.get("result")
        if resp.status_code != 200:
            resp = requests.post(url, json=params or {}, timeout=30)
            try:
                data = resp.json()
            except ValueError:
                return None
            if data.get("ok"):
                return data.get("result")
        print(f"[BALE ERROR] {method}: {data}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[BALE EXCEPTION] {method}: {e}", file=sys.stderr)
        return None


def send_chat_action(chat_id, action="typing"):
    bale_api("sendChatAction", chat_id=chat_id, action=action)


def send_message(chat_id, text, reply_to=None):
    params = {"chat_id": chat_id, "text": text}
    if reply_to:
        params["reply_to_message_id"] = reply_to
    return bale_api("sendMessage", **params)


def sticker_file():
    return os.path.join(DATA_DIR, "stickers.json")


def get_stickers():
    try:
        with open(sticker_file(), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else data.get("stickers", [])
    except Exception:
        return []


def add_sticker(file_id, file_unique_id=""):
    stickers = get_stickers()
    for s in stickers:
        if s.get("file_id") in (file_id, file_unique_id) or (file_unique_id and s.get("file_unique_id") in (file_id, file_unique_id)):
            return
    stickers.append({"file_id": file_id, "file_unique_id": file_unique_id})
    stickers = stickers[-200:]
    with data_lock:
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            tmp = sticker_file() + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(stickers, f, ensure_ascii=False, indent=2)
            os.replace(tmp, sticker_file())
        except Exception as e:
            print(f"[STICKER SAVE ERROR] {e}", file=sys.stderr)


def sticker_cache_path(file_id):
    return os.path.join(DATA_DIR, "stickers", f"{file_id}.webp")


def download_bale_file(file_id):
    info = bale_api("getFile", file_id=file_id)
    path = info.get("file_path") if info else None
    if not path:
        return None, None
    url = f"https://tapi.bale.ai/file/bot{BALE_TOKEN}/{path}"
    data = None
    try:
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            data = r.content
            try:
                os.makedirs(os.path.dirname(sticker_cache_path(file_id)), exist_ok=True)
                with open(sticker_cache_path(file_id), "wb") as f:
                    f.write(data)
            except Exception as e:
                print(f"[STICKER CACHE ERROR] {e}", file=sys.stderr)
    except Exception as e:
        print(f"[BALE DOWNLOAD] {e}", file=sys.stderr)
    return data, url


def send_sticker(chat_id, file_id, reply_to=None):
    params = {"chat_id": chat_id}
    if reply_to:
        params["reply_to_message_id"] = reply_to

    ok = bale_api("sendSticker", sticker=file_id, **params)
    if ok:
        return ok
    print(f"[STICKER] file_id failed, trying download URL + upload for {file_id}", file=sys.stderr)

    data, url = download_bale_file(file_id)
    if url:
        ok = bale_api("sendSticker", sticker=url, **params)
        if ok:
            print("[STICKER] sent via download URL", file=sys.stderr)
            return ok
    if data:
        try:
            resp = requests.post(
                f"{BALE_API}/sendSticker",
                data=params,
                files={"sticker": ("sticker.webp", data, "image/webp")},
                timeout=60,
            )
            j = resp.json()
            if j.get("ok"):
                print("[STICKER] sent via multipart upload", file=sys.stderr)
                return j.get("result")
            print(f"[STICKER UPLOAD ERROR] {j}", file=sys.stderr)
        except Exception as e:
            print(f"[STICKER UPLOAD EXC] {e}", file=sys.stderr)
    return None


def chunk_text(text, size=CHUNK_SIZE):
    if len(text) <= size:
        return [text]
    chunks = []
    while text:
        if len(text) <= size:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, size)
        if split_at == -1:
            split_at = text.rfind(" ", 0, size)
        if split_at == -1:
            split_at = size
        else:
            split_at += 1
        chunks.append(text[:split_at])
        text = text[split_at:]
    return chunks


def send_long_message(chat_id, text, reply_to=None):
    for chunk in chunk_text(text):
        send_message(chat_id, chunk, reply_to=reply_to)
        time.sleep(0.3)


def extract_memory(text):
    low = text.strip().lower()
    for prefix in MEMORY_PREFIXES:
        if low.startswith(prefix):
            note = text.strip()[len(prefix):].strip().strip(":،،،")
            if note:
                return note
    return None


def is_addressed(message):
    chat = message.get("chat", {})
    if chat.get("type") == "private":
        return True

    text = message.get("text", "") or ""
    low = text.lower()

    if BOT_USERNAME in low:
        return True
    if "!tacobot" in low:
        return True

    for entity in message.get("entities", []):
        if entity.get("type") == "mention":
            start = entity.get("offset", 0)
            length = entity.get("length", 0)
            seg = text[start:start + length].lower()
            if seg.replace("@", "") == BOT_USERNAME.replace("@", ""):
                return True

    for name in BOT_NAME_VARIANTS:
        if name in low:
            return True

    return False


def build_ai_messages(chat_id, search_context=None):
    persona = get_persona(chat_id)
    messages = [{"role": "system", "content": persona or SYSTEM_PROMPT}]
    memory = get_memory(chat_id)
    if memory:
        mem_text = "چیزهایی که باید درباره این گفتگو به خاطر بسپاری:\n" + "\n".join(f"- {m}" for m in memory)
        messages.append({"role": "system", "content": mem_text})
    if search_context:
        messages.append({"role": "system", "content": search_context})
    messages.extend(get_history(chat_id))
    return messages


def chat_completion(chat_id, messages):
    headers = {
        "Authorization": f"Bearer {G4F_API_KEY}",
        "Content-Type": "application/json",
    }

    chosen = get_chat_model(chat_id)
    models = [chosen]
    if chosen != G4F_MODEL:
        models.append(G4F_MODEL)
    for m in FALLBACK_MODELS:
        if m not in models:
            models.append(m)

    last_exc = None
    for model in models:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.8,
            "max_tokens": 1024,
        }
        try:
            resp = requests.post(G4F_BASE, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
            data = resp.json()
            if resp.status_code != 200 or "choices" not in data:
                print(f"[G4F WARN] model '{model}' failed ({resp.status_code}): {data}", file=sys.stderr)
                continue
            content = data["choices"][0]["message"].get("content")
            reply = (content or "").strip()
            if not reply:
                print(f"[G4F WARN] model '{model}' returned empty content", file=sys.stderr)
                continue
            return reply
        except Exception as e:
            last_exc = e
            print(f"[G4F WARN] model '{model}' exception: {e}", file=sys.stderr)

    print(f"[G4F ERROR] all models failed: {last_exc}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    return None


def ask_ai(chat_id, user_text, search_context=None):
    add_to_history(chat_id, "user", user_text)

    note = extract_memory(user_text)
    if note:
        add_to_memory(chat_id, note)

    messages = [m for m in build_ai_messages(chat_id, search_context) if (m.get("content") or "").strip()]
    sticker_ctx, sticker_index = build_sticker_context(get_stickers())
    if sticker_ctx:
        messages.append({"role": "system", "content": sticker_ctx})

    reply = chat_completion(chat_id, messages)
    if reply is None:
        reply = "ببخشید، یه مشکلی پیش اومد. بعداً دوباره امتحان کن."
    add_to_history(chat_id, "assistant", reply)
    return reply, sticker_index


STICKER_TAG_RE = re.compile(r"<sticker\s+n=['\"]?(\d+)['\"]?\s*/?>")
SSH_TAG_RE = re.compile(r"<ssh\s+host=(['\"])(.*?)\1(?:\s+port=(['\"])(\d+)\3)?>(.*?)</ssh>", re.S)


def execute_ssh(host, port, command):
    if not SSH_PASSWORD and not SSH_KEY:
        return None, "SSH is not configured (set SSH_PASSWORD or SSH_KEY env)"
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        connect_kwargs = {"hostname": host, "port": int(port), "username": SSH_USER, "timeout": 15}
        if SSH_KEY:
            with io.StringIO(SSH_KEY) as buf:
                connect_kwargs["pkey"] = paramiko.RSAKey.from_private_key(buf)
        else:
            connect_kwargs["password"] = SSH_PASSWORD
        client.connect(**connect_kwargs)
        _in, out, err = client.exec_command(command, timeout=90)
        stdout = out.read().decode("utf-8", "replace")
        stderr = err.read().decode("utf-8", "replace")
        code = out.channel.recv_exit_status()
        combined = (stdout + (("\n" + stderr) if stderr else "")).strip()
        return code, combined or "(no output)"
    except Exception as e:
        return None, f"SSH error: {e}"
    finally:
        try:
            client.close()
        except Exception:
            pass


def send_reply_with_actions(chat_id, reply, sticker_index=None, reply_to=None, user_text=""):
    sent_anything = False
    if sticker_index and STICKER_TAG_RE.search(reply):
        for m in STICKER_TAG_RE.finditer(reply):
            fid = sticker_index.get(int(m.group(1)))
            if fid and send_sticker(chat_id, fid, reply_to=reply_to):
                sent_anything = True
                time.sleep(0.3)
        reply = STICKER_TAG_RE.sub("", reply).strip()

    ssh_results = []
    for m in SSH_TAG_RE.finditer(reply):
        host = m.group(2)
        port = int(m.group(4)) if m.group(4) else SSH_PORT
        command = m.group(5).strip()
        print(f"[SSH] {host}:{port} $ {command[:200]}", file=sys.stderr)
        code, output = execute_ssh(host, port, command)
        ssh_results.append((host, port, command, code, output))
    if ssh_results:
        reply = SSH_TAG_RE.sub("", reply).strip()
        block = "\n\n".join(
            f"$ ssh -p {port} {host}: {command}\n[exit {code}]\n{output}"
            for host, port, command, code, output in ssh_results
        )
        messages = [m for m in build_ai_messages(chat_id) if (m.get("content") or "").strip()]
        messages.append({
            "role": "system",
            "content": (
                "You ran these commands over SSH for the user. Report the results clearly in the user's language:\n\n"
                f"{block}\n\n"
                "Give a concise summary of what ran and what it returned."
            ),
        })
        final = chat_completion(chat_id, messages)
        if final:
            add_to_history(chat_id, "assistant", final)
            reply = final
        sent_anything = True

    if reply:
        send_long_message(chat_id, reply, reply_to=reply_to)
        return True
    return sent_anything


def handle_command(chat_id, command, message_id, arg=""):
    if command == "/start":
        send_long_message(chat_id,
            "yo!! من Tacos chiled ام، خوش اومدی 🌀\n\n"
            "هرچی بخوای بگو، به زبان خودت جواب میدم. می‌تونم سرچ کنم، لینک باز کنم، و چیزایی که می‌گی رو یادم بمونه.\n\n"
            "/help - راهنما",
            reply_to=message_id)
        return True
    if command == "/help":
        send_long_message(chat_id,
            "yo 👋 من Tacos chiled ام! خلاصه‌ی کاری که بلدم:\n\n"
            "🤖 /act <شخصیت> - تبدیل شو به هر چی (برای همیشه)\n"
            "🔄 /resetact - برگرد به حالت معمولی\n"
            "🧠 /model - لیست مدل‌های رایگان / انتخاب مدل\n"
            "🔎 /search <موضوع> - سرچ توی اینترنت\n"
            "🔗 /url <لینک> - باز کردن یه صفحه وب\n"
            "🖥️ SSH - بهم بگو «برو روی فلان سرور و این دستور رو بزن» تا باهات حرف بزنم\n"
            "😎 /sticker - یه استیکر تصادفی بفرستم (از اونایی که دریافت کردم)\n"
            "📦 /stickerpack <نام> - یه پکیج استیکر کامل اضافه کنم\n"
            "💾 /clear - پاک کردن تاریخچه\n"
            "🗑️ /forget - فراموش کردن حافظه\n\n"
            "علاوه بر این، می‌تونی:\n"
            "- «سرچ کن»، «گوگل»، «اخبار» یا «آخرین» بگی تا خودم سرچ کنم\n"
            "- یه لینک مستقیم بفرستی تا بازش کنم\n"
            "- بهم بگی «یادت باشه ...» تا یادم بمونه\n"
            "- برام استیکر بفرستی تا یادم بمونه و بعد با /sticker بفرستمش 😏\n\n"
            "و البته... Minecraft رو عالیه سرم میشه 😎 (اگه بخوای)\n\n"
            "در گروه‌ها فقط وقتی جواب می‌دم که منو تگ کنی یا بگی !tacobot.",
            reply_to=message_id)
        return True
    if command == "/act":
        if not arg:
            send_message(chat_id, "استفاده: /act <چیزی که می‌خوای بشم>\nمثلاً: /act یه شاعر ایرانی باش", reply_to=message_id)
            return True
        set_persona(chat_id, f"You are now acting as: {arg}. Follow this persona from now on in every reply, forever, until reset. Respond in the user's language.")
        send_message(chat_id, f"حله، از این به بعد نقش «{arg}» رو بازی می‌کنم! 🎭", reply_to=message_id)
        return True
    if command == "/resetact":
        clear_persona(chat_id)
        send_message(chat_id, "باشه، برگشتم به حالت خودم! تازه شدم ✨", reply_to=message_id)
        return True
    if command == "/model":
        if arg:
            if arg.lower() == "default":
                set_chat_model(chat_id, "")
                send_message(chat_id, "مدل برگشت به پیش‌فرض شد!", reply_to=message_id)
                return True
            chosen = arg.lower()
            set_chat_model(chat_id, chosen)
            send_message(chat_id, f"مدل این چت شد: {chosen}", reply_to=message_id)
            return True
        current = get_chat_model(chat_id)
        lines = [f"مدل فعلی این چت: {current}", "", "برای انتخاب: /model <اسم>", "/model default - برگشت به پیش‌فرض", "", "مدل‌های رایگان:"]
        for mid, label in CURATED_MODELS:
            mark = " ✅" if mid == current else ""
            lines.append(f"• {label} — `{mid}`{mark}")
        send_long_message(chat_id, "\n".join(lines), reply_to=message_id)
        return True
    if command == "/clear":
        clear_history(chat_id)
        send_message(chat_id, "تاریخچه پاک شد!", reply_to=message_id)
        return True
    if command == "/sticker":
        stickers = get_stickers()
        if not stickers:
            send_message(chat_id, "هنوز هیچ استیکری ندارم! یه استیکر برام بفرست یا با /stickerpack یه پکیج بده 🥺", reply_to=message_id)
            return True
        s = random.choice(stickers)
        ok = send_sticker(chat_id, s["file_id"], reply_to=message_id)
        if not ok:
            send_message(chat_id, "نتونستم استیکر بفرستم! 😢", reply_to=message_id)
        return True
    if command == "/stickerpack":
        if not arg:
            send_message(chat_id, "استفاده: /stickerpack <نام پکیج>\nمثلاً: /stickerpack tacos_heart_by_tacobot", reply_to=message_id)
            return True
        result = bale_api("getStickerSet", name=arg)
        if not result or not result.get("stickers"):
            send_message(chat_id, "نتونستم همچین پکیجی پیدا کنم! 🥲", reply_to=message_id)
            return True
        before = len(get_stickers())
        for s in result.get("stickers", []):
            fid = s.get("file_id")
            if fid:
                add_sticker(fid, s.get("file_unique_id", ""))
        count = len(result.get("stickers", []))
        send_message(chat_id, f"پکیج استیکر اضافه شد! {count} تا استیکر (از {before} → {len(get_stickers())}) 😎", reply_to=message_id)
        return True
    if command == "/forget":
        clear_memory(chat_id)
        send_message(chat_id, "همه چیزهایی که یادم بود فراموش کردم!", reply_to=message_id)
        return True
    return False


def handle_message(update):
    message = update.get("message")
    if not message:
        return

    chat = message.get("chat", {})
    chat_id = chat.get("id")
    if not chat_id:
        return

    text = message.get("text", "").strip()
    message_id = message.get("message_id")
    from_user = message.get("from", {})
    username = from_user.get("username", from_user.get("first_name", "unknown"))
    is_bot = from_user.get("is_bot", False)

    if is_bot:
        return

    sticker = message.get("sticker")
    if sticker:
        file_id = sticker.get("file_id")
        if file_id:
            add_sticker(file_id, sticker.get("file_unique_id", ""))
            print(f"[STICKER] saved from {username} ({chat_id}): {file_id}", file=sys.stderr)
        if is_addressed(message):
            send_message(chat_id, "واو استیکر! ذخیره شد 📸 حالا با /sticker می‌تونم برات بفرستمش 😎", reply_to=message_id)
        return

    if not text:
        if is_addressed(message):
            send_message(chat_id, "متأسفم، فقط با متن و استیکر می‌تونم کار کنم!", reply_to=message_id)
        return

    print(f"[MSG] {username} ({chat_id}) [{chat.get('type')}]: {text[:100]}", file=sys.stderr)

    if text.startswith("/"):
        parts = text.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""
        if handle_command(chat_id, command, message_id, arg):
            return

    if not is_addressed(message):
        return

    send_chat_action(chat_id, "typing")

    user_text = text
    search_context = None
    url = extract_url(text)
    low = text.lower()

    if low.startswith("/search"):
        q = text[len("/search"):].strip()
        if not q:
            send_message(chat_id, "استفاده: /search <موضوع>", reply_to=message_id)
            return
        search_context = build_search_context(q)
        if not search_context:
            send_message(chat_id, "نتونستم چیزی پیدا کنم! 🥲", reply_to=message_id)
            return
    elif low.startswith("/url"):
        u = text[len("/url"):].strip()
        if not u:
            send_message(chat_id, "استفاده: /url <آدرس>", reply_to=message_id)
            return
        user_text = f"Summarize or answer about this page: {u}"
        search_context = build_page_context(u)
        if not search_context:
            send_message(chat_id, "نتونستم این صفحه رو باز کنم!", reply_to=message_id)
            return
    elif text.startswith("http://") or text.startswith("https://"):
        user_text = f"Summarize or answer about this page: {url}"
        search_context = build_page_context(url)
        if not search_context:
            send_message(chat_id, "نتونستم این صفحه رو باز کنم!", reply_to=message_id)
            return
    else:
        query = detect_search_query(text)
        if query:
            search_context = build_search_context(query)

    reply, sticker_index = ask_ai(chat_id, user_text, search_context)
    send_reply_with_actions(chat_id, reply, sticker_index, reply_to=message_id, user_text=user_text)


def poll_loop():
    offset = 0
    print("[INFO] Starting poll loop...", file=sys.stderr)

    while True:
        try:
            params = {"timeout": POLL_TIMEOUT}
            if offset > 0:
                params["offset"] = offset

            resp = requests.post(
                f"{BALE_API}/getUpdates",
                data=params,
                timeout=POLL_TIMEOUT + 15
            )
            data = resp.json()

            if not data.get("ok"):
                print(f"[POLL ERROR] {data}", file=sys.stderr)
                time.sleep(5)
                continue

            updates = data.get("result", [])
            for update in updates:
                offset = update["update_id"] + 1
                try:
                    handle_message(update)
                except Exception as e:
                    print(f"[HANDLE ERROR] {e}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)

        except requests.exceptions.Timeout:
            pass
        except requests.exceptions.ConnectionError:
            print("[CONN ERROR] Connection lost, retrying in 5s...", file=sys.stderr)
            time.sleep(5)
        except Exception as e:
            print(f"[POLL EXCEPTION] {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            time.sleep(5)


def main():
    if not BALE_TOKEN:
        print("ERROR: BALE_TOKEN not set in .env", file=sys.stderr)
        sys.exit(1)
    if not G4F_API_KEY:
        print("ERROR: G4F_API_KEY not set in .env", file=sys.stderr)
        sys.exit(1)

    print("[INFO] Testing Bale API connection...", file=sys.stderr)
    me = bale_api("getMe")
    if not me:
        print("ERROR: Could not connect to Bale API. Check your token.", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] Bot connected: @{me.get('username', 'unknown')} (id: {me.get('id')})", file=sys.stderr)
    print("[INFO] Tacos chiled is online! Waiting for messages...", file=sys.stderr)

    poll_loop()


if __name__ == "__main__":
    main()