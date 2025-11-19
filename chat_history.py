import os
import json
from typing import List, Dict

CHAT_DIR = "chat_sessions"
os.makedirs(CHAT_DIR, exist_ok=True)

def list_chats() -> List[Dict]:
    """List all saved chat sessions."""
    chats = []
    if not os.path.exists(CHAT_DIR):
        return chats
    for fname in os.listdir(CHAT_DIR):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(CHAT_DIR, fname), "r") as f:
                    data = json.load(f)
                    chat_id = fname[:-5]
                    # Skip invalid entries
                    if not chat_id or not chat_id.strip():
                        print(f"[WARN] Skipping invalid chat file: {fname}")
                        continue
                    chat_name = data.get("name", chat_id)
                    if not chat_name or not chat_name.strip():
                        chat_name = chat_id
                    chats.append({
                        "id": chat_id,
                        "name": chat_name
                    })
            except Exception as e:
                print(f"[WARN] Failed to load chat {fname}: {e}")
    return chats

def save_chat(chat_id: str, name: str, messages: List[Dict]):
    """Save a chat session."""
    with open(os.path.join(CHAT_DIR, f"{chat_id}.json"), "w") as f:
        json.dump({"name": name, "messages": messages}, f)

def load_chat(chat_id: str) -> Dict:
    """Load a chat session."""
    try:
        with open(os.path.join(CHAT_DIR, f"{chat_id}.json"), "r") as f:
            return json.load(f)
    except Exception:
        return {"name": chat_id, "messages": []}

def rename_chat(chat_id: str, new_name: str):
    """Rename a chat session."""
    data = load_chat(chat_id)
    save_chat(chat_id, new_name, data.get("messages", []))
