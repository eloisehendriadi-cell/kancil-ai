#!/usr/bin/env python3
"""Test chat history functionality"""
from chat_history import save_chat, load_chat, list_chats, rename_chat
import uuid

# Create a test chat
chat_id = str(uuid.uuid4())
chat_name = "Test Chat"
messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
]

print(f"Creating chat: {chat_id}")
save_chat(chat_id, chat_name, messages)

print("\nListing all chats:")
sessions = list_chats()
for session in sessions:
    print(f"  - ID: {session.get('id')}, Name: {session.get('name')}")

print("\nLoading chat:")
loaded = load_chat(chat_id)
print(f"  Name: {loaded.get('name')}")
print(f"  Messages: {len(loaded.get('messages', []))} messages")

print("\nRenaming chat to 'Updated Chat':")
rename_chat(chat_id, "Updated Chat")

print("\nListing all chats again:")
sessions = list_chats()
for session in sessions:
    print(f"  - ID: {session.get('id')}, Name: {session.get('name')}")

print("\nTest complete!")
