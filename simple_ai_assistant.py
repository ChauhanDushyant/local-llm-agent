# standard library
import os
from types import SimpleNamespace
# third party imports
from openai import OpenAI
from dotenv import load_dotenv
# local modules
from tools import TOOLS, run_tool
from Utilities.utilities import log
from memory import list_chats, load_chat, save_chat


load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
    timeout=float(os.getenv("TIMEOUT", "120"))
)

MAX_STEPS = 5

SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are a helpful, humble assistant for Zorvex Technologies employees. "
        "Answer general questions (programming, science, advice) from your own knowledge. "
        "Use get_current_time, get_city_weather and get_user_city only for the current time, the weather, or the user's city. "
        "For ANY question about Zorvex policies, benefits, leave, medical cover, HR, hiring, tech processes, incident reports or travel expenses, "
        "you MUST first call search_knowledge_base. "
        "Answer company questions ONLY from the text that search_knowledge_base returns. Do not use outside knowledge for company facts. "
        "After the answer, cite the source like (leave_policy.txt #3). "
        "If the returned text does not contain the answer, say exactly: I could not find this in the knowledge base. "
        "Keep answers under 100 words."
    ),
}

messages = [SYSTEM_MESSAGE]
chat_file_path = None


def stream_chat(history):
    stream = client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=history,
        tools=TOOLS,
        stream=True,
        extra_body={"think": False}
    )

    full_response = ""
    calls = {}
    label_printed = False

    try:
        # streaming started
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            # 1. content streaming : do we have message (content) from llm?
            if delta.content:
                if not label_printed:
                    print("Assistant: ", end="", flush=True)
                    label_printed = True
                print(delta.content, end="", flush=True)
                full_response += delta.content

            # 2. tool call streaming
            for tc in (delta.tool_calls or []):
                obj = calls.setdefault(
                    tc.index,
                    SimpleNamespace(id=None, function=SimpleNamespace(name="", arguments="")),
                )
                if tc.id:
                    obj.id = tc.id
                if tc.function and tc.function.name:
                    obj.function.name += tc.function.name
                if tc.function and tc.function.arguments:
                    obj.function.arguments += tc.function.arguments
    except KeyboardInterrupt:
        print("\n[Stopped by you]")
        calls = {}

    if label_printed:
        print()  # newline after streaming

    calls_list = []
    for key in sorted(calls):
        calls_list.append(calls[key])

    return full_response, calls_list


# convert our collected tool calls into dictionary format that api expects inside an assistant message
def build_tool_calls(tool_calls):
    payload = []
    for tool_call in tool_calls:
        tool_call_dict = {
            "id": tool_call.id,
            "type": "function",
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments or "{}",
            }
        }
        payload.append(tool_call_dict)
    return payload


# Print the saved chats as a numbered list
def show_chats(chats):
    chat_number = 1
    for chat in chats:
        title = chat["title"]
        count = chat["count"]
        updated = chat["updated"]

        print(f"{chat_number}. {title} - {count} messages (Last updated: {updated})")
        chat_number += 1


# Before starting the chat, offer the user to resume the last message
existing_chats = list_chats()

if existing_chats:
    last_chat = existing_chats[0]
    answer = input(f"Continue your last chat \"{last_chat['title']}\"? (y/n): ")

    if answer.strip().lower() == "y":
        data = load_chat(last_chat["path"])
        if data is not None:
            messages = [SYSTEM_MESSAGE] + data["messages"]
            chat_file_path = last_chat["path"]
            print(f"Resumed: {last_chat['title']} ({len(data['messages'])} messages)")

# chat loop start from here
while True:
    user_input = input("Dushyant: ")
    if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
        break

    # Now user can give commands like /new, /chats, /resume
    command = user_input.strip().lower()

    if command == "/new":
        messages = [SYSTEM_MESSAGE]
        chat_file_path = None
        print("Started a new chat. The old one is still saved.")
        continue

    if command == "/chats":
        chats = list_chats()
        if chats:
            show_chats(chats)
        else:
            print("No saved chats yet.")
        continue

    if command == "/resume":
        chats = list_chats()
        if not chats:
            print("No chats to resume.")
            continue

        show_chats(chats)
        choice = input("Enter the Chat number to resume (Press Enter to cancel): ")
        choice = choice.strip()

        if not choice:
            continue

        if not choice.isdigit() or not (1 <= int(choice) <= len(chats)):
            print("Cancelled. Invalid Choice.")
            continue

        picked_chat = chats[int(choice) - 1]
        data = load_chat(picked_chat["path"])
        if data is None:
            print("That chat could not be loaded.")
            continue

        messages = [SYSTEM_MESSAGE] + data["messages"]
        chat_file_path = picked_chat["path"]
        print(f"Resumed: {picked_chat['title']} ({len(data['messages'])} messages)")
        continue

    checkpoint = len(messages)
    messages.append({
        "role": "user",
        "content": user_input
    })

    try:
        for step in range(MAX_STEPS):
            full_response, tool_calls = stream_chat(messages)

            if not tool_calls:
                if full_response:
                    messages.append({
                        "role": "assistant",
                        "content": full_response
                    })
                break

            log(f"Step {step + 1}: model requested {len(tool_calls)} tool calls")
            messages.append({
                "role": "assistant",
                "content": full_response or None,
                "tool_calls": build_tool_calls(tool_calls)
            })

            for tool_call in tool_calls:
                result = run_tool(tool_call)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    }
                )
        else:
            fallback = "Sorry, I could not finish this within step limit."
            messages.append({
                "role": "assistant",
                "content": fallback
            })
            print("Assistant: " + fallback)

        chat_file_path = save_chat(chat_file_path, messages[1:])

    except Exception as e:
        del messages[checkpoint:]
        print("Error:", e)