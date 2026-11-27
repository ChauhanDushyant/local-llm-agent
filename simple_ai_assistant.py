# standard library
import os
from types import SimpleNamespace
# third party imports
from openai import OpenAI
from dotenv import load_dotenv
# local modules
from tools import TOOLS, run_tool
from Utilities.utilities import log


load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
    timeout=float(os.getenv("TIMEOUT", "120"))
)

messages = []

MAX_STEPS = 5
messages.append({
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
    )
})

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

while True:
    user_input = input("Dushyant: ")
    if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
        break

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

    except Exception as e:
        del messages[checkpoint:]
        print("Error:", e)