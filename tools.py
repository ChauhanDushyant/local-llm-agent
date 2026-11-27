import json
from datetime import datetime
from urllib.parse import quote
from rag import search
import requests

from Utilities.utilities import log


def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_user_city():
    return "Delhi"

def search_knowledge_base(query):
    results = search(query, top_k=2)

    log(f"📚 Knowledge base search: {query}")
    for r in results:
        log(f"   {r['score']:.2f} - {r['source']} #{r['chunk_id']}")

    blocks = []
    for r in results:
        sourceFileName = r['source']
        chunkId        = r['chunk_id']
        chunkText      = r['text']

        header = f"[{sourceFileName} #{chunkId}]"
        block  = f"{header}\n{chunkText}"
        blocks.append(block)

    return "\n\n".join(blocks)

def _weather_from_open_meteo(city):
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
        timeout=10
    )
    geo.raise_for_status()
    results = geo.json().get("results")
    if not results:
        return f"City not found: {city}"
    place = results[0]

    weather = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current_weather": "true"
        },
        timeout=10
    )
    weather.raise_for_status()
    current = weather.json()["current_weather"]
    return f"{place['name']}: {current['temperature']}C, wind {current['windspeed']} km/h"


def get_city_weather(city):
    if not city or not city.strip():
        return "City name is required."

    url = f"https://wttr.in/{quote(city)}?format=3"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text.strip()
    except requests.exceptions.RequestException:
        pass

    try:
        return _weather_from_open_meteo(city)
    except requests.exceptions.Timeout:
        return "Weather service timed out."
    except requests.exceptions.RequestException as e:
        return f"Weather error: {e}"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current date and time in the formatted string",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_city",
            "description": (
                "Returns the city where the user lives. Use this tool when the user says "
                "'my city', 'where I live', 'my location' or 'my home' without naming a city. "
                "Call this first, then use the returned city for other tools."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_city_weather",
            "description": "Returns the current weather information for a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City to get weather for."}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Searches the Zorvex Technologies company knowledge base: leave,medical,HR, hiring, tech, incident"
                           "reports and travel expense policies. use it for any question about company policies or document."
                           "Returns the most relevant passages with their source.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The users question, rewitten as a short, self-contained search query.",}
                },
                "required": ["query"],
            },
        },
    }
    
]

FUNCTION_NAME_TO_FUNCTION = {
    "get_current_time": get_current_time,
    "get_user_city": get_user_city,
    "get_city_weather": get_city_weather,
    "search_knowledge_base" : search_knowledge_base
}


def run_tool(tool_call):
    func_name = tool_call.function.name
    try:
        func_args = json.loads(tool_call.function.arguments or "{}")
        log(f"🔧 {func_name}({func_args})")
        fn = FUNCTION_NAME_TO_FUNCTION.get(func_name)
        if fn is None:
            result = f"Unknown tool: {func_name}"
        else:
            result = fn(**func_args)
    except Exception as e:
        result = f"Error while running {func_name} : {e}"

    log(f"   ↳ {result}")
    return str(result)