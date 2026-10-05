import datetime
import os
import requests
from dotenv import load_dotenv
import wikipedia

# Deletions are confined to this directory (override with AGENT_SANDBOX_ROOT).
SANDBOX_ROOT = os.path.realpath(os.environ.get("AGENT_SANDBOX_ROOT", os.path.dirname(os.path.abspath(__file__))))

def _within_sandbox(target_realpath: str) -> bool:
    try:
        return os.path.commonpath([SANDBOX_ROOT, target_realpath]) == SANDBOX_ROOT
    except ValueError:
        return False  # e.g. different drives on Windows, or an invalid path

def get_current_time() -> str:
    now = datetime.datetime.now()
    return f"The current date and time is {now.strftime('%Y-%m-%d %H:%M:%S')}."

def get_current_weather(lat: float = 34.05, lon: float = -118.24) -> str:
    load_dotenv()
    api_key = os.getenv("OPEN_WEATHER_API_KEY")
    url = (f"https://api.openweathermap.org/data/2.5/weather"
           f"?lat={lat}&lon={lon}&units=metric&appid={api_key}")   # units=metric -> Celsius
    resp = requests.get(url)
    if resp.status_code == 200:
        data = resp.json()
        city_name = data.get("name")
        temp = data["main"]["temp"]
        description = data["weather"][0]["description"]
        return f"Current weather in {city_name}: {temp}°C, {description}."
    return f"Failed to retrieve data. Status code: {resp.status_code}"

def wikipedia_search_tool(query: str) -> str:
    try:
        search_results = wikipedia.search(query, results=3)
        if not search_results:
            return f"No Wikipedia pages found matching '{query}'."
        page_title = search_results[0]
        summary = wikipedia.summary(page_title, sentences=3)
        return f"Page: {page_title}\n\nSummary:\n{summary}"
    except wikipedia.exceptions.DisambiguationError as e:
        return f"The term '{query}' is ambiguous. Did you mean one of these? {', '.join(e.options[:5])}"
    except wikipedia.exceptions.PageError:
        return f"Could not find a specific page for '{query}'."
    except Exception as e:
        return f"An error occurred: {str(e)}"

def delete_files(file_path: str) -> str:
    # realpath resolves symlinks and '..', so traversal escapes are caught by the sandbox check
    target = os.path.realpath(file_path)
    if not _within_sandbox(target):
        return f"Refused: '{file_path}' is outside the allowed directory ({SANDBOX_ROOT})."
    if target == SANDBOX_ROOT:
        return "Refused: cannot delete the sandbox root itself."
    if not os.path.exists(target):
        return f"File not found: '{file_path}'."
    if os.path.isdir(target):
        return f"Refused: '{file_path}' is a directory, not a file."
    os.remove(target)
    return f"Deleted: '{file_path}'."

def get_current_file_path(file_name: str) -> str:
    return os.path.abspath(file_name)
