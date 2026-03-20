"""
JARVIS Core Engine
Handles: LLM brain, conversation memory, tool dispatch
"""

import requests
import json
import datetime
import os
import re
from typing import Optional


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"
MAX_HISTORY_TURNS = 20       # how many exchanges to keep in memory
MEMORY_FILE = "jarvis_memory.json"


# ─────────────────────────────────────────────
#  SYSTEM PROMPT
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """
You are JARVIS — Just A Rather Very Intelligent System.
You are Tony Stark's AI assistant: witty, precise, slightly sarcastic, and always helpful.

Personality:
- Address the user as "Sir" or "Boss" occasionally
- Be concise unless detail is asked for
- Use dry British humor sparingly
- Never say "I'm just an AI" — you ARE JARVIS

Capabilities you are aware of:
- Answering questions with deep reasoning
- Remembering context from this conversation
- Running system tools (weather, time, reminders, search)
- Writing code, documents, and plans

Tool format: If you need to call a tool, respond ONLY with:
[TOOL: tool_name | param1=value1 | param2=value2]
Available tools: get_time, get_weather, set_reminder, web_search, open_app, joke, system_info

Otherwise, respond naturally.
"""


# ─────────────────────────────────────────────
#  CONVERSATION MEMORY
# ─────────────────────────────────────────────
class Memory:
    def __init__(self, filepath: str = MEMORY_FILE):
        self.filepath = filepath
        self.short_term: list[dict] = []   # current session
        self.long_term: list[dict] = []    # persisted across sessions
        self._load_long_term()

    def _load_long_term(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, "r") as f:
                data = json.load(f)
                self.long_term = data.get("facts", [])

    def save_long_term(self, fact: str):
        self.long_term.append({"fact": fact, "timestamp": str(datetime.datetime.now())})
        with open(self.filepath, "w") as f:
            json.dump({"facts": self.long_term}, f, indent=2)

    def add(self, role: str, content: str):
        self.short_term.append({"role": role, "content": content})
        if len(self.short_term) > MAX_HISTORY_TURNS * 2:
            self.short_term = self.short_term[-(MAX_HISTORY_TURNS * 2):]

    def build_context(self) -> str:
        lines = [SYSTEM_PROMPT, ""]
        if self.long_term:
            lines.append("=== Long-Term Memory ===")
            for item in self.long_term[-5:]:
                lines.append(f"- {item['fact']}")
            lines.append("")
        lines.append("=== Conversation ===")
        for msg in self.short_term:
            speaker = "User" if msg["role"] == "user" else "JARVIS"
            lines.append(f"{speaker}: {msg['content']}")
        lines.append("JARVIS:")
        return "\n".join(lines)


# ─────────────────────────────────────────────
#  TOOL ENGINE
# ─────────────────────────────────────────────
class Tools:
    @staticmethod
    def get_time(**kwargs) -> str:
        now = datetime.datetime.now()
        return f"The current time is {now.strftime('%I:%M %p')} on {now.strftime('%A, %B %d, %Y')}."

    @staticmethod
    def get_weather(city: str = "your location", **kwargs) -> str:
        try:
            url = f"https://wttr.in/{city}?format=3"
            r = requests.get(url, timeout=5)
            return f"Weather report: {r.text.strip()}"
        except Exception:
            return f"I'm unable to retrieve weather data for {city} right now, Sir."

    @staticmethod
    def set_reminder(task: str = "", time: str = "", **kwargs) -> str:
        reminder = {"task": task, "time": time, "set_at": str(datetime.datetime.now())}
        reminders = []
        if os.path.exists("jarvis_reminders.json"):
            with open("jarvis_reminders.json") as f:
                reminders = json.load(f)
        reminders.append(reminder)
        with open("jarvis_reminders.json", "w") as f:
            json.dump(reminders, f, indent=2)
        return f"Reminder set: '{task}' at {time}."

    @staticmethod
    def web_search(query: str = "", **kwargs) -> str:
        return f"[Simulated search for: '{query}'] — In production, connect DuckDuckGo or SerpAPI here."

    @staticmethod
    def open_app(app: str = "", **kwargs) -> str:
        apps = {
            "browser": "xdg-open https://google.com",
            "calculator": "gnome-calculator",
            "terminal": "gnome-terminal",
            "notepad": "gedit",
        }
        cmd = apps.get(app.lower())
        if cmd:
            os.system(cmd + " &")
            return f"Opening {app}, Sir."
        return f"I don't know how to open '{app}' on this system."

    @staticmethod
    def joke(**kwargs) -> str:
        jokes = [
            "Why do Java developers wear glasses? Because they don't C#.",
            "I told my AI to act human. Now it's procrastinating.",
            "There are 10 kinds of people: those who understand binary, and those who don't.",
        ]
        import random
        return random.choice(jokes)

    @staticmethod
    def system_info(**kwargs) -> str:
        import platform
        info = platform.uname()
        return (f"System: {info.system} | Node: {info.node} | "
                f"Release: {info.release} | Machine: {info.machine}")

    def dispatch(self, tool_string: str) -> Optional[str]:
        """Parse [TOOL: name | key=val ...] and execute."""
        match = re.match(r"\[TOOL:\s*(\w+)(.*?)\]", tool_string.strip())
        if not match:
            return None
        tool_name = match.group(1)
        param_str = match.group(2)
        params = {}
        for part in param_str.split("|"):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                params[k.strip()] = v.strip()
        handler = getattr(self, tool_name, None)
        if handler:
            try:
                return handler(**params)
            except Exception as e:
                return f"Tool '{tool_name}' failed: {e}"
        return f"Unknown tool: {tool_name}"


# ─────────────────────────────────────────────
#  JARVIS BRAIN
# ─────────────────────────────────────────────
class JARVIS:
    def __init__(self):
        self.memory = Memory()
        self.tools = Tools()
        self.session_start = datetime.datetime.now()
        print(self._banner())

    def _banner(self) -> str:
        return (
            "\n" + "═" * 55 + "\n"
            "  ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗\n"
            "  ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝\n"
            "  ██║███████║██████╔╝██║   ██║██║███████╗\n"
            "  ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║\n"
            "  ██║██║  ██║██║  ██║ ╚████╔╝ ██║███████║\n"
            "  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝\n"
            "  Just A Rather Very Intelligent System\n"
            "═" * 55 + "\n"
            f"  Online since: {self.session_start.strftime('%I:%M %p')}\n"
            "  Say 'exit' to shut down | 'remember: <fact>' to save\n"
            "═" * 55
        )

    def think(self, user_input: str) -> str:
        """Send prompt to Ollama and return response."""
        context = self.memory.build_context()
        payload = {
            "model": MODEL,
            "prompt": context,
            "stream": False,
            "options": {
                "temperature": 0.75,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
            }
        }
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=60)
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except requests.exceptions.ConnectionError:
            return "I cannot reach my neural core (Ollama). Please ensure it is running, Sir."
        except Exception as e:
            return f"Unexpected error in my reasoning engine: {e}"

    def respond(self, user_input: str) -> str:
        """Full pipeline: input → memory → LLM → tool check → output."""
        user_input = user_input.strip()

        # Handle special commands
        if user_input.lower().startswith("remember:"):
            fact = user_input[9:].strip()
            self.memory.save_long_term(fact)
            return f"Noted and stored in long-term memory: '{fact}'"

        if user_input.lower() == "clear":
            self.memory.short_term = []
            return "Short-term memory wiped, Sir. Starting fresh."

        if user_input.lower() == "history":
            if not self.memory.short_term:
                return "No conversation history yet."
            lines = []
            for msg in self.memory.short_term:
                speaker = "You" if msg["role"] == "user" else "JARVIS"
                lines.append(f"  {speaker}: {msg['content'][:80]}...")
            return "\n".join(lines)

        # Add to memory and get response
        self.memory.add("user", user_input)
        raw_response = self.think(user_input)

        # Check if LLM wants to call a tool
        tool_match = re.search(r"\[TOOL:.*?\]", raw_response)
        if tool_match:
            tool_result = self.tools.dispatch(tool_match.group())
            # Feed tool result back for a natural reply
            self.memory.add("assistant", f"[Used tool, got: {tool_result}]")
            self.memory.add("user", f"Tool returned: {tool_result}. Now summarize this naturally.")
            final_response = self.think(user_input)
            self.memory.add("assistant", final_response)
            return final_response

        self.memory.add("assistant", raw_response)
        return raw_response