import tkinter as tk
from tkinter import scrolledtext
from tkinter import ttk
import threading
import time
import random
import re
from dataclasses import dataclass


# ==== CATR1.1 Backend – Distilled from DeepSeek-V4 & Janus Pro ====
# 14B parameter mock, bilingual (English & Mandarin), with heuristic interpreter.
# No files, no API keys, pure software – just clever Python code.
@dataclass
class CATR11Config:
    name: str = "CATR1.1-14B"
    max_tokens: int = 512
    temperature: float = 0.7


class CATR11LLM:
    """
    A playful mock of a 14B parameter CATR1.1 LLM,
    distilled from DeepSeek-V4 and Janus Pro architectures.
    Bilingual (EN/ZH) with heuristic understanding:
    - Language detection (based on Unicode range)
    - Pattern matching for intents (jokes, weather, time, etc.)
    - Keyword extraction for contextual fallback
    - No external files or APIs – all data embedded.
    """

    def __init__(self, config: CATR11Config | None = None):
        self.config = config or CATR11Config()
        # Built-in phrase banks and knowledge (no external files)
        self._init_english()
        self._init_mandarin()

        # Common patterns for intent detection (English)
        self.en_patterns = {
            "joke": re.compile(r"\b(joke|funny|laugh)\b", re.I),
            "weather": re.compile(r"\b(weather|rain|sun|temperature|forecast)\b", re.I),
            "time": re.compile(r"\b(time|clock|hour|minute|what time)\b", re.I),
            "name": re.compile(r"\b(your name|who are you|call you)\b", re.I),
            "capabilities": re.compile(r"\b(what can you do|capabilities|help|function)\b", re.I),
            "greeting": re.compile(r"\b(hello|hi|hey|greetings|howdy)\b", re.I),
            "farewell": re.compile(r"\b(bye|goodbye|see you|later|farewell)\b", re.I),
            "thanks": re.compile(r"\b(thank|thanks|appreciate|grateful)\b", re.I),
        }

        # Common patterns for intent detection (Mandarin)
        self.zh_patterns = {
            "joke": re.compile(r"[笑玩笑段子]", re.I),
            "weather": re.compile(r"[天气气候下雨晴]", re.I),
            "time": re.compile(r"[时间几点钟时分秒]", re.I),
            "name": re.compile(r"[你叫什么你是谁]", re.I),
            "capabilities": re.compile(r"[能做什么功能帮助]", re.I),
            "greeting": re.compile(r"[你好您好嗨哈喽]", re.I),
            "farewell": re.compile(r"[再见拜拜明天见后会有期]", re.I),
            "thanks": re.compile(r"[谢谢感谢]", re.I),
        }

        # Store compiled patterns per language
        self.patterns = {"en": self.en_patterns, "zh": self.zh_patterns}

        # Simple stopwords for keyword extraction (to ignore common words)
        self.en_stopwords = {"a", "an", "the", "is", "are", "was", "were", "i", "you", "he", "she", "it", "we", "they", "and", "or", "but", "if", "because", "as", "what", "which", "this", "that", "these", "those", "then", "just", "so", "too", "very", "can", "will", "be", "have", "do"}
        self.zh_stopwords = {"的", "了", "是", "在", "我", "你", "他", "她", "它", "我们", "你们", "他们", "和", "与", "或", "但是", "如果", "因为", "所以", "这", "那", "这些", "那些", "然后", "就", "太", "很", "能", "会", "将", "有", "做"}

    def _init_english(self):
        """English phrase bank."""
        self.en = {
            "greeting": [
                "Meow! How can I assist you today?",
                "Hi there! What's on your mind?",
                "Greetings! I'm ready to help.",
            ],
            "farewell": [
                "Goodbye! Feel free to return anytime.",
                "Take care! If you have more questions, I'm here.",
                "Bye! It was nice chatting with you.",
            ],
            "thanks": [
                "You're welcome! Happy to help.",
                "My pleasure! Anything else?",
                "Glad I could assist!",
            ],
            "name": [
                "I'm CATR1.1, a 14B parameter model distilled from DeepSeek-V4 and Janus Pro.",
                "You can call me CATR1.1. I combine the strengths of DeepSeek-V4 and Janus Pro.",
                "I'm your AI assistant, based on a distillation of two powerful architectures.",
            ],
            "capabilities": [
                "I can answer questions, help with writing, explain concepts, and more.",
                "My training covers a wide range of topics up to early 2025.",
                "I'm designed to be helpful, harmless, and honest.",
            ],
            "joke": [
                "Why don't cats play poker in the jungle? Too many cheetahs!",
                "What do you call a cat that loves to bowl? An alley cat!",
                "Why did the cat go to school? To improve his purr-suasion!",
            ],
            "weather": [
                "I can't access real weather data, but I hope it's sunny where you are!",
                "The forecast for today: a high chance of awesome!",
                "I wish I could tell you the weather, but I'm just a software cat.",
            ],
            "time": [
                f"The current time (simulated) is {time.strftime('%I:%M %p')}.",
                "I don't have a real clock, but my internal time says it's chat-o'clock!",
                "Time is an illusion. Lunchtime doubly so.",
            ],
            "default": [
                "That's interesting. Could you tell me more?",
                "I see. Let me think about that for a moment.",
                "Hmm, I need a bit more context to give a good answer.",
                "Interesting question! Here's what I think...",
            ],
        }

    def _init_mandarin(self):
        """Mandarin phrase bank."""
        self.zh = {
            "greeting": [
                "喵！今天我能帮你什么？",
                "你好！有什么想法吗？",
                "欢迎！我随时准备帮忙。",
            ],
            "farewell": [
                "再见！随时欢迎再来。",
                "保重！如果有更多问题，我在这里。",
                "拜拜！和你聊天很开心。",
            ],
            "thanks": [
                "不客气！很高兴帮忙。",
                "我的荣幸！还有别的吗？",
                "很高兴能帮到你！",
            ],
            "name": [
                "我是CATR1.1，一个140亿参数的模型，由DeepSeek-V4和Janus Pro蒸馏而来。",
                "你可以叫我CATR1.1，我结合了DeepSeek-V4和Janus Pro的优势。",
                "我是你的AI助手，基于两个强大架构的蒸馏。",
            ],
            "capabilities": [
                "我能回答问题、帮助写作、解释概念等等。",
                "我的训练涵盖截至2025年初的广泛主题。",
                "我被设计为乐于助人、无害且诚实。",
            ],
            "joke": [
                "为什么猫不喜欢玩扑克？因为有很多猎豹（cheetah，也指作弊者）！",
                "猫去学校学什么？学喵喵叫！",
                "有一只猫掉进了水里，变成了落汤猫。",
            ],
            "weather": [
                "我无法获取实时天气，但希望你那里阳光明媚！",
                "今天的天气预报：大概率有可爱的你！",
                "天气什么的，我猜是适合聊天的好日子。",
            ],
            "time": [
                f"模拟时间是 {time.strftime('%H:%M')}。",
                "我没有真正的时钟，但我的内部时间显示现在是聊天时间！",
                "时间是个幻觉，聊天时间更是双倍的幻觉。",
            ],
            "default": [
                "这很有趣。能再多说一点吗？",
                "我明白了。让我想一想。",
                "嗯，我需要更多背景才能给出好答案。",
                "有趣的问题！我是这么想的……",
            ],
        }

    def _detect_language(self, text: str) -> str:
        """Detect language: 'zh' if any Chinese character is present, else 'en'."""
        for ch in text:
            if '\u4e00' <= ch <= '\u9fff':
                return "zh"
        return "en"

    def _extract_keywords(self, text: str, lang: str) -> list:
        """Extract meaningful keywords (simple split and stopword removal)."""
        if lang == "en":
            words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
            stopwords = self.en_stopwords
            return [w for w in words if w not in stopwords and len(w) > 2]
        else:  # zh
            # For Chinese, treat each character as potential keyword, but skip common ones.
            chars = list(text)
            stopwords = self.zh_stopwords
            return [ch for ch in chars if ch not in stopwords and not ch.isspace()]

    def _match_intent(self, text: str, lang: str) -> str | None:
        """Return intent category if pattern matches, else None."""
        patterns = self.patterns[lang]
        for intent, pattern in patterns.items():
            if pattern.search(text):
                return intent
        return None

    def _generate_heuristic_default(self, prompt: str, lang: str) -> str:
        """Generate a contextual fallback response using extracted keywords."""
        keywords = self._extract_keywords(prompt, lang)
        if keywords:
            if lang == "en":
                sample = random.choice(keywords)
                templates = [
                    f"Tell me more about '{sample}'.",
                    f"What specifically about '{sample}' interests you?",
                    f"I'd love to discuss '{sample}'. Can you elaborate?",
                ]
            else:
                sample = random.choice(keywords)
                templates = [
                    f"Regarding '{sample}', could you say a bit more?",
                    f"Do you have any specific thoughts on '{sample}'?",
                    f"'{sample}' is interesting, could you expand on that?",
                ]
            return random.choice(templates)
        else:
            # Fallback to generic default
            return random.choice(self.en["default"] if lang == "en" else self.zh["default"])

    def generate(self, prompt: str) -> str:
        """
        Generate a response – entirely local, no I/O.
        Uses language detection, intent matching, and heuristic fallback.
        """
        # Simulate thinking time (varies slightly)
        time.sleep(random.uniform(0.5, 1.2))

        text = prompt.strip()
        if not text:
            return "🐱 CATR1.1: I'm listening, please say something. / 请说点什么 – 我在听。"

        lang = self._detect_language(text)

        # Try intent matching
        intent = self._match_intent(text, lang)
        if intent and intent in (self.en if lang == "en" else self.zh):
            # Use phrase bank for known intents
            phrase_bank = self.en if lang == "en" else self.zh
            base = random.choice(phrase_bank[intent])
        else:
            # No intent match → heuristic default
            base = self._generate_heuristic_default(text, lang)

        # Optional "reasoning trace" (30% chance)
        if random.random() < 0.3:
            if lang == "en":
                trace = (
                    "🧠 Distilled from DeepSeek-V4 & Janus Pro ...\n"
                    "   Multi-Head Latent Attention & Janus fusion applied.\n"
                    "   Heuristic interpreter active.\n"
                    "   Generating response...\n\n"
                )
            else:
                trace = (
                    "🧠 Distilling from DeepSeek-V4 and Janus Pro...\n"
                    "   Multi-head latent attention and Janus fusion applied.\n"
                    "   Heuristic interpreter active.\n"
                    "   Generating response...\n\n"
                )
        else:
            trace = ""

        emoji = "🐱"
        header = f"{emoji} CATR1.1 (14B, DeepSeek-V4 × Janus Pro, {lang.upper()}):\n"
        return f"{header}{trace}{base}"


# ==== GUI Application (CATR1.1 Theme) ====
class CATR11ChatApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("CATR1.1 · 14B Chat (Bilingual Heuristic)")
        self.geometry("900x600")
        self.minsize(600, 400)

        # Optional modern ttk theme
        try:
            self.style = ttk.Style(self)
            if "clam" in self.style.theme_names():
                self.style.theme_use("clam")
        except Exception:
            pass

        self.configure(bg="#020617")

        # Backend LLM – pure software, no files/keys
        self.llm = CATR11LLM()
        self._current_thread: threading.Thread | None = None

        self._build_layout()

    def _build_layout(self):
        # Header bar
        header = tk.Frame(self, bg="#020617", height=48)
        header.pack(side=tk.TOP, fill=tk.X)

        title_label = tk.Label(
            header,
            text="CATR1.1",
            fg="#e5e7eb",
            bg="#020617",
            font=("Segoe UI", 14, "bold"),
        )
        title_label.pack(side=tk.LEFT, padx=14, pady=(10, 6))

        model_pill = tk.Label(
            header,
            text="14B (DeepSeek-V4 × Janus Pro) · Heuristic Interpreter",
            fg="#e5e7eb",
            bg="#111827",
            font=("Segoe UI", 9),
            padx=10,
            pady=3,
        )
        model_pill.pack(side=tk.LEFT, padx=(8, 0), pady=(12, 6))

        # Black background, Blue text
        clear_btn = tk.Button(
            header,
            text="Clear Chat",
            command=self._clear_chat,
            bg="black",
            fg="blue",
            activebackground="#1a1a1a",
            activeforeground="#3b82f6",
            relief=tk.FLAT,
            padx=8,
            pady=2,
        )
        clear_btn.pack(side=tk.RIGHT, padx=14, pady=(10, 6))

        # Chat area
        main_frame = tk.Frame(self, bg="#020617")
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=12, pady=(2, 4))

        self.chat_box = scrolledtext.ScrolledText(
            main_frame,
            wrap=tk.WORD,
            state=tk.DISABLED,
            bg="#020617",
            fg="#e5e7eb",
            insertbackground="#e5e7eb",
            font=("Segoe UI", 10),
            relief=tk.FLAT,
            borderwidth=0,
        )
        self.chat_box.pack(fill=tk.BOTH, expand=True)

        # Input area
        input_container = tk.Frame(self, bg="#020617")
        input_container.pack(side=tk.BOTTOM, fill=tk.X, padx=12, pady=(4, 12))

        input_frame = tk.Frame(input_container, bg="#020617")
        input_frame.pack(side=tk.TOP, fill=tk.X)

        self.input_text = tk.Text(
            input_frame,
            height=3,
            bg="#020617",
            fg="#e5e7eb",
            insertbackground="#e5e7eb",
            relief=tk.FLAT,
            font=("Segoe UI", 10),
            wrap=tk.WORD,
        )
        self.input_text.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 4), pady=4)
        self.input_text.bind("<Return>", self._on_enter_pressed)

        buttons_frame = tk.Frame(input_frame, bg="#020617")
        buttons_frame.pack(side=tk.RIGHT, padx=(4, 0), pady=4)

        # Black background, Blue text
        stop_btn = tk.Button(
            buttons_frame,
            text="Stop",
            command=self._stop_response,
            bg="black",
            fg="blue",
            activebackground="#1a1a1a",
            activeforeground="#3b82f6",
            relief=tk.FLAT,
            padx=10,
            pady=4,
        )
        stop_btn.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))

        # Black background, Blue text
        send_btn = tk.Button(
            buttons_frame,
            text="Send",
            command=self._on_send_click,
            bg="black",
            fg="blue",
            activebackground="#1a1a1a",
            activeforeground="#3b82f6",
            relief=tk.FLAT,
            padx=10,
            pady=4,
        )
        send_btn.pack(side=tk.TOP, fill=tk.X)

        hint_label = tk.Label(
            input_container,
            text="🐱 Bilingual EN/ZH · Heuristic Interpreter · No Files/API Keys · Python 3.14 Ready",
            fg="#6b7280",
            bg="#020617",
            font=("Segoe UI", 8),
            anchor="w",
        )
        hint_label.pack(side=tk.TOP, fill=tk.X, padx=(4, 4), pady=(2, 0))

        # Initial system message
        self._append_message(
            "system",
            "Welcome to CATR1.1 (14B).\n"
            "This is a bilingual (English/Mandarin) heuristic interpreter – responses are more contextual.\n"
            "Try asking for a joke, the weather, the time, or just chat. Type in English or Chinese.\n"
            "Press Enter to send (Shift+Enter for newline).",
        )

        self.after(200, lambda: self.input_text.focus_set())

    # ==== Chat Helper Functions ====
    def _append_message(self, sender: str, message: str):
        self.chat_box.config(state=tk.NORMAL)

        if sender == "user":
            label = "You"
            tag = "user"
        elif sender == "assistant":
            label = "CATR1.1"
            tag = "assistant"
        else:
            label = ""
            tag = "system"

        if self.chat_box.index("end-1c") != "1.0":
            self.chat_box.insert(tk.END, "\n")

        if label:
            self.chat_box.insert(tk.END, f"{label}:\n", (f"{tag}_label",))
        self.chat_box.insert(tk.END, message + "\n", (tag,))

        self.chat_box.tag_config(
            "user_label",
            foreground="#a5b4fc",
            font=("Segoe UI", 9, "bold"),
        )
        self.chat_box.tag_config(
            "assistant_label",
            foreground="#6ee7b7",
            font=("Segoe UI", 9, "bold"),
        )
        self.chat_box.tag_config(
            "user",
            foreground="#e5e7eb",
            font=("Segoe UI", 10),
        )
        self.chat_box.tag_config(
            "assistant",
            foreground="#d1fae5",
            font=("Segoe UI", 10),
        )
        self.chat_box.tag_config(
            "system",
            foreground="#9ca3af",
            font=("Segoe UI", 9, "italic"),
        )

        self.chat_box.config(state=tk.DISABLED)
        self.chat_box.see(tk.END)

    def _on_enter_pressed(self, event):
        if event.state & 0x0001:
            return
        self._on_send_click()
        return "break"

    def _on_send_click(self):
        user_text = self.input_text.get("1.0", tk.END).strip()
        if not user_text:
            return

        self.input_text.delete("1.0", tk.END)
        self._append_message("user", user_text)

        if self._current_thread and self._current_thread.is_alive():
            return

        self._current_thread = threading.Thread(
            target=self._handle_llm_response,
            args=(user_text,),
            daemon=True,
        )
        self._current_thread.start()

    def _handle_llm_response(self, user_text: str):
        try:
            reply = self.llm.generate(user_text)
        except Exception as e:
            reply = f"(CATR1.1 Backend Error: {e})"
        self.after(0, lambda: self._append_message("assistant", reply))

    def _clear_chat(self):
        self.chat_box.config(state=tk.NORMAL)
        self.chat_box.delete("1.0", tk.END)
        self.chat_box.config(state=tk.DISABLED)

    def _stop_response(self):
        pass


if __name__ == "__main__":
    app = CATR11ChatApp()
    app.mainloop()
