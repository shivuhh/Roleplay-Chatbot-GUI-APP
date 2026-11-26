from __future__ import annotations

import json
import platform
import queue
import threading
import tkinter as tk
import tkinter.font as tkfont
from dataclasses import dataclass, field
from datetime import datetime
from tkinter import filedialog, messagebox, ttk
from typing import Any, Iterator, Optional


# --------------------------------------------------------------------------
# Core logic
# --------------------------------------------------------------------------

@dataclass
class Settings:
    model: str = "matthew"
    trim_ratio: float = 0.85


@dataclass
class Stats:
    prompt_tokens: int = 0
    reply_tokens: int = 0
    tokens_per_sec: float = 0.0


@dataclass
class StreamEvent:
    kind: str  # "chunk" | "error" | "done"
    text: str = ""
    stats: Optional[Stats] = None
    cancelled: bool = False


def estimate_tokens(text: str) -> int:
    """Rough fallback (~4 chars/token) used only until real counts arrive."""
    return max(1, len(text) // 4)


@dataclass
class ChatClient:
    settings: Settings = field(default_factory=Settings)
    client: Any = None
    history: list[dict] = field(default_factory=list)
    last_stats: Optional[Stats] = None
    session_tokens: int = 0
    context_limit: Optional[int] = None

    def __post_init__(self) -> None:
        if self.client is None:
            import ollama
            self.client = ollama.Client()

    def list_models(self) -> list[str]:
        try:
            names = []

            for model in self.client.list().models:
                name = (
                    getattr(model, "model", None)
                    or getattr(model, "name", None)
                )

                if name:
                    names.append(name)

            return sorted(names) or [self.settings.model]
        except Exception:
            return [self.settings.model]

    def refresh_context_limit(self) -> Optional[int]:
        """Read PARAMETER num_ctx from the model's Modelfile."""
        self.context_limit = None

        try:
            parameters = getattr(
                self.client.show(self.settings.model),
                "parameters",
                "",
            ) or ""

            for line in parameters.splitlines():
                parts = line.split()

                if len(parts) == 2 and parts[0] == "num_ctx":
                    self.context_limit = int(parts[1])
        except Exception:
            pass

        return self.context_limit

    def add_user(self, text: str) -> None:
        self.history.append({
            "role": "user",
            "content": text,
            "tokens": estimate_tokens(text),
        })

    def pop_last_assistant(self) -> bool:
        if self.history and self.history[-1]["role"] == "assistant":
            self.history.pop()
            self.last_stats = None
            return True

        return False

    def last_assistant_text(self) -> str:
        if self.history and self.history[-1]["role"] == "assistant":
            return self.history[-1]["content"]

        return ""

    def clear(self) -> None:
        self.history.clear()
        self.last_stats = None
        self.session_tokens = 0

    @property
    def context_used(self) -> int:
        if self.last_stats:
            return (
                self.last_stats.prompt_tokens
                + self.last_stats.reply_tokens
            )

        return sum(message["tokens"] for message in self.history)

    def trim_history(self) -> None:
        """Drop oldest messages until they fit the context budget."""
        if not self.context_limit:
            return

        budget = int(
            self.context_limit * self.settings.trim_ratio
        )
        total = sum(
            message["tokens"] for message in self.history
        )

        while total > budget and len(self.history) > 1:
            total -= self.history.pop(0)["tokens"]

        while (
            len(self.history) > 1
            and self.history[0]["role"] != "user"
        ):
            self.history.pop(0)

    def _messages(self) -> list[dict]:
        return [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in self.history
        ]

    def stream_reply(
        self,
        cancel: threading.Event,
    ) -> Iterator[StreamEvent]:
        self.trim_history()

        parts: list[str] = []
        stats: Optional[Stats] = None
        error: Optional[str] = None

        try:
            stream = self.client.chat(
                model=self.settings.model,
                messages=self._messages(),
                stream=True,
            )

            for chunk in stream:
                if cancel.is_set():
                    break

                text = chunk.message.content or ""

                if text:
                    parts.append(text)
                    yield StreamEvent("chunk", text)

                if getattr(chunk, "done", False):
                    eval_count = (
                        getattr(chunk, "eval_count", 0) or 0
                    )
                    eval_ns = (
                        getattr(chunk, "eval_duration", 0) or 0
                    )

                    stats = Stats(
                        prompt_tokens=(
                            getattr(
                                chunk,
                                "prompt_eval_count",
                                0,
                            ) or 0
                        ),
                        reply_tokens=eval_count,
                        tokens_per_sec=(
                            eval_count / (eval_ns / 1e9)
                            if eval_ns else 0.0
                        ),
                    )

        except ConnectionError:
            error = (
                "Can't reach Ollama. Is it running? "
                "(try `ollama serve`)"
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        reply = "".join(parts)

        if reply:
            tokens = (
                stats.reply_tokens
                if stats else estimate_tokens(reply)
            )

            self.history.append({
                "role": "assistant",
                "content": reply,
                "tokens": tokens,
            })

        if stats:
            self.last_stats = stats
            self.session_tokens += stats.reply_tokens

        if error:
            yield StreamEvent("error", error)

        yield StreamEvent(
            "done",
            stats=stats,
            cancelled=cancel.is_set(),
        )

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "model": self.settings.model,
                    "history": self.history,
                },
                file,
                indent=2,
                ensure_ascii=False,
            )

    def load(self, path: str) -> None:
        with open(path, encoding="utf-8") as file:
            data = json.load(file)

        history = []

        for message in data["history"]:
            if message["role"] not in ("user", "assistant"):
                raise ValueError(
                    f"Bad role in file: {message['role']!r}"
                )

            history.append({
                "role": message["role"],
                "content": message["content"],
                "tokens": int(
                    message.get("tokens")
                    or estimate_tokens(message["content"])
                ),
            })

        self.history = history
        self.last_stats = None


# --------------------------------------------------------------------------
# Visual theme
# --------------------------------------------------------------------------

BG = "#1f1f22"
PANEL = "#262629"
FIELD = "#2e2e33"
FIELD_HOVER = "#3a3a41"
FG = "#f5f5f7"
MUTED = "#b4b4bc"
SUBTLE = "#92929c"
BORDER = "#3b3b42"
FOCUS_BORDER = "#0a84ff"
INPUT_FOCUS = "#62626c"

ACCENT = "#0a84ff"
ACCENT_HOVER = "#2b95ff"
ACCENT_PRESSED = "#0872df"

BOT_BUBBLE = "#2b2b30"
USER_BUBBLE = BOT_BUBBLE
BOT_BORDER = "#333339"
USER_BORDER = BOT_BORDER

SECONDARY_BG = "#2b2b30"
SECONDARY_HOVER = "#37373e"
SECONDARY_PRESSED = "#414149"
SECONDARY_BORDER = ""
SECONDARY_FG = "#dedee3"
DISABLED_SECONDARY_BG = "#26262b"
DISABLED_SECONDARY_FG = "#696972"

ERROR = "#ff7a72"
WARNING = "#d8b876"
SUCCESS = "#70c491"

INPUT_HINT = "Return to send  ·  Shift+Return for a new line"


def pick_fonts() -> tuple[str, str]:
    """Prefer Tk's system fonts, with sensible platform fallbacks."""
    system = platform.system()

    try:
        ui_font = tkfont.nametofont(
            "TkDefaultFont"
        ).actual("family")
        mono_font = tkfont.nametofont(
            "TkFixedFont"
        ).actual("family")
    except tk.TclError:
        ui_font = ""
        mono_font = ""

    if system == "Darwin":
        return (
            ui_font or "Helvetica Neue",
            mono_font or "Menlo",
        )

    if system == "Windows":
        return (
            ui_font or "Segoe UI",
            mono_font or "Consolas",
        )

    return (
        ui_font or "DejaVu Sans",
        mono_font or "DejaVu Sans Mono",
    )


def rounded_points(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    radius: float,
) -> list[float]:
    radius = max(
        0,
        min(
            radius,
            abs(x2 - x1) / 2,
            abs(y2 - y1) / 2,
        ),
    )

    return [
        x1 + radius, y1,
        x2 - radius, y1,
        x2, y1,
        x2, y1 + radius,
        x2, y2 - radius,
        x2, y2,
        x2 - radius, y2,
        x1 + radius, y2,
        x1, y2,
        x1, y2 - radius,
        x1, y1 + radius,
        x1, y1,
    ]


# --------------------------------------------------------------------------
# Custom controls
# --------------------------------------------------------------------------

class RoundedButton(tk.Canvas):
    """A lightweight rounded button with quiet hover and pressed states."""

    def __init__(
        self,
        parent: tk.Misc,
        text: str,
        command=None,
        font: tuple = ("Helvetica Neue", 10, "normal"),
        kind: str = "secondary",
        width: Optional[int] = None,
        height: int = 34,
        radius: int = 10,
        background: str = BG,
    ) -> None:
        self.command = command
        self.kind = kind
        self.font = font
        self.radius = radius
        self.surface = background
        self._state = "normal"
        self._hovering = False
        self._pressed = False
        self._focused = False

        if width is None:
            measure_font = tkfont.Font(font=font)
            horizontal_padding = 30 if kind == "primary" else 24
            width = (
                measure_font.measure(text)
                + horizontal_padding
            )

        super().__init__(
            parent,
            width=width,
            height=height,
            bg=self.surface,
            bd=0,
            highlightthickness=0,
            relief="flat",
            cursor="hand2",
            takefocus=1,
        )

        self._shape = self.create_polygon(
            (0, 0, 0, 0),
            smooth=True,
            splinesteps=32,
            fill=SECONDARY_BG,
            outline="",
            width=1,
        )
        self._label = self.create_text(
            (0, 0),
            text=text,
            font=font,
            fill=SECONDARY_FG,
            anchor="center",
        )

        super().bind("<Configure>", self._layout)
        super().bind("<Enter>", self._on_enter)
        super().bind("<Leave>", self._on_leave)
        super().bind("<ButtonPress-1>", self._on_press)
        super().bind("<ButtonRelease-1>", self._on_release)
        super().bind("<FocusIn>", self._on_focus_in)
        super().bind("<FocusOut>", self._on_focus_out)
        super().bind("<Return>", self._on_key)
        super().bind("<space>", self._on_key)

        self.after_idle(self._layout)

    def set_state(self, state: str) -> None:
        self._state = (
            "disabled" if state == "disabled" else "normal"
        )
        self._pressed = False

        self.configure(
            cursor=(
                "arrow"
                if self._state == "disabled"
                else "hand2"
            ),
            takefocus=(
                0 if self._state == "disabled" else 1
            ),
        )
        self._redraw()

    def _layout(self, event=None) -> None:
        width = self.winfo_width()
        height = self.winfo_height()

        if width <= 1 or height <= 1:
            return

        self.coords(
            self._shape,
            *rounded_points(
                1,
                1,
                width - 1,
                height - 1,
                self.radius,
            ),
        )
        self.coords(
            self._label,
            width / 2,
            height / 2,
        )
        self.tag_raise(self._label)
        self._redraw()

    def _colors(self) -> tuple[str, str, str]:
        if self._state == "disabled":
            if self.kind == "primary":
                return "#35353b", "#85858e", ""

            return (
                DISABLED_SECONDARY_BG,
                DISABLED_SECONDARY_FG,
                "",
            )

        if self.kind == "primary":
            if self._pressed:
                background = ACCENT_PRESSED
            elif self._hovering:
                background = ACCENT_HOVER
            else:
                background = ACCENT

            outline = (
                "#8fb4ff" if self._focused else ""
            )
            return background, "#ffffff", outline

        if self._pressed:
            background = SECONDARY_PRESSED
            foreground = FG
            outline = FOCUS_BORDER
        elif self._hovering:
            background = SECONDARY_HOVER
            foreground = FG
            outline = "#454550"
        else:
            background = SECONDARY_BG
            foreground = SECONDARY_FG
            outline = SECONDARY_BORDER

        if self._focused and not self._hovering:
            outline = FOCUS_BORDER

        return background, foreground, outline

    def _redraw(self) -> None:
        background, foreground, outline = self._colors()

        self.itemconfigure(
            self._shape,
            fill=background,
            outline=outline,
        )
        self.itemconfigure(
            self._label,
            fill=foreground,
        )

    def _on_enter(self, event=None) -> None:
        self._hovering = True
        self._redraw()

    def _on_leave(self, event=None) -> None:
        self._hovering = False
        self._pressed = False
        self._redraw()

    def _on_press(self, event=None) -> str:
        if self._state != "disabled":
            self.focus_set()
            self._pressed = True
            self._redraw()

        return "break"

    def _on_release(self, event) -> str:
        was_pressed = self._pressed
        self._pressed = False
        self._redraw()

        inside = (
            0 <= event.x <= self.winfo_width()
            and 0 <= event.y <= self.winfo_height()
        )

        if (
            was_pressed
            and inside
            and self._state != "disabled"
            and self.command is not None
        ):
            self.command()

        return "break"

    def _on_key(self, event=None) -> str:
        if (
            self._state != "disabled"
            and self.command is not None
        ):
            self.command()

        return "break"

    def _on_focus_in(self, event=None) -> None:
        self._focused = True
        self._redraw()

    def _on_focus_out(self, event=None) -> None:
        self._focused = False
        self._pressed = False
        self._redraw()


class ContextMeter(tk.Canvas):
    """A thin, borderless context indicator."""

    def __init__(
        self,
        parent: tk.Misc,
        width: int = 112,
        height: int = 5,
    ) -> None:
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=PANEL,
            bd=0,
            highlightthickness=0,
            relief="flat",
        )

        self._value = 0.0

        self._track = self.create_polygon(
            (0, 0, 0, 0),
            smooth=True,
            splinesteps=24,
            fill="#343439",
            outline="",
        )
        self._fill = self.create_polygon(
            (0, 0, 0, 0),
            smooth=True,
            splinesteps=24,
            fill=ACCENT,
            outline="",
        )

        super().bind(
            "<Configure>",
            lambda event: self._redraw(),
        )

    def set(self, value: float) -> None:
        self._value = max(0.0, min(100.0, value))
        self._redraw()

    def _redraw(self) -> None:
        width = max(2, self.winfo_width())
        height = max(2, self.winfo_height())
        radius = height / 2

        self.coords(
            self._track,
            *rounded_points(
                0,
                0,
                width,
                height,
                radius,
            ),
        )

        fill_width = int(width * self._value / 100)

        if fill_width <= 1:
            self.itemconfigure(
                self._fill,
                state="hidden",
            )
            return

        self.itemconfigure(
            self._fill,
            state="normal",
        )
        self.coords(
            self._fill,
            *rounded_points(
                0,
                0,
                max(height, fill_width),
                height,
                radius,
            ),
        )


class RoundedComposer(tk.Canvas):
    """A compact composer that grows with its content, up to six lines."""

    MIN_HEIGHT = 54
    MAX_HEIGHT = 146
    PAD_X = 16
    PAD_Y = 13

    def __init__(
        self,
        parent: tk.Misc,
        font: tuple,
        placeholder: str,
    ) -> None:
        super().__init__(
            parent,
            height=self.MIN_HEIGHT,
            bg=BG,
            bd=0,
            highlightthickness=0,
            relief="flat",
        )
        self.pack_propagate(False)

        self._font = font
        self._focused = False
        self._height = self.MIN_HEIGHT
        self._resize_pending = False
        self._metrics = tkfont.Font(font=font)
        self._line_height = max(
            20,
            self._metrics.metrics("linespace") + 4,
        )

        self._shape = self.create_polygon(
            (0, 0, 0, 0),
            smooth=True,
            splinesteps=32,
            fill=FIELD,
            outline=BORDER,
            width=1,
        )

        self.text_widget = tk.Text(
            self,
            height=1,
            wrap=tk.WORD,
            bg=FIELD,
            fg=FG,
            insertbackground=FG,
            selectbackground=ACCENT,
            selectforeground="#ffffff",
            relief="flat",
            bd=0,
            highlightthickness=0,
            font=font,
            padx=0,
            pady=0,
            spacing1=2,
            spacing3=2,
            insertwidth=2,
        )

        self._text_window = self.create_window(
            (self.PAD_X, self.PAD_Y),
            window=self.text_widget,
            anchor="nw",
        )

        # A Label stays visible above the embedded Text widget.
        self.placeholder = tk.Label(
            self,
            text=placeholder,
            bg=FIELD,
            fg=SUBTLE,
            font=(font[0], 13, "normal"),
            anchor="w",
            cursor="xterm",
        )
        self._placeholder_window = self.create_window(
            (self.PAD_X, self.PAD_Y + 2),
            window=self.placeholder,
            anchor="nw",
        )
        self.placeholder.bind(
            "<Button-1>",
            lambda event: self.text_widget.focus_set(),
        )

        super().bind("<Configure>", self._layout)
        super().bind(
            "<Button-1>",
            lambda event: self.text_widget.focus_set(),
        )

        self.text_widget.bind(
            "<KeyRelease>",
            self._on_change,
        )
        self.text_widget.bind(
            "<FocusIn>",
            lambda event: self._set_focused(True),
        )
        self.text_widget.bind(
            "<FocusOut>",
            lambda event: self._set_focused(False),
        )
        self.text_widget.bind(
            "<<Paste>>",
            lambda event: self.after_idle(self._on_change),
        )
        self.text_widget.bind(
            "<<Cut>>",
            lambda event: self.after_idle(self._on_change),
        )

        self.after_idle(self._layout)

    def _layout(self, event=None) -> None:
        width = self.winfo_width()
        height = self.winfo_height()

        if width <= 1 or height <= 1:
            return

        self.coords(
            self._shape,
            *rounded_points(
                1,
                1,
                width - 2,
                height - 2,
                15,
            ),
        )

        self.coords(
            self._text_window,
            self.PAD_X,
            self.PAD_Y,
        )
        self.itemconfigure(
            self._text_window,
            width=max(40, width - (2 * self.PAD_X)),
            height=max(24, height - (2 * self.PAD_Y)),
        )
        self.coords(
            self._placeholder_window,
            self.PAD_X,
            self.PAD_Y + 2,
        )
        self._refresh_placeholder()

    def _set_focused(self, focused: bool) -> None:
        self._focused = focused
        self.itemconfigure(
            self._shape,
            outline=(
                INPUT_FOCUS if focused else BORDER
            ),
        )
        self._refresh_placeholder()

    def _on_change(self, event=None) -> None:
        self._refresh_placeholder()
        self._schedule_resize()

    def _refresh_placeholder(self) -> None:
        empty = not self.text_widget.get(
            "1.0",
            "end-1c",
        )

        self.itemconfigure(
            self._placeholder_window,
            state="normal" if empty else "hidden",
        )
        if empty:
            self.tag_raise(self._placeholder_window)
        else:
            self.tag_raise(self._text_window)

    def _schedule_resize(self) -> None:
        if self._resize_pending:
            return

        self._resize_pending = True
        self.after_idle(self._resize_to_content)

    def _resize_to_content(self) -> None:
        self._resize_pending = False

        try:
            self.text_widget.update_idletasks()
            last_line = int(
                self.text_widget.index("end-1c").split(".")[0]
            )
            line_count = 0

            for line in range(1, last_line + 1):
                line_count += max(
                    1,
                    self.text_widget.count(
                        f"{line}.0",
                        f"{line}.end",
                        "displaylines",
                    ) or 1,
                )

            content_height = (
                line_count * self._line_height
                + (2 * self.PAD_Y)
            )
            height = max(
                self.MIN_HEIGHT,
                min(self.MAX_HEIGHT, content_height),
            )

            if height != self._height:
                self._height = height
                self.configure(height=height)
        except tk.TclError:
            pass

    def set_placeholder(self, text: str) -> None:
        self.placeholder.configure(text=text)

    # Delegate the small Text API used by ChatApp.
    def get(self, *args) -> str:
        return self.text_widget.get(*args)

    def delete(self, *args) -> None:
        self.text_widget.delete(*args)
        self._on_change()

    def insert(self, *args) -> None:
        self.text_widget.insert(*args)
        self._on_change()

    def focus_set(self) -> None:
        self.text_widget.focus_set()

    def bind(self, sequence, func, add=None):
        return self.text_widget.bind(
            sequence,
            func,
            add,
        )


class RoundedBubble(tk.Canvas):
    """A rounded rectangle containing automatically wrapping text."""

    PAD_X = 20
    PAD_Y = 17
    RADIUS = 20
    MIN_WIDTH = 48
    MIN_HEIGHT = 52

    def __init__(
        self,
        parent: tk.Misc,
        text: str,
        background: str,
        foreground: str,
        font: tuple,
        pending: bool = False,
        border: Optional[str] = None,
    ) -> None:
        super().__init__(
            parent,
            bg=BG,
            bd=0,
            highlightthickness=0,
            relief="flat",
        )

        self.text = text
        self.background = background
        self.foreground = foreground
        self.font = font
        self.pending = pending
        self.border = border
        self.max_width = 580

        self.shape = self.create_polygon(
            (0, 0, 0, 0),
            smooth=True,
            splinesteps=32,
            fill=background,
            outline=border or "",
            width=1,
        )

        self.text_item = self.create_text(
            self.PAD_X,
            self.PAD_Y,
            anchor="nw",
            font=self.font,
            fill=self.foreground,
            text="",
            justify="left",
        )

        self._layout()

    def set_max_width(self, width: int) -> None:
        width = max(140, int(width))

        if width != self.max_width:
            self.max_width = width
            self._layout()

    def set_text(
        self,
        text: str,
        pending: bool = False,
    ) -> None:
        self.text = text
        self.pending = pending
        self._layout()

    def _layout(self) -> None:
        shown = self.text

        if not shown and self.pending:
            shown = "Thinking…"
        elif not shown:
            shown = " "

        text_color = (
            MUTED
            if self.pending and not self.text
            else self.foreground
        )

        # Preserve explicit blank lines as comfortable paragraph spacing.
        shown = shown.replace("\n\n", "\n\u200b\n")

        self.itemconfigure(
            self.text_item,
            text=shown,
            width=0,
            fill=text_color,
        )

        bounds = self.bbox(self.text_item)

        if bounds is None:
            return

        text_width = max(1, bounds[2] - bounds[0])
        text_height = max(1, bounds[3] - bounds[1])
        maximum_text_width = max(
            80,
            self.max_width - (2 * self.PAD_X),
        )

        if text_width > maximum_text_width:
            self.itemconfigure(
                self.text_item,
                width=maximum_text_width,
            )

            bounds = self.bbox(self.text_item)

            if bounds is not None:
                text_width = max(
                    1,
                    bounds[2] - bounds[0],
                )
                text_height = max(
                    1,
                    bounds[3] - bounds[1],
                )

        bubble_width = max(
            self.MIN_WIDTH,
            min(
                self.max_width,
                text_width + (2 * self.PAD_X),
            ),
        )
        bubble_height = max(
            self.MIN_HEIGHT,
            text_height + (2 * self.PAD_Y),
        )

        self.configure(
            width=bubble_width,
            height=bubble_height,
        )

        x1, y1 = 1, 1
        x2 = bubble_width - 1
        y2 = bubble_height - 1

        radius = min(
            self.RADIUS,
            max(1, (x2 - x1) // 2),
            max(1, (y2 - y1) // 2),
        )

        self.coords(
            self.shape,
            *rounded_points(
                x1,
                y1,
                x2,
                y2,
                radius,
            ),
        )

        self.coords(
            self.text_item,
            self.PAD_X,
            self.PAD_Y,
        )
        self.tag_raise(self.text_item)


class MessageView:
    """One message row, its bubble, and its inline editor."""

    def __init__(
        self,
        parent: tk.Misc,
        role: str,
        text: str,
        name: str,
        stamp: str,
        ui_font: str,
        message: Optional[dict] = None,
        pending: bool = False,
        on_edit=None,
        on_save=None,
        on_cancel=None,
        on_copy=None,
    ) -> None:
        self.role = role
        self.text = text
        self.message = message
        self.pending = pending
        self.ui_font = ui_font
        self.message_font = (ui_font, 14, "normal")

        self.on_edit = on_edit
        self.on_save = on_save
        self.on_cancel = on_cancel
        self.on_copy = on_copy

        self.editor: Optional[tk.Frame] = None
        self.edit_text: Optional[tk.Text] = None
        self.actions_enabled = True

        is_user = role == "user"
        self.anchor = "e" if is_user else "w"

        self.row = tk.Frame(parent, bg=BG)
        self.row.pack(
            fill="x",
            padx=24,
            pady=(8, 14),
        )

        self.inner = tk.Frame(self.row, bg=BG)
        self.inner.pack(anchor=self.anchor)

        self.header = tk.Frame(self.inner, bg=BG)
        self.header.pack(
            fill="x",
            padx=(43 if not is_user else 3, 3),
            pady=(0, 7),
        )

        self.name_label = tk.Label(
            self.header,
            text=name,
            bg=BG,
            fg=MUTED if is_user else FG,
            font=(ui_font, 11, "bold"),
        )
        self.stamp_label = tk.Label(
            self.header,
            text=stamp,
            bg=BG,
            fg=SUBTLE,
            font=(ui_font, 10, "normal"),
        )

        # Labels avoid the native macOS white-button rendering problem.
        self.edit_button = tk.Label(
            self.header,
            text="✎",
            bg=BG,
            fg=SUBTLE,
            disabledforeground="#5b5b64",
            font=(ui_font, 11, "normal"),
            width=2,
            padx=2,
            pady=0,
            cursor="hand2",
            takefocus=1,
        )

        self.copy_button = tk.Label(
            self.header,
            text="Copy",
            bg=BG,
            fg=SUBTLE,
            disabledforeground="#5b5b64",
            font=(ui_font, 10, "normal"),
            padx=4,
            pady=0,
            cursor="hand2",
            takefocus=1,
        )
        self.copy_button.bind(
            "<Button-1>",
            self._copy_click,
        )
        self.copy_button.bind(
            "<Return>",
            self._copy_click,
        )
        self.copy_button.bind(
            "<space>",
            self._copy_click,
        )
        self.copy_button.bind(
            "<Enter>",
            lambda event: self._copy_hover(True),
        )
        self.copy_button.bind(
            "<Leave>",
            lambda event: self._copy_hover(False),
        )

        self.edit_button.bind(
            "<Button-1>",
            self._edit_click,
        )
        self.edit_button.bind(
            "<Return>",
            self._edit_click,
        )
        self.edit_button.bind(
            "<space>",
            self._edit_click,
        )
        self.edit_button.bind(
            "<Enter>",
            lambda event: self._edit_hover(True),
        )
        self.edit_button.bind(
            "<Leave>",
            lambda event: self._edit_hover(False),
        )

        if is_user:
            self.edit_button.pack(
                side="right",
                padx=(5, 0),
            )
            self.copy_button.pack(
                side="right",
                padx=(5, 0),
            )
            self.stamp_label.pack(side="right")
            self.name_label.pack(
                side="right",
                padx=(0, 7),
            )
        else:
            self.name_label.pack(side="left")
            self.stamp_label.pack(
                side="left",
                padx=(7, 0),
            )
            self.edit_button.pack(
                side="left",
                padx=(5, 0),
            )
            self.copy_button.pack(
                side="left",
                padx=(5, 0),
            )

        self.content = tk.Frame(self.inner, bg=BG)
        self.content.pack(anchor=self.anchor)

        self.bubble = RoundedBubble(
            self.content,
            text=text,
            background=(
                USER_BUBBLE if is_user else BOT_BUBBLE
            ),
            foreground=FG,
            font=self.message_font,
            pending=pending,
            border=(USER_BORDER if is_user else BOT_BORDER),
        )

        if is_user:
            self.bubble.pack(side="right")
        else:
            self.avatar = tk.Canvas(
                self.content,
                width=30,
                height=30,
                bg=BG,
                bd=0,
                highlightthickness=0,
            )
            self.avatar.create_oval(
                2, 2, 28, 28,
                fill="#35353b",
                outline="#424249",
            )
            self.avatar.create_text(
                15, 15,
                text=(name[:1].upper() or "M"),
                fill=FG,
                font=(ui_font, 10, "bold"),
            )
            self.avatar.pack(
                side="left",
                anchor="n",
                padx=(0, 10),
                pady=(3, 0),
            )
            self.bubble.pack(side="left", anchor="n")

        self._update_edit_state()

    def _request_edit(self) -> None:
        if self.on_edit:
            self.on_edit(self)

    def _edit_click(self, event=None) -> str:
        if self.edit_button.cget("state") != "disabled":
            self._request_edit()

        return "break"

    def _edit_hover(self, hovering: bool) -> None:
        if self.edit_button.cget("state") == "disabled":
            return

        self.edit_button.configure(
            bg=FIELD_HOVER if hovering else BG,
            fg=FG if hovering else SUBTLE,
        )

    def _copy_click(self, event=None) -> str:
        if (
            self.copy_button.cget("state") != "disabled"
            and self.on_copy
        ):
            self.on_copy(self)

        return "break"

    def _copy_hover(self, hovering: bool) -> None:
        if self.copy_button.cget("state") == "disabled":
            return

        self.copy_button.configure(
            bg=FIELD_HOVER if hovering else BG,
            fg=FG if hovering else SUBTLE,
        )

    def _update_edit_state(self) -> None:
        enabled = (
            self.actions_enabled
            and not self.pending
            and self.message is not None
            and self.editor is None
        )

        self.edit_button.configure(
            state="normal" if enabled else "disabled",
            bg=BG,
            fg=SUBTLE if enabled else "#4b4b53",
            cursor="hand2" if enabled else "arrow",
        )
        copy_enabled = (
            self.actions_enabled
            and not self.pending
            and bool(self.text)
        )
        self.copy_button.configure(
            state="normal" if copy_enabled else "disabled",
            bg=BG,
            fg=SUBTLE if copy_enabled else "#4b4b53",
            cursor="hand2" if copy_enabled else "arrow",
        )

    def set_actions_enabled(self, enabled: bool) -> None:
        self.actions_enabled = enabled
        self._update_edit_state()

    def set_message(self, message: dict) -> None:
        self.message = message
        self._update_edit_state()

    def set_max_width(self, width: int) -> None:
        self.bubble.set_max_width(width)

        if self.edit_text is not None:
            self.edit_text.configure(
                width=self._editor_width()
            )

    def _editor_width(self) -> int:
        return max(
            25,
            min(70, int(self.bubble.max_width / 8)),
        )

    def set_text(
        self,
        text: str,
        pending: Optional[bool] = None,
    ) -> None:
        self.text = text

        if pending is not None:
            self.pending = pending

        self.bubble.set_text(
            text,
            pending=self.pending,
        )
        self._update_edit_state()

    def begin_edit(self) -> None:
        if (
            self.editor is not None
            or self.message is None
            or self.pending
        ):
            return

        self.content.pack_forget()

        self.editor = tk.Frame(self.inner, bg=BG)
        self.editor.pack(anchor=self.anchor)

        line_count = self.text.count("\n") + 1
        height = max(3, min(14, line_count + 1))

        self.edit_text = tk.Text(
            self.editor,
            width=self._editor_width(),
            height=height,
            wrap=tk.WORD,
            bg=FIELD,
            fg=FG,
            insertbackground=FG,
            selectbackground=ACCENT,
            selectforeground="#ffffff",
            relief="flat",
            font=self.message_font,
            padx=12,
            pady=10,
            spacing1=2,
            spacing3=2,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=FOCUS_BORDER,
        )
        self.edit_text.pack(fill="both", expand=True)
        self.edit_text.insert("1.0", self.text)

        buttons = tk.Frame(self.editor, bg=BG)
        buttons.pack(fill="x", pady=(7, 0))

        def save() -> None:
            if self.on_save and self.edit_text:
                self.on_save(
                    self,
                    self.edit_text.get("1.0", "end-1c"),
                    False,
                )

        def save_and_regenerate() -> None:
            if self.on_save and self.edit_text:
                self.on_save(
                    self,
                    self.edit_text.get("1.0", "end-1c"),
                    True,
                )

        def cancel() -> None:
            if self.on_cancel:
                self.on_cancel(self)

        def save_event(event=None):
            save()
            return "break"

        def cancel_event(event=None):
            cancel()
            return "break"

        RoundedButton(
            buttons,
            text="Cancel",
            command=cancel,
            font=(self.ui_font, 10, "normal"),
            height=32,
        ).pack(side="right")

        RoundedButton(
            buttons,
            text="Save",
            kind="primary",
            command=save,
            font=(self.ui_font, 10, "bold"),
            height=32,
        ).pack(side="right", padx=(0, 7))

        if self.role == "user":
            RoundedButton(
                buttons,
                text="Save & regenerate",
                command=save_and_regenerate,
                font=(self.ui_font, 10, "normal"),
                height=32,
            ).pack(side="left")

        self.edit_text.bind("<Escape>", cancel_event)
        self.edit_text.bind(
            "<Command-Return>",
            save_event,
        )
        self.edit_text.bind(
            "<Control-Return>",
            save_event,
        )

        self.edit_text.focus_set()
        self.edit_text.mark_set("insert", "end")
        self._update_edit_state()

    def cancel_edit(self) -> None:
        if self.editor is not None:
            self.editor.destroy()

        self.editor = None
        self.edit_text = None
        self.content.pack(anchor=self.anchor)
        self._update_edit_state()

    def destroy(self) -> None:
        self.row.destroy()


# --------------------------------------------------------------------------
# Main application
# --------------------------------------------------------------------------

class ChatApp:
    POLL_MS = 30

    def __init__(
        self,
        root: tk.Tk,
        client: Optional[ChatClient] = None,
    ) -> None:
        self.root = root
        self.client = client or ChatClient()
        self.events: queue.Queue[StreamEvent] = queue.Queue()
        self.cancel = threading.Event()

        self.busy = False
        self.closing = False
        self.stream_failed = False

        self.message_views: list[MessageView] = []
        self.system_labels: list[tk.Label] = []
        self.pending_view: Optional[MessageView] = None
        self.editing_view: Optional[MessageView] = None
        self.scroll_pending = False

        self.ui_font, self.mono_font = pick_fonts()

        root.title("Matthew — Ollama Chat")
        root.geometry("1000x780")
        root.configure(bg=BG)
        root.minsize(720, 560)

        self._build_style()
        self._build_widgets()

        self.client.refresh_context_limit()
        self._refresh_stats()
        self.status.set(INPUT_HINT)
        self._warn_if_no_limit()
        self._set_busy(False)

        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(self.POLL_MS, self._poll)

    # ---- setup -----------------------------------------------------------

    def _build_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure(
            "TFrame",
            background=BG,
        )
        style.configure(
            "Panel.TFrame",
            background=PANEL,
        )
        style.configure(
            "TLabel",
            background=PANEL,
            foreground=MUTED,
            font=(self.ui_font, 11, "normal"),
        )
        style.configure(
            "Title.TLabel",
            background=PANEL,
            foreground=FG,
            font=(self.ui_font, 14, "bold"),
        )
        style.configure(
            "Caption.TLabel",
            background=PANEL,
            foreground=SUBTLE,
            font=(self.ui_font, 10, "normal"),
        )
        style.configure(
            "Stats.TLabel",
            background=PANEL,
            foreground=SUBTLE,
            font=(self.ui_font, 10, "normal"),
        )

        style.configure(
            "Quiet.TCombobox",
            fieldbackground=FIELD,
            background=FIELD,
            foreground=FG,
            arrowcolor=MUTED,
            bordercolor=FIELD,
            lightcolor=FIELD,
            darkcolor=FIELD,
            borderwidth=0,
            padding=(10, 7),
            font=(self.ui_font, 12, "normal"),
        )
        style.map(
            "Quiet.TCombobox",
            fieldbackground=[
                ("disabled", FIELD),
                ("focus", FIELD),
            ],
            background=[
                ("active", FIELD_HOVER),
                ("disabled", FIELD),
            ],
            foreground=[
                ("disabled", SUBTLE),
            ],
            arrowcolor=[
                ("active", MUTED),
                ("disabled", "#4b4b53"),
            ],
            bordercolor=[
                ("focus", FOCUS_BORDER),
            ],
        )

        style.configure(
            "Chat.Vertical.TScrollbar",
            background="#505058",
            troughcolor=BG,
            bordercolor=BG,
            arrowcolor=BG,
            lightcolor="#505058",
            darkcolor="#505058",
            width=5,
        )
        style.map(
            "Chat.Vertical.TScrollbar",
            background=[
                ("active", "#666670"),
                ("pressed", "#74747e"),
            ],
        )

        try:
            style.layout(
                "Chat.Vertical.TScrollbar",
                [
                    (
                        "Vertical.Scrollbar.trough",
                        {
                            "children": [
                                (
                                    "Vertical.Scrollbar.thumb",
                                    {
                                        "expand": "1",
                                        "sticky": "nswe",
                                    },
                                )
                            ],
                            "sticky": "ns",
                        },
                    )
                ],
            )
        except tk.TclError:
            pass

        self.root.option_add(
            "*TCombobox*Listbox.background",
            FIELD,
        )
        self.root.option_add(
            "*TCombobox*Listbox.foreground",
            FG,
        )
        self.root.option_add(
            "*TCombobox*Listbox.selectBackground",
            ACCENT,
        )
        self.root.option_add(
            "*TCombobox*Listbox.selectForeground",
            "#ffffff",
        )

    def _build_widgets(self) -> None:
        # Toolbar: unchanged position.
        top = ttk.Frame(
            self.root,
            style="Panel.TFrame",
        )
        top.pack(fill="x")

        inner = ttk.Frame(
            top,
            style="Panel.TFrame",
        )
        inner.pack(
            fill="x",
            padx=22,
            pady=(18, 11),
        )

        ttk.Label(
            inner,
            text="Model",
        ).pack(side="left")

        self.model_var = tk.StringVar(
            value=self.client.settings.model
        )
        self.model_box = ttk.Combobox(
            inner,
            textvariable=self.model_var,
            style="Quiet.TCombobox",
            width=22,
            values=self.client.list_models(),
        )
        self.model_box.pack(
            side="left",
            padx=(10, 12),
        )

        ttk.Label(
            inner,
            text="Private · Local",
            style="Caption.TLabel",
        ).pack(side="right")
        self.model_box.bind(
            "<<ComboboxSelected>>",
            self._on_model_change,
        )

        # Stats row: unchanged position.
        stats = ttk.Frame(
            self.root,
            style="Panel.TFrame",
        )
        stats.pack(fill="x")

        stats_inner = ttk.Frame(
            stats,
            style="Panel.TFrame",
        )
        stats_inner.pack(
            fill="x",
            padx=22,
            pady=(0, 16),
        )

        self.ctx_label = ttk.Label(
            stats_inner,
            text="",
            style="Stats.TLabel",
        )
        self.ctx_label.pack(side="left")

        self.ctx_bar = ContextMeter(stats_inner)
        self.ctx_bar.pack(
            side="left",
            padx=10,
        )

        self.speed_label = ttk.Label(
            stats_inner,
            text="",
            style="Stats.TLabel",
        )
        self.speed_label.pack(side="right")

        self.total_label = ttk.Label(
            stats_inner,
            text="",
            style="Stats.TLabel",
        )
        self.total_label.pack(
            side="right",
            padx=14,
        )

        tk.Frame(
            self.root,
            bg="#343439",
            height=1,
        ).pack(fill="x")

        # Scrollable message list.
        chat_area = ttk.Frame(self.root)
        chat_area.pack(
            fill="both",
            expand=True,
            padx=8,
            pady=(14, 3),
        )

        self.messages_canvas = tk.Canvas(
            chat_area,
            bg=BG,
            bd=0,
            highlightthickness=0,
            relief="flat",
        )

        self.scrollbar = ttk.Scrollbar(
            chat_area,
            orient="vertical",
            style="Chat.Vertical.TScrollbar",
            command=self.messages_canvas.yview,
        )
        self.messages_canvas.configure(
            yscrollcommand=self.scrollbar.set,
        )

        self.scrollbar.pack(
            side="right",
            fill="y",
            padx=(2, 0),
            pady=8,
        )
        self.messages_canvas.pack(
            side="left",
            fill="both",
            expand=True,
        )

        self.messages_frame = tk.Frame(
            self.messages_canvas,
            bg=BG,
        )
        self.messages_window = (
            self.messages_canvas.create_window(
                (0, 0),
                window=self.messages_frame,
                anchor="nw",
            )
        )

        self.messages_frame.bind(
            "<Configure>",
            self._on_messages_configure,
        )
        self.messages_canvas.bind(
            "<Configure>",
            self._on_canvas_configure,
        )

        self._bind_wheel_tree(self.messages_canvas)
        self._build_empty_state()

        # Status line: unchanged position.
        self.status = tk.StringVar(value="")

        tk.Label(
            self.root,
            textvariable=self.status,
            fg=SUBTLE,
            bg=BG,
            height=1,
            font=(self.ui_font, 11, "normal"),
            anchor="w",
        ).pack(
            fill="x",
            padx=24,
            pady=(0, 4),
        )

        # Input: unchanged position.
        bottom = ttk.Frame(self.root)
        bottom.pack(
            fill="x",
            padx=22,
            pady=(6, 9),
        )

        self.input = RoundedComposer(
            bottom,
            font=(self.ui_font, 14, "normal"),
            placeholder=(
                f"Message {self.client.settings.model}…"
            ),
        )
        self.input.pack(
            side="left",
            fill="both",
            expand=True,
        )
        self.input.bind("<Return>", self.send)
        self.input.bind(
            "<Shift-Return>",
            lambda event: None,
        )
        self.input.focus_set()

        buttons = ttk.Frame(bottom)
        buttons.pack(
            side="right",
            padx=(10, 0),
            fill="y",
        )

        self.send_btn = RoundedButton(
            buttons,
            text="Send",
            kind="primary",
            width=82,
            height=36,
            radius=12,
            font=(self.ui_font, 12, "normal"),
            command=self.send,
        )
        self.send_btn.pack(fill="x")

        self.stop_btn = RoundedButton(
            buttons,
            text="Stop",
            width=82,
            height=30,
            radius=10,
            font=(self.ui_font, 10, "normal"),
            command=self.stop,
        )
        self.stop_btn.pack(
            fill="x",
            pady=(5, 0),
        )

        # Action row: unchanged position and order.
        actions = ttk.Frame(self.root)
        actions.pack(
            fill="x",
            padx=22,
            pady=(0, 16),
        )

        self.action_btns = []

        for text, command in [
            ("New chat", self.new_chat),
            ("Save", self.save_chat),
            ("Load", self.load_chat),
            ("Regenerate", self.regenerate),
            ("Copy last reply", self.copy_last),
        ]:
            button = RoundedButton(
                actions,
                text=text,
                command=command,
                font=(self.ui_font, 10, "normal"),
                height=32,
                radius=9,
            )
            button.pack(
                side="left",
                padx=(0, 10 if text in ("New chat", "Load") else 4),
            )
            self.action_btns.append(button)

    def _build_empty_state(self) -> None:
        self.empty_state = tk.Frame(
            self.messages_canvas,
            bg=BG,
        )
        self.empty_window = self.messages_canvas.create_window(
            (0, 0),
            window=self.empty_state,
            anchor="center",
            width=460,
        )

        avatar = tk.Canvas(
            self.empty_state,
            width=64,
            height=64,
            bg=BG,
            bd=0,
            highlightthickness=0,
        )
        avatar.create_oval(
            3, 3, 61, 61,
            fill="#35353b",
            outline="#424249",
            width=1,
        )
        avatar.create_text(
            32, 32,
            text=self.client.settings.model[:1].upper() or "M",
            fill=FG,
            font=(self.ui_font, 22, "bold"),
        )
        avatar.pack(pady=(0, 16))

        self.empty_title = tk.Label(
            self.empty_state,
            text=f"Chat with {self.client.settings.model}",
            bg=BG,
            fg=FG,
            font=(self.ui_font, 21, "bold"),
        )
        self.empty_title.pack()

        tk.Label(
            self.empty_state,
            text=(
                "A private, local roleplay conversation.\n"
                "Start with a scene, a greeting, or an opening line."
            ),
            bg=BG,
            fg=MUTED,
            justify="center",
            font=(self.ui_font, 13, "normal"),
        ).pack(pady=(10, 18))

        tk.Label(
            self.empty_state,
            text="Nothing is written to a file until you choose Save.",
            bg=BG,
            fg=SUBTLE,
            font=(self.ui_font, 11, "normal"),
        ).pack()

        self._bind_wheel_tree(self.empty_state)
        self._update_empty_state()

    def _update_empty_state(self) -> None:
        if not hasattr(self, "empty_window"):
            return

        visible = (
            not self.client.history
            and not self.message_views
        )
        self.messages_canvas.itemconfigure(
            self.empty_window,
            state="normal" if visible else "hidden",
        )

    # ---- scrolling and layout -------------------------------------------

    def _bind_wheel_tree(self, widget: tk.Misc) -> None:
        """Bind scrolling directly to chat widgets and their children."""
        widget.bind(
            "<MouseWheel>",
            self._on_mousewheel,
        )
        widget.bind(
            "<Button-4>",
            self._on_mousewheel,
        )
        widget.bind(
            "<Button-5>",
            self._on_mousewheel,
        )

        for child in widget.winfo_children():
            self._bind_wheel_tree(child)

    def _on_mousewheel(self, event: tk.Event) -> str:
        number = getattr(event, "num", None)

        if number == 4:
            pixels = -52
        elif number == 5:
            pixels = 52
        else:
            delta = getattr(event, "delta", 0)

            if not delta:
                return "break"

            if platform.system() == "Darwin":
                # Small, frequent trackpad deltas become smooth pixel steps.
                pixels = -int(delta * 4)

                if pixels == 0:
                    pixels = -1 if delta > 0 else 1

                pixels = max(-160, min(160, pixels))
            elif abs(delta) >= 120:
                pixels = -int(delta / 120) * 52
            else:
                pixels = -int(delta)

                if pixels == 0:
                    pixels = -1 if delta > 0 else 1

        self._scroll_pixels(pixels)
        return "break"

    def _scroll_pixels(self, pixels: int) -> None:
        bounds = self.messages_canvas.bbox(self.messages_window)

        if not bounds:
            return

        total_height = max(1, bounds[3] - bounds[1])
        visible_height = self.messages_canvas.winfo_height()

        if total_height <= visible_height:
            return

        first, last = self.messages_canvas.yview()
        current_top = first * total_height
        maximum_top = total_height - visible_height

        new_top = max(
            0,
            min(maximum_top, current_top + pixels),
        )

        self.messages_canvas.yview_moveto(
            new_top / total_height
        )

    def _on_messages_configure(
        self,
        event: Optional[tk.Event] = None,
    ) -> None:
        bounds = self.messages_canvas.bbox(self.messages_window)

        if bounds:
            self.messages_canvas.configure(
                scrollregion=bounds
            )

    def _on_canvas_configure(
        self,
        event: tk.Event,
    ) -> None:
        if event.width <= 1:
            return

        follow = self._at_bottom()

        self.messages_canvas.itemconfigure(
            self.messages_window,
            width=event.width,
        )
        self.messages_canvas.coords(
            self.empty_window,
            event.width / 2,
            max(130, event.height / 2 - 10),
        )
        self.messages_canvas.itemconfigure(
            self.empty_window,
            width=min(460, max(280, event.width - 60)),
        )

        bubble_width = self._bubble_width(
            event.width
        )

        for view in self.message_views:
            view.set_max_width(bubble_width)

        for label in self.system_labels:
            label.configure(
                wraplength=max(
                    220,
                    event.width - 80,
                )
            )

        if follow:
            self._scroll_to_bottom()

    def _bubble_width(
        self,
        host_width: Optional[int] = None,
    ) -> int:
        if host_width is None or host_width <= 1:
            host_width = (
                self.messages_canvas.winfo_width()
            )

        if host_width <= 1:
            host_width = 800

        return min(
            580,
            max(220, int(host_width * 0.64)),
        )

    def _at_bottom(self) -> bool:
        try:
            return (
                self.messages_canvas.yview()[1]
                >= 0.995
            )
        except tk.TclError:
            return False

    def _scroll_to_bottom(self) -> None:
        if self.closing or self.scroll_pending:
            return

        self.scroll_pending = True
        self.root.after_idle(
            self._scroll_bottom_now
        )

    def _scroll_bottom_now(self) -> None:
        self.scroll_pending = False

        if self.closing:
            return

        try:
            bounds = self.messages_canvas.bbox(self.messages_window)

            if bounds:
                self.messages_canvas.configure(
                    scrollregion=bounds
                )

            self.messages_canvas.yview_moveto(1.0)
        except tk.TclError:
            pass

    # ---- display helpers -------------------------------------------------

    def _stamp(self) -> str:
        return datetime.now().strftime("%H:%M")

    def _system(
        self,
        text: str,
        tone: str = "muted",
    ) -> None:
        follow = self._at_bottom()

        color = {
            "muted": SUBTLE,
            "error": ERROR,
            "warning": WARNING,
            "success": SUCCESS,
        }.get(tone, SUBTLE)

        label = tk.Label(
            self.messages_frame,
            text=text,
            bg=BG,
            fg=color,
            font=(self.ui_font, 10, "normal"),
            justify="center",
            wraplength=max(
                220,
                self.messages_canvas.winfo_width() - 80,
            ),
        )
        label.pack(
            fill="x",
            padx=30,
            pady=(4, 7),
        )

        self.system_labels.append(label)
        self._bind_wheel_tree(label)

        if follow:
            self._scroll_to_bottom()

    def _add_message(
        self,
        role: str,
        text: str,
        message: Optional[dict] = None,
        pending: bool = False,
        force_scroll: bool = False,
    ) -> MessageView:
        follow = force_scroll or self._at_bottom()

        view = MessageView(
            self.messages_frame,
            role=role,
            text=text,
            name=(
                "You"
                if role == "user"
                else self.client.settings.model
            ),
            stamp=self._stamp(),
            ui_font=self.ui_font,
            message=message,
            pending=pending,
            on_edit=self._begin_edit,
            on_save=self._save_edit,
            on_cancel=self._cancel_edit,
            on_copy=self._copy_message,
        )
        view.set_actions_enabled(not self.busy)
        view.set_max_width(
            self._bubble_width()
        )

        self.message_views.append(view)
        self._bind_wheel_tree(view.row)
        self._update_empty_state()

        if follow:
            self._scroll_to_bottom()

        return view

    def _write_user(self, text: str) -> MessageView:
        return self._add_message(
            "user",
            text,
            message=self.client.history[-1],
            force_scroll=True,
        )

    def _remove_view(self, view: MessageView) -> None:
        try:
            self.message_views.remove(view)
        except ValueError:
            pass

        if self.editing_view is view:
            self.editing_view = None

        view.destroy()
        self._update_empty_state()

    def _delete_last_assistant_view(self) -> bool:
        for index in range(
            len(self.message_views) - 1,
            -1,
            -1,
        ):
            view = self.message_views[index]

            if view.role == "assistant":
                view.destroy()
                del self.message_views[index]
                return True

        return False

    def _render_history(self) -> None:
        self.pending_view = None
        self.editing_view = None

        for child in self.messages_frame.winfo_children():
            child.destroy()

        self.message_views.clear()
        self.system_labels.clear()

        for message in self.client.history:
            self._add_message(
                message["role"],
                message["content"],
                message=message,
            )

        self._update_empty_state()
        self._scroll_to_bottom()

    # ---- editing ---------------------------------------------------------

    def _begin_edit(self, view: MessageView) -> None:
        if self.busy:
            return

        if view.message is None or view.pending:
            return

        if (
            self.editing_view is not None
            and self.editing_view is not view
        ):
            self.status.set(
                "Finish or cancel the current edit first."
            )
            return

        self.editing_view = view
        view.begin_edit()
        self.status.set("Editing message…")

    def _cancel_edit(self, view: MessageView) -> None:
        view.cancel_edit()

        if self.editing_view is view:
            self.editing_view = None

        self.status.set("")
        self.input.focus_set()

    def _history_index(
        self,
        message: dict,
    ) -> Optional[int]:
        for index, item in enumerate(
            self.client.history
        ):
            if item is message:
                return index

        return None

    def _save_edit(
        self,
        view: MessageView,
        text: str,
        regenerate: bool = False,
    ) -> None:
        if self.busy or view.message is None:
            return

        text = text.strip()

        if not text:
            self.status.set(
                "A message cannot be empty."
            )
            return

        index = self._history_index(view.message)

        if index is None:
            self.status.set(
                "This message is no longer in the active conversation."
            )
            return

        if regenerate and view.role == "user":
            later_count = (
                len(self.client.history) - index - 1
            )

            if later_count and not messagebox.askyesno(
                "Regenerate from here",
                f"This will remove {later_count} later "
                "message(s) from the current chat. Continue?",
            ):
                return

            view.message["content"] = text
            view.message["tokens"] = estimate_tokens(text)

            del self.client.history[index + 1:]
            self.client.last_stats = None

            view.cancel_edit()
            self.editing_view = None

            self._render_history()
            self._refresh_stats()
            self._start_generation()
            return

        # Ordinary edits never trigger generation or file saving.
        view.message["content"] = text
        view.message["tokens"] = estimate_tokens(text)

        self.client.last_stats = None

        view.cancel_edit()
        view.set_text(text, pending=False)
        self.editing_view = None

        self._refresh_stats()
        self.status.set(
            "Edited in memory. Click Save to write the chat to JSON."
        )
        self.input.focus_set()

    # ---- state -----------------------------------------------------------

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        idle = "disabled" if busy else "normal"

        self.send_btn.set_state(idle)
        self.stop_btn.set_state(
            "normal" if busy else "disabled"
        )

        for button in self.action_btns:
            button.set_state(idle)

        for view in self.message_views:
            view.set_actions_enabled(not busy)

        self.model_box.config(
            state="disabled" if busy else "normal"
        )
        self.status.set(
            f"{self.client.settings.model} is responding…"
            if busy else INPUT_HINT
        )

    def _apply_settings(self) -> None:
        self.client.settings.model = (
            self.model_var.get().strip()
            or self.client.settings.model
        )

        if hasattr(self, "input"):
            self.input.set_placeholder(
                f"Message {self.client.settings.model}…"
            )
        if hasattr(self, "empty_title"):
            self.empty_title.configure(
                text=f"Chat with {self.client.settings.model}"
            )

    def _on_model_change(
        self,
        event: Optional[tk.Event] = None,
    ) -> None:
        self._apply_settings()
        self.client.refresh_context_limit()
        self._warn_if_no_limit()
        self._refresh_stats()

    def _warn_if_no_limit(self) -> None:
        if not self.client.context_limit:
            self._system(
                "No `PARAMETER num_ctx` found in this model's "
                "Modelfile, so I can't show a context limit "
                "or trim history. Add one to the Modelfile.",
                tone="warning",
            )

    def _refresh_stats(self) -> None:
        client = self.client
        used = client.context_used
        limit = client.context_limit

        if limit:
            self.ctx_label.config(
                text=f"Context: {used:,} / {limit:,}"
            )
            self.ctx_bar.set(
                min(100, used / limit * 100)
            )
        else:
            self.ctx_label.config(
                text=f"Context: {used:,} (limit unknown)"
            )
            self.ctx_bar.set(0)

        self.total_label.config(
            text=f"Session tokens: {client.session_tokens:,}"
        )

        tokens_per_sec = (
            client.last_stats.tokens_per_sec
            if client.last_stats else 0.0
        )

        self.speed_label.config(
            text=(
                f"{tokens_per_sec:.1f} tok/s"
                if tokens_per_sec else ""
            )
        )

    # ---- actions ---------------------------------------------------------

    def send(
        self,
        event: Optional[tk.Event] = None,
    ) -> str:
        if self.editing_view is not None:
            self.status.set(
                "Finish or cancel the current edit first."
            )
            return "break"

        if not self.busy:
            text = self.input.get(
                "1.0",
                "end-1c",
            ).strip()

            if text:
                self.input.delete("1.0", "end")
                self._apply_settings()
                self.client.add_user(text)
                self._write_user(text)
                self._start_generation()

        return "break"

    def regenerate(self) -> None:
        if self.busy or not self.client.history:
            return

        if self.editing_view is not None:
            self.status.set(
                "Finish or cancel the current edit first."
            )
            return

        if self.client.pop_last_assistant():
            self._delete_last_assistant_view()

        if (
            self.client.history
            and self.client.history[-1]["role"] == "user"
        ):
            self._apply_settings()
            self._start_generation()

    def stop(self) -> None:
        self.cancel.set()

    def new_chat(self) -> None:
        if self.busy:
            return

        if self.editing_view is not None:
            self.status.set(
                "Finish or cancel the current edit first."
            )
            return

        if self.client.history and not messagebox.askyesno(
            "New chat",
            "Clear the current conversation?",
        ):
            return

        self.client.clear()
        self._render_history()
        self._refresh_stats()

    def save_chat(self) -> None:
        if self.editing_view is not None:
            self.status.set(
                "Finish or cancel the current edit before saving."
            )
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[
                ("Chat JSON", "*.json"),
            ],
        )

        if path:
            try:
                self.client.save(path)
                self.status.set(f"Saved to {path}")
            except OSError as exc:
                messagebox.showerror(
                    "Save failed",
                    str(exc),
                )

    def load_chat(self) -> None:
        if self.busy:
            return

        if self.editing_view is not None:
            self.status.set(
                "Finish or cancel the current edit first."
            )
            return

        path = filedialog.askopenfilename(
            filetypes=[
                ("Chat JSON", "*.json"),
            ]
        )

        if not path:
            return

        try:
            self.client.load(path)
        except (
            OSError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            messagebox.showerror(
                "Load failed",
                str(exc),
            )
            return

        self._render_history()
        self._refresh_stats()

    def _copy_message(self, view: MessageView) -> None:
        if view.text:
            self.root.clipboard_clear()
            self.root.clipboard_append(view.text)
            self.status.set("Message copied.")

    def copy_last(self) -> None:
        text = self.client.last_assistant_text()

        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status.set("Last reply copied.")

    # ---- generation ------------------------------------------------------

    def _start_generation(self) -> None:
        self.cancel.clear()
        self.stream_failed = False
        self._set_busy(True)

        self.pending_view = self._add_message(
            "assistant",
            "",
            pending=True,
            force_scroll=True,
        )

        threading.Thread(
            target=self._worker,
            daemon=True,
        ).start()

    def _worker(self) -> None:
        for event in self.client.stream_reply(
            self.cancel
        ):
            self.events.put(event)

    def _poll(self) -> None:
        try:
            while True:
                self._handle(
                    self.events.get_nowait()
                )
        except queue.Empty:
            pass

        if not self.closing:
            self.root.after(
                self.POLL_MS,
                self._poll,
            )

    def _handle(self, event: StreamEvent) -> None:
        if event.kind == "chunk":
            if self.pending_view is None:
                self.pending_view = self._add_message(
                    "assistant",
                    "",
                    force_scroll=True,
                )

            follow = self._at_bottom()
            view = self.pending_view

            view.set_text(
                view.text + event.text,
                pending=False,
            )
            self.status.set(
                f"{self.client.settings.model} is responding…"
            )

            if follow:
                self._scroll_to_bottom()

        elif event.kind == "error":
            self.stream_failed = True

            if (
                self.pending_view is not None
                and not self.pending_view.text
            ):
                self._remove_view(
                    self.pending_view
                )
                self.pending_view = None

            self._system(
                f"Error: {event.text}",
                tone="error",
            )

        elif event.kind == "done":
            view = self.pending_view

            if view is not None:
                if view.text:
                    if (
                        self.client.history
                        and self.client.history[-1]["role"]
                        == "assistant"
                    ):
                        view.set_message(
                            self.client.history[-1]
                        )

                    view.set_text(
                        view.text,
                        pending=False,
                    )
                else:
                    self._remove_view(view)

                    if (
                        not event.cancelled
                        and not self.stream_failed
                    ):
                        self._system(
                            "No response received. "
                            "Click Regenerate to try again."
                        )

            self.pending_view = None

            if event.cancelled:
                self._system("(stopped)")

            self.stream_failed = False
            self._set_busy(False)
            self._refresh_stats()

    def on_close(self) -> None:
        self.closing = True
        self.cancel.set()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    ChatApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
