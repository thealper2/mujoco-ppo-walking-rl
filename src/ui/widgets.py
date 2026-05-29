from tkinter import Button, Frame, Label

from .theme import ACCENT, ACCENT3, BG, BORDER, GREEN, TEXT, TEXT2, YELLOW


def styled_btn(parent, text, cmd, color=ACCENT, w=16, h=1):
    b = Button(
        parent,
        text=text,
        command=cmd,
        bg=color,
        fg=BG if color in (ACCENT, GREEN, YELLOW) else TEXT,
        font=("Courier", 9, "bold"),
        relief="flat",
        activebackground=TEXT2,
        activeforeground=BG,
        width=w,
        height=h,
        cursor="hand2",
        bd=0,
        highlightthickness=1,
        highlightbackground=BORDER,
    )
    return b


def sep(parent, color=BORDER, pady=4):
    f = Frame(parent, bg=color, height=1)
    f.pack(fill="x", padx=8, pady=pady)


def section_label(parent, text):
    Label(
        parent, text=f"  {text}  ", bg=ACCENT3, fg=BG, font=("Courier", 9, "bold")
    ).pack(fill="x", padx=8, pady=(10, 4))
