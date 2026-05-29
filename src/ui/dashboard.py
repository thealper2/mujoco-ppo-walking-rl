import copy
import json
import os
import threading
import time

import matplotlib
import numpy as np

matplotlib.use("TkAgg")
import tkinter.font as tkfont
from tkinter import (
    DISABLED,
    END,
    NORMAL,
    Button,
    Canvas,
    DoubleVar,
    Entry,
    Frame,
    IntVar,
    Label,
    Scrollbar,
    StringVar,
    Text,
    Tk,
    filedialog,
    messagebox,
    ttk,
)

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.rl import PPOAgent, parse_hidden_dims, safe_make_env

from .constants import ENV_DESC, ENVS, HP_DEFAULTS
from .theme import (
    ACCENT,
    ACCENT2,
    ACCENT3,
    BG,
    BG2,
    BG3,
    BORDER,
    GREEN,
    PLT_AX,
    PLT_BG,
    TEXT,
    TEXT2,
    YELLOW,
)
from .widgets import section_label, sep


# ============================================================
# MAIN UI CLASS
# ============================================================
class PPOUI:
    def __init__(self, root):
        self.root = root
        self.root.title("WALKER — PPO MuJoCo Training Dashboard")
        self.root.geometry("1820x1060")
        self.root.configure(bg=BG)
        self.root.minsize(1400, 800)

        # State
        self.agent = None
        self.agent_signature = None
        self.total_returns = []
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_history = []
        self.ep_lengths = []
        self.total_episodes = 0
        self.is_training = False
        self.stop_flag = False
        self.best_return = -np.inf
        self.best_model_path = None

        # Sweep state
        self.sweep_results = []
        self.sweep_running = False

        # Vars
        self.status_var = StringVar(value="IDLE  —  configure and press TRAIN")
        self.env_info_var = StringVar(value="No environment loaded")
        self.env_desc_var = StringVar(value="")
        self.device_var = StringVar(value="")
        self.eps_var = StringVar(value="0")
        self.best_var = StringVar(value="—")
        self.step_var = StringVar(value="0")

        self._detect_device()
        self._build_ui()
        self._plot_graphs()

    def _detect_device(self):
        try:
            import torch

            d = "CUDA  " if __import__("torch").cuda.is_available() else "CPU  "
            self.device_var.set(d)
        except Exception:
            self.device_var.set("CPU")

    # --------------------------
    # UI BUILD
    # --------------------------
    def _build_ui(self):
        # Top bar
        top = Frame(self.root, bg=BG2, height=48)
        top.pack(fill="x")
        top.pack_propagate(False)
        Label(
            top, text="  ◈ WALKER", bg=BG2, fg=ACCENT, font=("Courier", 16, "bold")
        ).pack(side="left", padx=12)
        Label(
            top,
            text="PPO · MuJoCo · Walking Locomotion Training Dashboard",
            bg=BG2,
            fg=TEXT2,
            font=("Courier", 10),
        ).pack(side="left")
        Label(
            top,
            textvariable=self.device_var,
            bg=BG2,
            fg=GREEN,
            font=("Courier", 10, "bold"),
        ).pack(side="right", padx=16)
        Label(top, text="DEVICE: ", bg=BG2, fg=TEXT2, font=("Courier", 9)).pack(
            side="right"
        )

        # Main layout
        body = Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True)

        # Left panel (control)
        left = Frame(body, bg=BG2, width=420)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        # Center panel (graphs)
        center = Frame(body, bg=BG)
        center.pack(side="left", fill="both", expand=True)

        # Right panel (log + sweep)
        right = Frame(body, bg=BG2, width=360)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        self._build_left(left)
        self._build_center(center)
        self._build_right(right)

    # -- LEFT PANEL ----------------------------
    def _build_left(self, parent):
        canvas = Canvas(parent, bg=BG2, highlightthickness=0)
        scroll = Scrollbar(parent, orient="vertical", command=canvas.yview, bg=BG2)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        frame = Frame(canvas, bg=BG2)
        win = canvas.create_window((0, 0), window=frame, anchor="nw")

        def on_cfg(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(win, width=canvas.winfo_width())

        frame.bind("<Configure>", on_cfg)
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win, width=e.width))

        # Environment
        section_label(frame, "ENVIRONMENT")
        ef = Frame(frame, bg=BG2)
        ef.pack(fill="x", padx=8, pady=2)
        Label(ef, text="ENV", bg=BG2, fg=TEXT2, font=("Courier", 8)).grid(
            row=0, column=0, sticky="w", padx=4, pady=3
        )
        self.env_var = StringVar(value="HalfCheetah-v4")
        cb = ttk.Combobox(
            ef,
            textvariable=self.env_var,
            values=ENVS,
            state="readonly",
            width=28,
            font=("Courier", 9),
        )
        cb.grid(row=0, column=1, padx=4, pady=3)
        self.env_var.trace("w", self._on_env_change)
        self._setup_combobox_style()

        Label(ef, text="NET", bg=BG2, fg=TEXT2, font=("Courier", 8)).grid(
            row=1, column=0, sticky="w", padx=4, pady=3
        )
        self.network_var = StringVar(value="Separate")
        ttk.Combobox(
            ef,
            textvariable=self.network_var,
            values=["Separate", "MultiHead"],
            state="readonly",
            width=28,
            font=("Courier", 9),
        ).grid(row=1, column=1, padx=4, pady=3)

        self.env_desc_lbl = Label(
            frame,
            textvariable=self.env_desc_var,
            bg=BG2,
            fg=TEXT2,
            font=("Courier", 7),
            wraplength=380,
            justify="left",
        )
        self.env_desc_lbl.pack(fill="x", padx=12, pady=2)
        self._on_env_change()

        # Hyperparams
        section_label(frame, "HYPERPARAMETERS")
        self.entries = {}

        hp_groups = [
            [
                ("EPISODES", "episodes"),
                ("LR", "lr"),
                ("GAMMA", "gamma"),
                ("GAE λ", "gae_lambda"),
                ("PPO CLIP ε", "clip_eps"),
                ("PPO EPOCHS", "ppo_epochs"),
            ],
            [
                ("ENTROPY C", "entropy_coef"),
                ("VALUE C", "value_coef"),
                ("BUFFER", "buffer_size"),
                ("MINIBATCH", "minibatch_size"),
                ("HIDDEN", "hidden_dims"),
                ("GRAD CLIP", "grad_clip"),
                ("NORM ADV", "normalize_adv"),
                ("SLEEP", "render_sleep"),
            ],
        ]
        col_f = [Frame(frame, bg=BG2), Frame(frame, bg=BG2)]
        cf = Frame(frame, bg=BG2)
        cf.pack(fill="x", padx=4)
        c0 = Frame(cf, bg=BG2)
        c0.pack(side="left", fill="both", expand=True)
        c1 = Frame(cf, bg=BG2)
        c1.pack(side="left", fill="both", expand=True)
        cols = [c0, c1]
        for ci, group in enumerate(hp_groups):
            for ri, (lbl, key) in enumerate(group):
                Label(
                    cols[ci], text=lbl, bg=BG2, fg=TEXT2, font=("Courier", 7, "bold")
                ).grid(row=ri, column=0, sticky="w", padx=(8, 2), pady=2)
                e = Entry(
                    cols[ci],
                    justify="center",
                    width=11,
                    bg=BG3,
                    fg=ACCENT,
                    insertbackground=ACCENT,
                    relief="flat",
                    font=("Courier", 9),
                    highlightthickness=1,
                    highlightbackground=BORDER,
                )
                e.insert(0, HP_DEFAULTS[key])
                e.grid(row=ri, column=1, padx=(0, 6), pady=2)
                self.entries[key] = e

        # Stat badges
        sep(frame)
        bf = Frame(frame, bg=BG2)
        bf.pack(fill="x", padx=8, pady=4)
        for i, (lbl, var, col) in enumerate(
            [
                ("EP", self.eps_var, ACCENT),
                ("BEST", self.best_var, GREEN),
                ("STEPS", self.step_var, YELLOW),
            ]
        ):
            f = Frame(bf, bg=BG3, highlightbackground=BORDER, highlightthickness=1)
            f.pack(side="left", expand=True, fill="x", padx=3, pady=2)
            Label(f, text=lbl, bg=BG3, fg=TEXT2, font=("Courier", 7)).pack(pady=(4, 0))
            Label(
                f, textvariable=var, bg=BG3, fg=col, font=("Courier", 11, "bold")
            ).pack(pady=(0, 4))

        # Controls
        section_label(frame, "CONTROLS")
        cf2 = Frame(frame, bg=BG2)
        cf2.pack(fill="x", padx=8, pady=4)
        btns = [
            ("▶  TRAIN", self.run_training, ACCENT, ACCENT),
            ("■  STOP", self.stop_training, ACCENT2, ACCENT2),
            ("▷  TEST", self.test_animation, GREEN, GREEN),
            ("⟳  RESET", self.reset_all, TEXT2, TEXT2),
            ("↓  SAVE MODEL", self.save_model, YELLOW, YELLOW),
            ("↑  LOAD MODEL", self.load_model, YELLOW, YELLOW),
        ]
        for i, (txt, cmd, fg, hl) in enumerate(btns):
            b = Button(
                cf2,
                text=txt,
                command=cmd,
                bg=BG3,
                fg=fg,
                font=("Courier", 9, "bold"),
                relief="flat",
                width=17,
                height=1,
                cursor="hand2",
                activebackground=BORDER,
                activeforeground=fg,
                highlightthickness=1,
                highlightbackground=hl,
            )
            b.grid(row=i // 2, column=i % 2, padx=3, pady=3, sticky="ew")

        # Status
        sep(frame)
        Label(frame, text="STATUS", bg=BG2, fg=TEXT2, font=("Courier", 7, "bold")).pack(
            anchor="w", padx=12
        )
        Label(
            frame,
            textvariable=self.status_var,
            bg=BG2,
            fg=TEXT,
            font=("Courier", 8),
            wraplength=380,
            justify="left",
            anchor="w",
        ).pack(fill="x", padx=12, pady=6)

        # Env info
        Label(
            frame,
            textvariable=self.env_info_var,
            bg=BG2,
            fg=ACCENT3,
            font=("Courier", 8, "bold"),
            wraplength=380,
            justify="left",
        ).pack(fill="x", padx=12, pady=4)

        # Sweep config
        section_label(frame, "HYPERPARAMETER SWEEP")
        sf = Frame(frame, bg=BG2)
        sf.pack(fill="x", padx=8, pady=4)
        self.sweep_entries = {}
        sweep_params = [
            ("LR VALUES", "sweep_lrs", "0.0001,0.0003,0.001"),
            ("CLIP VALUES", "sweep_clips", "0.1,0.2,0.3"),
            ("EPOCHS", "sweep_epochs", "5,10"),
            ("EPISODES/RUN", "sweep_ep", "500"),
        ]
        for ri, (lbl, key, default) in enumerate(sweep_params):
            Label(sf, text=lbl, bg=BG2, fg=TEXT2, font=("Courier", 7, "bold")).grid(
                row=ri, column=0, sticky="w", padx=4, pady=2
            )
            e = Entry(
                sf,
                width=22,
                bg=BG3,
                fg=YELLOW,
                insertbackground=YELLOW,
                relief="flat",
                font=("Courier", 9),
                highlightthickness=1,
                highlightbackground=BORDER,
            )
            e.insert(0, default)
            e.grid(row=ri, column=1, padx=4, pady=2)
            self.sweep_entries[key] = e

        Button(
            frame,
            text="⚡  RUN SWEEP",
            command=self.run_sweep,
            bg=ACCENT3,
            fg=BG,
            font=("Courier", 9, "bold"),
            relief="flat",
            cursor="hand2",
            height=1,
        ).pack(fill="x", padx=8, pady=(4, 12))

    # -- CENTER PANEL --------------------------
    def _build_center(self, parent):
        Label(
            parent,
            text="TRAINING METRICS",
            bg=BG,
            fg=ACCENT,
            font=("Courier", 13, "bold"),
        ).pack(pady=(10, 4))
        self.fig = plt.Figure(figsize=(11, 8), facecolor=PLT_BG)
        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas_widget.get_tk_widget().pack(
            fill="both", expand=True, padx=8, pady=6
        )

    # -- RIGHT PANEL ---------------------------
    def _build_right(self, parent):
        Label(
            parent, text="TRAINING LOG", bg=BG2, fg=ACCENT, font=("Courier", 11, "bold")
        ).pack(pady=(12, 4))

        log_frame = Frame(parent, bg=BG2)
        log_frame.pack(fill="both", expand=True, padx=6)
        scroll = Scrollbar(log_frame, bg=BG2)
        scroll.pack(side="right", fill="y")
        self.log_text = Text(
            log_frame,
            bg=BG3,
            fg=TEXT,
            font=("Courier", 8),
            relief="flat",
            yscrollcommand=scroll.set,
            state=DISABLED,
            wrap="word",
            highlightthickness=0,
        )
        self.log_text.pack(side="left", fill="both", expand=True)
        scroll.config(command=self.log_text.yview)

        # Tag colors
        self.log_text.tag_config("ep", foreground=ACCENT)
        self.log_text.tag_config("ret", foreground=GREEN)
        self.log_text.tag_config("loss", foreground=ACCENT2)
        self.log_text.tag_config("best", foreground=YELLOW)
        self.log_text.tag_config("info", foreground=TEXT2)
        self.log_text.tag_config("sweep", foreground=ACCENT3)
        self.log_text.tag_config("err", foreground=ACCENT2)

        # Sweep results
        sep(parent)
        Label(
            parent,
            text="SWEEP RESULTS",
            bg=BG2,
            fg=ACCENT3,
            font=("Courier", 10, "bold"),
        ).pack(pady=(6, 2))
        self.sweep_text = Text(
            parent,
            bg=BG3,
            fg=ACCENT3,
            font=("Courier", 7),
            relief="flat",
            height=12,
            state=DISABLED,
            wrap="word",
            highlightthickness=0,
        )
        self.sweep_text.pack(fill="x", padx=6, pady=4)

    # --------------------------
    # COMBOBOX STYLE
    # --------------------------
    def _setup_combobox_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "TCombobox",
            fieldbackground=BG3,
            background=BG3,
            foreground=TEXT,
            arrowcolor=ACCENT,
            bordercolor=BORDER,
            selectbackground=BORDER,
            selectforeground=ACCENT,
            insertcolor=ACCENT,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", BG3)],
            foreground=[("readonly", TEXT)],
            selectbackground=[("readonly", BORDER)],
        )

    def _on_env_change(self, *_):
        env = self.env_var.get() if hasattr(self, "env_var") else "HalfCheetah-v4"
        self.env_desc_var.set(ENV_DESC.get(env, ""))

    # --------------------------
    # HELPERS
    # --------------------------
    def _rf(self, k):
        return float(self.entries[k].get())

    def _ri(self, k):
        return int(float(self.entries[k].get()))

    def _rb(self, k):
        return bool(int(float(self.entries[k].get())))

    def _hd(self):
        return parse_hidden_dims(self.entries["hidden_dims"].get())

    def _safe_ui(self, fn, *a, **kw):
        self.root.after(0, lambda: fn(*a, **kw))

    def _log(self, msg, tag="info"):
        def _do():
            self.log_text.config(state=NORMAL)
            self.log_text.insert(END, msg + "\n", tag)
            self.log_text.see(END)
            self.log_text.config(state=DISABLED)

        self.root.after(0, _do)

    def _signature(self, overrides=None):
        kw = {
            k: self._rf(k)
            if k not in ("ppo_epochs", "buffer_size", "minibatch_size", "normalize_adv")
            else self._ri(k)
            if k != "normalize_adv"
            else self._rb(k)
            for k in (
                "lr",
                "gamma",
                "gae_lambda",
                "clip_eps",
                "ppo_epochs",
                "entropy_coef",
                "value_coef",
                "buffer_size",
                "minibatch_size",
                "grad_clip",
                "normalize_adv",
            )
        }
        if overrides:
            kw.update(overrides)
        return (
            self.env_var.get(),
            self.network_var.get(),
            *kw.values(),
            tuple(self._hd()),
        )

    # --------------------------
    # AGENT INIT
    # --------------------------
    def _make_agent(self, overrides=None):
        env = safe_make_env(self.env_var.get())
        obs, _ = env.reset()
        sd = int(np.asarray(obs, np.float32).shape[0])
        asp = env.action_space
        kw = dict(
            state_dim=sd,
            action_space=asp,
            network_type=self.network_var.get(),
            lr=self._rf("lr"),
            gamma=self._rf("gamma"),
            gae_lambda=self._rf("gae_lambda"),
            clip_eps=self._rf("clip_eps"),
            ppo_epochs=self._ri("ppo_epochs"),
            entropy_coef=self._rf("entropy_coef"),
            value_coef=self._rf("value_coef"),
            hidden_dims=self._hd(),
            minibatch_size=self._ri("minibatch_size"),
            buffer_size=self._ri("buffer_size"),
            grad_clip=self._rf("grad_clip"),
            normalize_adv=self._rb("normalize_adv"),
        )
        if overrides:
            kw.update(overrides)
        agent = PPOAgent(**kw)
        ai = f"Continuous{asp.shape}"
        env.close()
        return agent, sd, ai

    def _ensure_agent(self):
        sig = self._signature()
        if self.agent is not None:
            if self.agent_signature == sig:
                return True
            messagebox.showwarning(
                "Warning", "Hyperparameters changed — RESET ALL first."
            )
            return False
        try:
            self.agent, sd, ai = self._make_agent()
        except Exception as e:
            messagebox.showerror("Env Error", str(e))
            return False
        self.agent_signature = sig
        self.env_info_var.set(
            f"ENV: {self.env_var.get()}  |  STATE: {sd}  |  ACT: {ai}"
        )
        self._log(f"Agent created — {self.env_var.get()}", "info")
        self.status_var.set(f"Agent ready  |  {self.env_var.get()}")
        return True

    # --------------------------
    # TRAINING
    # --------------------------
    def run_training(self):
        if self.is_training:
            return
        if not self._ensure_agent():
            return
        self.stop_flag = False
        self.is_training = True
        n = self._ri("episodes")
        threading.Thread(target=self._train_loop, args=(n,), daemon=True).start()

    def _train_loop(self, n_episodes):
        try:
            env = safe_make_env(self.env_var.get())
            steps_total = 0
            for ep_i in range(n_episodes):
                if self.stop_flag:
                    break
                obs, _ = env.reset()
                state = np.asarray(obs, np.float32)
                done = False
                total_ret = 0.0
                ep_steps = 0
                ep_al, ep_cl = [], []
                self.agent.buffer.clear()
                last_state, last_done = state, False

                while not done and not self.stop_flag:
                    action, lp, val = self.agent.select_action_train(state)
                    nobs, reward, term, trunc, _ = env.step(action)
                    done = term or trunc
                    ns = np.asarray(nobs, np.float32)
                    self.agent.add_transition(state, action, reward, done, lp, val)
                    total_ret += reward
                    ep_steps += 1
                    last_state, last_done = ns, done

                    if self.agent.ready_to_update():
                        al, cl = self.agent.update(last_state, last_done)
                        if al is not None:
                            ep_al.append(al)
                        if cl is not None:
                            ep_cl.append(cl)
                    state = ns

                if len(self.agent.buffer) > 0:
                    al, cl = self.agent.update(last_state, last_done)
                    if al is not None:
                        ep_al.append(al)
                    if cl is not None:
                        ep_cl.append(cl)

                steps_total += ep_steps
                a_loss = float(np.mean(ep_al)) if ep_al else float("nan")
                c_loss = float(np.mean(ep_cl)) if ep_cl else float("nan")
                ent = float(np.nanmean(ep_al)) if ep_al else 0.0  # placeholder

                self.total_returns.append(float(total_ret))
                self.actor_losses.append(a_loss)
                self.critic_losses.append(c_loss)
                self.ep_lengths.append(ep_steps)
                self.total_episodes += 1

                is_best = total_ret > self.best_return
                if is_best:
                    self.best_return = total_ret

                # UI updates
                self._safe_ui(self.eps_var.set, str(self.total_episodes))
                self._safe_ui(self.best_var.set, f"{self.best_return:.0f}")
                self._safe_ui(self.step_var.set, str(steps_total))
                self._safe_ui(
                    self.status_var.set,
                    f"EP {self.total_episodes}  RET {total_ret:.1f}  "
                    f"A_LOSS {a_loss:.4f}  C_LOSS {c_loss:.4f}",
                )
                self._safe_ui(self._plot_graphs)

                tag = "best" if is_best else "ret"
                prefix = "★ " if is_best else "  "
                self._log(
                    f"{prefix}EP {self.total_episodes:>5} | "
                    f"RET {total_ret:>9.2f} | "
                    f"LEN {ep_steps:>5} | "
                    f"A {a_loss:>8.4f} | "
                    f"C {c_loss:>8.4f}",
                    tag,
                )

            env.close()
            if self.stop_flag:
                self._safe_ui(
                    self.status_var.set,
                    f"Paused at EP {self.total_episodes}  |  RUN to continue",
                )
                self._log("— training paused —", "info")
            else:
                self._safe_ui(
                    self.status_var.set,
                    f"Done! {self.total_episodes} episodes  |  best: {self.best_return:.1f}",
                )
                self._log(f"Training complete. Best: {self.best_return:.2f}", "best")
        except Exception as e:
            self._safe_ui(messagebox.showerror, "Training Error", str(e))
            self._log(f"ERROR: {e}", "err")
        finally:
            self.is_training = False
            self.stop_flag = False

    def stop_training(self):
        self.stop_flag = True
        self.status_var.set("STOP requested…")
        self._log("Stop requested.", "info")

    # --------------------------
    # TEST / RENDER
    # --------------------------
    def test_animation(self):
        if self.agent is None:
            messagebox.showwarning("No Agent", "Train or load a model first.")
            return
        snap = self.agent.build_snapshot()
        threading.Thread(target=self._test_loop, args=(snap,), daemon=True).start()

    def _test_loop(self, snapshot):
        try:
            env = safe_make_env(self.env_var.get(), render_mode="human")
            obs, _ = env.reset()
            state = np.asarray(obs, np.float32)
            done = False
            total_ret = 0.0
            st = self._rf("render_sleep")
            self._safe_ui(self.status_var.set, "Testing…  MuJoCo window open")
            self._log("Test episode started.", "info")
            while not done:
                action = self.agent.select_action_eval(state, snapshot=snapshot)
                nobs, r, term, trunc, _ = env.step(action)
                done = term or trunc
                state = np.asarray(nobs, np.float32)
                total_ret += r
                time.sleep(st)
            env.close()
            self._safe_ui(self.status_var.set, f"Test done  |  return: {total_ret:.2f}")
            self._log(f"Test return: {total_ret:.2f}", "best")
        except Exception as e:
            self._safe_ui(messagebox.showerror, "Test Error", str(e))
            self._log(f"Test ERROR: {e}", "err")

    # --------------------------
    # SAVE / LOAD
    # --------------------------
    def save_model(self):
        if self.agent is None:
            messagebox.showwarning("No Agent", "No model to save.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".pt",
            filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")],
            initialfile=f"ppo_{self.env_var.get().replace('-', '_')}.pt",
        )
        if not path:
            return
        self.agent.save(path)
        # Also save metadata
        meta = {
            "env": self.env_var.get(),
            "network_type": self.network_var.get(),
            "total_episodes": self.total_episodes,
            "best_return": self.best_return,
            "returns": self.total_returns[-50:],
        }
        with open(path.replace(".pt", "_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        self._log(f"Saved → {os.path.basename(path)}", "best")
        self.status_var.set(f"Saved: {os.path.basename(path)}")

    def load_model(self):
        path = filedialog.askopenfilename(
            filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")]
        )
        if not path:
            return
        # Try to read metadata
        meta_path = path.replace(".pt", "_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            env_name = meta.get("env", self.env_var.get())
            self.env_var.set(env_name)
            self._log(
                f"Meta: env={env_name}, best={meta.get('best_return', '?'):.1f}", "info"
            )
        try:
            if self.agent is None:
                self._ensure_agent()
            if self.agent:
                self.agent.load(path)
                self._log(f"Loaded ← {os.path.basename(path)}", "best")
                self.status_var.set(f"Model loaded: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    # --------------------------
    # HYPERPARAMETER SWEEP
    # --------------------------
    def run_sweep(self):
        if self.is_training:
            messagebox.showwarning("Busy", "Stop training first.")
            return
        if self.sweep_running:
            messagebox.showwarning("Busy", "Sweep already running.")
            return
        threading.Thread(target=self._sweep_loop, daemon=True).start()

    def _sweep_loop(self):
        self.sweep_running = True
        try:
            lrs = [
                float(x.strip())
                for x in self.sweep_entries["sweep_lrs"].get().split(",")
            ]
            clips = [
                float(x.strip())
                for x in self.sweep_entries["sweep_clips"].get().split(",")
            ]
            epochs = [
                int(x.strip())
                for x in self.sweep_entries["sweep_epochs"].get().split(",")
            ]
            ep_run = int(self.sweep_entries["sweep_ep"].get())

            combos = [(lr, cl, ep) for lr in lrs for cl in clips for ep in epochs]
            self._log(f"⚡ Sweep: {len(combos)} configs × {ep_run} eps", "sweep")
            results = []

            def clear_sweep_text():
                self.sweep_text.config(state=NORMAL)
                self.sweep_text.delete("1.0", END)
                self.sweep_text.config(state=DISABLED)

            self._safe_ui(clear_sweep_text)

            for ci, (lr, cl, ep_cnt) in enumerate(combos):
                if not self.sweep_running:
                    break
                self._log(
                    f"  [{ci + 1}/{len(combos)}] lr={lr}  clip={cl}  epochs={ep_cnt}",
                    "sweep",
                )
                try:
                    agent, _, _ = self._make_agent(
                        overrides={"lr": lr, "clip_eps": cl, "ppo_epochs": ep_cnt}
                    )
                    env = safe_make_env(self.env_var.get())
                    ep_returns = []
                    for _ in range(ep_cnt):
                        obs, _ = env.reset()
                        s = np.asarray(obs, np.float32)
                        done = False
                        total_r = 0.0
                        agent.buffer.clear()
                        last_s, last_d = s, False
                        while not done:
                            a, lp, v = agent.select_action_train(s)
                            ns_obs, r, term, trunc, _ = env.step(a)
                            done = term or trunc
                            ns = np.asarray(ns_obs, np.float32)
                            agent.add_transition(s, a, r, done, lp, v)
                            total_r += r
                            last_s, last_d = ns, done
                            if agent.ready_to_update():
                                agent.update(last_s, last_d)
                            s = ns
                        if len(agent.buffer) > 0:
                            agent.update(last_s, last_d)
                        ep_returns.append(total_r)
                    env.close()
                    avg = float(np.mean(ep_returns[-20:]))
                    results.append(
                        {"lr": lr, "clip": cl, "epochs": ep_cnt, "avg_ret": avg}
                    )
                    self._log(f"    → avg_return(last20): {avg:.2f}", "ret")
                except Exception as e:
                    self._log(f"    ✗ {e}", "err")

            self.sweep_results = sorted(results, key=lambda x: -x["avg_ret"])
            self._safe_ui(self._update_sweep_display)
            self._log("⚡ Sweep complete!", "sweep")
        finally:
            self.sweep_running = False

    def _update_sweep_display(self):
        self.sweep_text.config(state=NORMAL)
        self.sweep_text.delete("1.0", END)
        self.sweep_text.insert(
            END, f"{'RANK':<5} {'LR':<8} {'CLIP':<7} {'EP':<5} {'AVG RET'}\n"
        )
        self.sweep_text.insert(END, "─" * 38 + "\n")
        for i, r in enumerate(self.sweep_results[:12]):
            line = f"#{i + 1:<4} {r['lr']:<8} {r['clip']:<7} {r['epochs']:<5} {r['avg_ret']:>8.2f}\n"
            self.sweep_text.insert(END, line)
        self.sweep_text.config(state=DISABLED)

    # --------------------------
    # GRAPHS
    # --------------------------
    def _plot_graphs(self):
        self.fig.clear()
        self.fig.patch.set_facecolor(PLT_BG)
        gs = gridspec.GridSpec(
            2,
            2,
            figure=self.fig,
            hspace=0.45,
            wspace=0.35,
            left=0.08,
            right=0.97,
            top=0.93,
            bottom=0.08,
        )
        axes = [self.fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

        def style_ax(ax, title, xlabel, ylabel):
            ax.set_facecolor(PLT_AX)
            ax.set_title(title, color=ACCENT, fontsize=9, fontfamily="monospace", pad=6)
            ax.set_xlabel(xlabel, color=TEXT2, fontsize=7, fontfamily="monospace")
            ax.set_ylabel(ylabel, color=TEXT2, fontsize=7, fontfamily="monospace")
            ax.tick_params(colors=TEXT2, labelsize=6)
            for spine in ax.spines.values():
                spine.set_edgecolor(BORDER)
            ax.grid(True, color=BORDER, alpha=0.5, linewidth=0.5)

        n = len(self.total_returns)

        # 1) Total Return
        ax = axes[0]
        style_ax(ax, "Total Return", "Episode", "Return")
        if n:
            ax.plot(self.total_returns, color=ACCENT, linewidth=0.8, alpha=0.5)
            # rolling mean
            w = max(1, n // 20)
            rm = np.convolve(self.total_returns, np.ones(w) / w, mode="valid")
            ax.plot(range(w - 1, n), rm, color=GREEN, linewidth=1.5, label=f"MA-{w}")
            ax.axhline(
                self.best_return, color=YELLOW, linewidth=0.8, linestyle="--", alpha=0.7
            )
            ax.legend(fontsize=6, facecolor=PLT_AX, edgecolor=BORDER, labelcolor=TEXT2)

        # 2) Actor Loss
        ax = axes[1]
        style_ax(ax, "Actor Loss", "Episode", "Loss")
        if n:
            vals = [v for v in self.actor_losses if not np.isnan(v)]
            if vals:
                x_vals = [i for i, v in enumerate(self.actor_losses) if not np.isnan(v)]
                ax.plot(x_vals, vals, color=ACCENT2, linewidth=0.9)

        # 3) Critic Loss
        ax = axes[2]
        style_ax(ax, "Critic Loss", "Episode", "Loss")
        if n:
            vals = [v for v in self.critic_losses if not np.isnan(v)]
            if vals:
                x_vals = [
                    i for i, v in enumerate(self.critic_losses) if not np.isnan(v)
                ]
                ax.plot(x_vals, vals, color=ACCENT3, linewidth=0.9)

        # 4) Episode Length
        ax = axes[3]
        style_ax(ax, "Episode Length", "Episode", "Steps")
        if self.ep_lengths:
            ax.bar(
                range(len(self.ep_lengths)),
                self.ep_lengths,
                color=YELLOW,
                alpha=0.5,
                width=1.0,
            )
            w = max(1, len(self.ep_lengths) // 20)
            rm = np.convolve(self.ep_lengths, np.ones(w) / w, mode="valid")
            ax.plot(range(w - 1, len(self.ep_lengths)), rm, color=ACCENT, linewidth=1.5)

        self.canvas_widget.draw()

    # --------------------------
    # RESET
    # --------------------------
    def reset_all(self):
        if self.is_training:
            self.stop_training()
        self.sweep_running = False
        self.agent = None
        self.agent_signature = None
        self.total_returns = []
        self.actor_losses = []
        self.critic_losses = []
        self.ep_lengths = []
        self.total_episodes = 0
        self.best_return = -np.inf
        for k, v in HP_DEFAULTS.items():
            self.entries[k].delete(0, "end")
            self.entries[k].insert(0, v)
        self.env_var.set("HalfCheetah-v4")
        self.network_var.set("Separate")
        self.env_info_var.set("No environment loaded")
        self.eps_var.set("0")
        self.best_var.set("—")
        self.step_var.set("0")
        self.status_var.set("Reset complete  |  configure and press TRAIN")
        self._log("─── RESET ───", "info")
        self._plot_graphs()
