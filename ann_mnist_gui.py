"""
ANN MNIST GUI — Full-featured Neural Network Studio
AIN58XS Assignment | Cape Peninsula University of Technology

Requires:
    pip install tensorflow customtkinter pillow scikit-learn seaborn matplotlib numpy
"""

import os
import threading
import io
import numpy as np
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ── Lazy imports ───────────────────────────────────────────────────────────────
tf = None
mnist = None
Sequential = None
Dense = None
Dropout = None
to_categorical = None
Adam = None
SGD = None
RMSprop = None
confusion_matrix_fn = None
classification_report_fn = None

def load_tf():
    global tf, mnist, Sequential, Dense, Dropout, to_categorical
    global Adam, SGD, RMSprop, confusion_matrix_fn, classification_report_fn
    if tf is None:
        import tensorflow as _tf
        tf = _tf
        from tensorflow.keras.datasets import mnist as _mnist
        from tensorflow.keras.models import Sequential as _Seq
        from tensorflow.keras.layers import Dense as _Dense, Dropout as _Drop
        from tensorflow.keras.utils import to_categorical as _tc
        from tensorflow.keras.optimizers import Adam as _Adam, SGD as _SGD, RMSprop as _RMS
        from sklearn.metrics import confusion_matrix as _cm, classification_report as _cr
        mnist        = _mnist
        Sequential   = _Seq
        Dense        = _Dense
        Dropout      = _Drop
        to_categorical = _tc
        Adam         = _Adam
        SGD          = _SGD
        RMSprop      = _RMS
        confusion_matrix_fn      = _cm
        classification_report_fn = _cr

# ── Theme ──────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

ACCENT   = "#4F8EF7"
ACCENT2  = "#A78BFA"
ACCENT3  = "#34D399"
BG_DARK  = "#0D0F18"
BG_CARD  = "#161926"
BG_INPUT = "#1F2235"
TEXT_P   = "#E8EAF0"
TEXT_S   = "#8B91A8"
SUCCESS  = "#34D399"
WARNING  = "#FBBF24"
DANGER   = "#F87171"
BORDER   = "#2A2E45"

# ── Main App ───────────────────────────────────────────────────────────────────
class ANNApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("ANN MNIST — Neural Network Studio  |  AIN58XS")
        self.geometry("1380x860")
        self.minsize(1100, 700)
        self.configure(fg_color=BG_DARK)

        # State
        self.model        = None
        self.history      = None
        self.x_train = self.y_train = self.x_test = self.y_test = None
        self.x_train_raw  = self.x_test_raw = None
        self.y_train_int  = self.y_test_int = None
        self.data_loaded  = False
        self.training     = False
        self._stop_training = False

        self._build_ui()

    # ── UI Construction ────────────────────────────────────────────────────────
    def _build_ui(self):
        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=210, fg_color=BG_CARD, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        ctk.CTkLabel(self.sidebar, text="ANN\nSTUDIO",
                     font=ctk.CTkFont("Courier", 24, "bold"),
                     text_color=ACCENT).pack(pady=(30, 2))
        ctk.CTkLabel(self.sidebar, text="AIN58XS · MNIST",
                     font=ctk.CTkFont(size=10), text_color=TEXT_S).pack(pady=(0, 20))
        ctk.CTkFrame(self.sidebar, height=1, fg_color=BORDER).pack(fill="x", padx=16, pady=4)

        self.nav_buttons = {}
        pages = [
            ("📊  Dataset",      "dataset"),
            ("🏗  Architecture", "arch"),
            ("🚀  Train",        "train"),
            ("🔬  Experiments",  "experiments"),
            ("📈  Evaluate",     "evaluate"),
            ("🎨  Predict",      "predict"),
        ]
        for label, key in pages:
            btn = ctk.CTkButton(
                self.sidebar, text=label, anchor="w",
                fg_color="transparent", hover_color=BG_INPUT,
                text_color=TEXT_S, font=ctk.CTkFont(size=13),
                height=42, corner_radius=8,
                command=lambda k=key: self._show_page(k)
            )
            btn.pack(fill="x", padx=12, pady=2)
            self.nav_buttons[key] = btn

        ctk.CTkFrame(self.sidebar, height=1, fg_color=BORDER).pack(
            fill="x", padx=16, pady=(12, 4), side="bottom")
        self.status_dot = ctk.CTkLabel(
            self.sidebar, text="● No model loaded",
            font=ctk.CTkFont(size=11), text_color=TEXT_S)
        self.status_dot.pack(pady=8, side="bottom")

        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.pack(side="left", fill="both", expand=True)

        self.pages = {}
        self._build_page_dataset()
        self._build_page_arch()
        self._build_page_train()
        self._build_page_experiments()
        self._build_page_evaluate()
        self._build_page_predict()

        self._show_page("dataset")

    def _show_page(self, key):
        for frame in self.pages.values():
            frame.pack_forget()
        self.pages[key].pack(fill="both", expand=True, padx=24, pady=24)
        for k, btn in self.nav_buttons.items():
            btn.configure(
                fg_color=BG_INPUT if k == key else "transparent",
                text_color=TEXT_P if k == key else TEXT_S)

    def _card(self, parent, title=None, **kwargs):
        outer = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=12,
                              border_width=1, border_color=BORDER, **kwargs)
        if title:
            ctk.CTkLabel(outer, text=title,
                         font=ctk.CTkFont(size=12, weight="bold"),
                         text_color=TEXT_S).pack(anchor="w", padx=16, pady=(14, 2))
            ctk.CTkFrame(outer, height=1, fg_color=BORDER).pack(fill="x", padx=16, pady=(0, 10))
        return outer

    def _section_title(self, parent, text, subtitle=None):
        ctk.CTkLabel(parent, text=text,
                     font=ctk.CTkFont(size=22, weight="bold"),
                     text_color=TEXT_P).pack(anchor="w", pady=(0, 2))
        if subtitle:
            ctk.CTkLabel(parent, text=subtitle,
                         font=ctk.CTkFont(size=12), text_color=TEXT_S).pack(anchor="w", pady=(0, 16))
        else:
            ctk.CTkFrame(parent, height=6, fg_color="transparent").pack()

    # ── PAGE: DATASET ──────────────────────────────────────────────────────────
    def _build_page_dataset(self):
        page = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        self.pages["dataset"] = page
        self._section_title(page, "Dataset",
                            "MNIST — 70,000 handwritten digit images (28×28 pixels, greyscale)")

        row = ctk.CTkFrame(page, fg_color="transparent")
        row.pack(fill="x", pady=(0, 16))
        for val, lbl in [("60,000", "Training images"), ("10,000", "Test images"),
                          ("28 × 28", "Image dimensions"), ("784", "Input neurons"),
                          ("10", "Output classes (0–9)")]:
            c = self._card(row)
            c.pack(side="left", fill="x", expand=True, padx=(0, 10))
            ctk.CTkLabel(c, text=val, font=ctk.CTkFont("Courier", 20, "bold"),
                         text_color=ACCENT).pack(pady=(14, 2))
            ctk.CTkLabel(c, text=lbl, font=ctk.CTkFont(size=11),
                         text_color=TEXT_S).pack(pady=(0, 14))

        btn_card = self._card(page, title="Load MNIST Dataset")
        btn_card.pack(fill="x", pady=(0, 16))
        inner = ctk.CTkFrame(btn_card, fg_color="transparent")
        inner.pack(fill="x", padx=16, pady=(0, 16))
        ctk.CTkButton(inner, text="⬇  Download & Load MNIST",
                      fg_color=ACCENT, hover_color="#3A7DE8",
                      font=ctk.CTkFont(size=13, weight="bold"),
                      height=42, corner_radius=8,
                      command=self._load_data).pack(side="left", padx=(0, 12))
        self.data_status = ctk.CTkLabel(inner, text="Not loaded",
                                         text_color=TEXT_S, font=ctk.CTkFont(size=12))
        self.data_status.pack(side="left")

        # First-20 values display
        self.first20_card = self._card(page, title="First 20 Training Labels (Question b)")
        self.first20_card.pack(fill="x", pady=(0, 16))
        self.first20_box = ctk.CTkTextbox(self.first20_card, height=60, fg_color=BG_INPUT,
                                           font=ctk.CTkFont("Courier", 12),
                                           text_color=ACCENT, border_width=0)
        self.first20_box.pack(fill="x", padx=16, pady=(0, 16))
        self.first20_box.insert("1.0", "Load dataset to see first 20 training labels...")
        self.first20_box.configure(state="disabled")

        # Sample grid
        self.sample_card = self._card(page, title="First 20 Sample Images")
        self.sample_card.pack(fill="both", expand=True)
        self.sample_fig_frame = ctk.CTkFrame(self.sample_card, fg_color="transparent")
        self.sample_fig_frame.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        ctk.CTkLabel(self.sample_fig_frame, text="Load dataset to see samples",
                     text_color=TEXT_S).pack(pady=40)

    def _load_data(self):
        def _worker():
            self.data_status.configure(text="⟳  Loading...", text_color=WARNING)
            try:
                load_tf()
                (x_tr, y_tr), (x_te, y_te) = mnist.load_data()
                self.x_train_raw = x_tr
                self.x_test_raw  = x_te
                self.y_train_int = y_tr
                self.y_test_int  = y_te
                # Normalise: divide by 255 (pixels range 0-255 → 0.0-1.0)
                self.x_train = x_tr.reshape(60000, 784) / 255.0
                self.x_test  = x_te.reshape(10000, 784) / 255.0
                self.y_train = to_categorical(y_tr, 10)
                self.y_test  = to_categorical(y_te, 10)
                self.data_loaded = True
                self.data_status.configure(
                    text=f"✓  Loaded — shape: train={x_tr.shape}, test={x_te.shape}",
                    text_color=SUCCESS)
                # Show first 20 labels
                first20 = y_tr[:20].tolist()
                self.first20_box.configure(state="normal")
                self.first20_box.delete("1.0", "end")
                self.first20_box.insert("1.0",
                    f"First 20 training labels: {first20}\n"
                    f"Training data shape (before flatten): {x_tr.shape}  "
                    f"→ flattened to (60000, 784)")
                self.first20_box.configure(state="disabled")
                self._draw_samples()
            except Exception as e:
                self.data_status.configure(text=f"✗  Error: {e}", text_color=DANGER)
        threading.Thread(target=_worker, daemon=True).start()

    def _draw_samples(self):
        for w in self.sample_fig_frame.winfo_children():
            w.destroy()
        fig = Figure(figsize=(10, 2.8), facecolor=BG_CARD)
        for i in range(20):
            ax = fig.add_subplot(2, 10, i + 1)
            ax.imshow(self.x_train_raw[i], cmap="gray")
            ax.set_title(str(self.y_train_int[i]), fontsize=9, color=ACCENT, pad=2)
            ax.axis("off")
        fig.tight_layout(pad=0.3)
        canvas = FigureCanvasTkAgg(fig, master=self.sample_fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # ── PAGE: ARCHITECTURE ─────────────────────────────────────────────────────
    def _build_page_arch(self):
        page = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        self.pages["arch"] = page
        self._section_title(page, "Architecture",
                            "Configure the network layers. Input is always 784, output always 10.")

        config_card = self._card(page, title="Layer Configuration")
        config_card.pack(fill="x", pady=(0, 16))
        cfg = ctk.CTkFrame(config_card, fg_color="transparent")
        cfg.pack(fill="x", padx=16, pady=(0, 16))

        def _row(parent, label, default, row):
            ctk.CTkLabel(parent, text=label, text_color=TEXT_S,
                         font=ctk.CTkFont(size=12), width=180, anchor="w").grid(
                row=row, column=0, padx=(0, 12), pady=6, sticky="w")
            var = ctk.StringVar(value=default)
            ctk.CTkEntry(parent, textvariable=var, width=120,
                          fg_color=BG_INPUT, border_color=BORDER).grid(
                row=row, column=1, padx=(0, 16), pady=6, sticky="w")
            return var

        self.var_h1      = _row(cfg, "Hidden Layer 1 (neurons)", "128", 0)
        self.var_h2      = _row(cfg, "Hidden Layer 2 (neurons)", "64",  1)
        self.var_h3      = _row(cfg, "Hidden Layer 3 (optional)", "0",  2)
        self.var_act     = _row(cfg, "Hidden activation",         "relu", 3)
        self.var_dropout = _row(cfg, "Dropout rate (0 = off)",    "0.2", 4)

        ctk.CTkLabel(cfg, text="Activation options: relu · sigmoid · tanh · elu · selu",
                     font=ctk.CTkFont(size=10), text_color=TEXT_S).grid(
            row=5, column=0, columnspan=2, pady=(0, 4), sticky="w")

        diag_card = self._card(page, title="Network Diagram")
        diag_card.pack(fill="x", pady=(0, 16))
        self.diag_frame = ctk.CTkFrame(diag_card, fg_color="transparent")
        self.diag_frame.pack(fill="x", padx=16, pady=(0, 4))
        ctk.CTkButton(diag_card, text="Refresh Diagram", fg_color=BG_INPUT,
                      hover_color=BORDER, text_color=TEXT_P, height=32,
                      command=self._draw_diagram).pack(padx=16, pady=(0, 14))
        self._draw_diagram()

    def _draw_diagram(self):
        for w in self.diag_frame.winfo_children():
            w.destroy()
        try:
            h1 = int(self.var_h1.get() or 0)
            h2 = int(self.var_h2.get() or 0)
            h3 = int(self.var_h3.get() or 0)
        except:
            h1, h2, h3 = 128, 64, 0

        layers = [("Input\n784", 784, ACCENT)]
        if h1 > 0: layers.append((f"H1\n{h1}", h1, ACCENT2))
        if h2 > 0: layers.append((f"H2\n{h2}", h2, ACCENT2))
        if h3 > 0: layers.append((f"H3\n{h3}", h3, ACCENT2))
        layers.append(("Output\n10", 10, SUCCESS))

        fig = Figure(figsize=(10, 3.2), facecolor=BG_CARD)
        ax  = fig.add_subplot(111)
        ax.set_facecolor(BG_CARD)
        ax.axis("off")
        n_layers   = len(layers)
        x_positions = np.linspace(0.08, 0.92, n_layers)
        max_vis     = 12

        for i, (name, size, color) in enumerate(layers):
            n_vis = min(size, max_vis)
            y_pos = np.linspace(0.1, 0.9, n_vis)
            if i < n_layers - 1:
                n_next = min(layers[i+1][1], max_vis)
                y_next = np.linspace(0.1, 0.9, n_next)
                for y in y_pos:
                    for yn in y_next:
                        ax.plot([x_positions[i], x_positions[i+1]], [y, yn],
                                color=BORDER, lw=0.35, alpha=0.7, zorder=1)
            for y in y_pos:
                ax.add_patch(plt.Circle((x_positions[i], y), 0.024, color=color, zorder=2))
            if size > max_vis:
                ax.text(x_positions[i], 0.5, f"···\n{size}", ha="center", va="center",
                        fontsize=7, color=color, zorder=3)
            ax.text(x_positions[i], -0.04, name, ha="center", va="top",
                    fontsize=8, color=TEXT_S)

        ax.set_xlim(0, 1); ax.set_ylim(-0.14, 1.04)
        fig.tight_layout(pad=0.4)
        canvas = FigureCanvasTkAgg(fig, master=self.diag_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="x")

    # ── PAGE: TRAIN ────────────────────────────────────────────────────────────
    def _build_page_train(self):
        page = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        self.pages["train"] = page
        self._section_title(page, "Train", "Configure hyperparameters and start training.")

        top = ctk.CTkFrame(page, fg_color="transparent")
        top.pack(fill="x", pady=(0, 16))

        hp_card = self._card(top, title="Hyperparameters")
        hp_card.pack(side="left", fill="both", expand=True, padx=(0, 12))
        hp = ctk.CTkFrame(hp_card, fg_color="transparent")
        hp.pack(fill="x", padx=16, pady=(0, 16))

        def _p(parent, label, default, row):
            ctk.CTkLabel(parent, text=label, text_color=TEXT_S,
                         font=ctk.CTkFont(size=12), anchor="w").grid(
                row=row, column=0, padx=(0, 10), pady=6, sticky="w")
            var = ctk.StringVar(value=default)
            ctk.CTkEntry(parent, textvariable=var, width=110,
                          fg_color=BG_INPUT, border_color=BORDER).grid(
                row=row, column=1, pady=6, sticky="w")
            return var

        self.var_epochs   = _p(hp, "Epochs",          "15",    0)
        self.var_lr       = _p(hp, "Learning rate",   "0.001", 1)
        self.var_batch    = _p(hp, "Batch size",       "32",    2)
        self.var_valsplit = _p(hp, "Validation %",    "20",    3)

        # Optimiser + loss dropdowns
        ctk.CTkLabel(hp, text="Optimiser", text_color=TEXT_S,
                     font=ctk.CTkFont(size=12), anchor="w").grid(
            row=4, column=0, padx=(0, 10), pady=6, sticky="w")
        self.var_opt = ctk.StringVar(value="Adam")
        ctk.CTkOptionMenu(hp, variable=self.var_opt,
                          values=["Adam", "SGD", "RMSprop"],
                          fg_color=BG_INPUT, button_color=BORDER,
                          text_color=TEXT_P, width=110).grid(
            row=4, column=1, pady=6, sticky="w")

        ctk.CTkLabel(hp, text="Loss function", text_color=TEXT_S,
                     font=ctk.CTkFont(size=12), anchor="w").grid(
            row=5, column=0, padx=(0, 10), pady=6, sticky="w")
        self.var_loss = ctk.StringVar(value="categorical_crossentropy")
        ctk.CTkOptionMenu(hp, variable=self.var_loss,
                          values=["categorical_crossentropy", "mean_squared_error",
                                  "kullback_leibler_divergence"],
                          fg_color=BG_INPUT, button_color=BORDER,
                          text_color=TEXT_P, width=200).grid(
            row=5, column=1, pady=6, sticky="w")

        # Control
        ctrl_card = self._card(top, title="Training Control")
        ctrl_card.pack(side="left", fill="both", expand=True)
        ctrl = ctk.CTkFrame(ctrl_card, fg_color="transparent")
        ctrl.pack(fill="x", padx=16, pady=(0, 12))
        self.train_btn = ctk.CTkButton(
            ctrl, text="▶  Start Training",
            fg_color=ACCENT, hover_color="#3A7DE8",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=44, width=180, corner_radius=8,
            command=self._start_training)
        self.train_btn.pack(side="left", padx=(0, 12))
        self.stop_btn = ctk.CTkButton(
            ctrl, text="■  Stop",
            fg_color=DANGER, hover_color="#D45050",
            height=44, width=100, corner_radius=8,
            state="disabled", command=self._stop_training_fn)
        self.stop_btn.pack(side="left", padx=(0, 16))
        self.train_status = ctk.CTkLabel(ctrl, text="Ready", text_color=TEXT_S,
                                          font=ctk.CTkFont(size=12))
        self.train_status.pack(side="left")

        self.progress_bar = ctk.CTkProgressBar(ctrl_card, height=6,
                                                fg_color=BG_INPUT, progress_color=ACCENT)
        self.progress_bar.pack(fill="x", padx=16, pady=(0, 6))
        self.progress_bar.set(0)

        log_card = self._card(page, title="Training Log")
        log_card.pack(fill="x", pady=(0, 16))
        self.log_box = ctk.CTkTextbox(log_card, height=180, fg_color=BG_INPUT,
                                       font=ctk.CTkFont("Courier", 11),
                                       text_color=TEXT_P, border_width=0)
        self.log_box.pack(fill="x", padx=16, pady=(0, 16))

        plot_card = self._card(page, title="Live Training Curves")
        plot_card.pack(fill="both", expand=True)
        self.train_fig_frame = ctk.CTkFrame(plot_card, fg_color="transparent")
        self.train_fig_frame.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        ctk.CTkLabel(self.train_fig_frame, text="Train a model to see curves",
                     text_color=TEXT_S).pack(pady=40)

    def _log(self, msg):
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")

    def _start_training(self):
        if not self.data_loaded:
            messagebox.showerror("Error", "Load dataset first (Dataset tab).")
            return
        if self.training:
            return
        self._stop_training = False
        self.training = True
        self.train_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.log_box.delete("1.0", "end")
        threading.Thread(target=self._train_worker, daemon=True).start()

    def _stop_training_fn(self):
        self._stop_training = True
        self._log("⚠  Stop requested — finishing current epoch...")

    def _get_optimizer(self, name, lr):
        if name == "Adam":    return Adam(learning_rate=lr)
        if name == "SGD":     return SGD(learning_rate=lr, momentum=0.9)
        if name == "RMSprop": return RMSprop(learning_rate=lr)
        return Adam(learning_rate=lr)

    def _build_model(self, h1, h2, h3, act, dropout):
        layers = [Dense(h1, activation=act, input_shape=(784,))]
        if dropout > 0: layers.append(Dropout(dropout))
        if h2 > 0:
            layers.append(Dense(h2, activation=act))
            if dropout > 0: layers.append(Dropout(dropout))
        if h3 > 0:
            layers.append(Dense(h3, activation=act))
        layers.append(Dense(10, activation="softmax"))
        return Sequential(layers)

    def _train_worker(self):
        try:
            load_tf()
            epochs  = int(self.var_epochs.get())
            lr      = float(self.var_lr.get())
            batch   = int(self.var_batch.get())
            val_pct = float(self.var_valsplit.get()) / 100
            h1      = int(self.var_h1.get() or 128)
            h2      = int(self.var_h2.get() or 64)
            h3      = int(self.var_h3.get() or 0)
            act     = self.var_act.get() or "relu"
            dropout = float(self.var_dropout.get() or 0)
            opt_name = self.var_opt.get()
            loss_fn  = self.var_loss.get()

            self._log(f"══════════════════════════════════════")
            self._log(f"  Model: Input(784) → {h1} → {h2}" +
                      (f" → {h3}" if h3 else "") + " → Output(10)")
            self._log(f"  Activation : {act}  |  Dropout: {dropout}")
            self._log(f"  Optimiser  : {opt_name}  |  LR: {lr}")
            self._log(f"  Loss       : {loss_fn}")
            self._log(f"  Epochs: {epochs}  |  Batch: {batch}")
            self._log(f"══════════════════════════════════════\n")

            self.model = self._build_model(h1, h2, h3, act, dropout)
            self.model.compile(
                optimizer=self._get_optimizer(opt_name, lr),
                loss=loss_fn,
                metrics=["accuracy"])

            self._log(f"Total parameters: {self.model.count_params():,}\n")

            acc_hist, val_hist, loss_hist, val_loss_hist = [], [], [], []

            class GUICallback(tf.keras.callbacks.Callback):
                def __init__(cb): super().__init__()
                def on_epoch_end(cb, epoch, logs=None):
                    if self._stop_training:
                        cb.model.stop_training = True
                    a  = logs.get("accuracy", 0)
                    va = logs.get("val_accuracy", 0)
                    l  = logs.get("loss", 0)
                    vl = logs.get("val_loss", 0)
                    acc_hist.append(a); val_hist.append(va)
                    loss_hist.append(l); val_loss_hist.append(vl)
                    self._log(f"  Epoch {epoch+1:2d}/{epochs}  "
                              f"acc={a:.4f}  val_acc={va:.4f}  "
                              f"loss={l:.4f}  val_loss={vl:.4f}")
                    self.progress_bar.set((epoch + 1) / epochs)
                    self._update_train_plot(acc_hist, val_hist, loss_hist, val_loss_hist)

            self.model.fit(self.x_train, self.y_train,
                           epochs=epochs, batch_size=batch,
                           validation_split=val_pct,
                           callbacks=[GUICallback()], verbose=0)

            _, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
            self._log(f"\n✓  Training complete!  Test accuracy: {acc*100:.2f}%")
            target = "✓ Target met!" if acc >= 0.97 else ("⚠ Above 90%" if acc >= 0.90 else "✗ Below 90%")
            self._log(f"   97% benchmark: {target}")
            self.status_dot.configure(
                text=f"● {acc*100:.1f}% accuracy",
                text_color=SUCCESS if acc >= 0.97 else (WARNING if acc >= 0.90 else DANGER))

        except Exception as e:
            self._log(f"✗  Error: {e}")
        finally:
            self.training = False
            self.train_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")

    def _update_train_plot(self, acc, val_acc, loss, val_loss):
        for w in self.train_fig_frame.winfo_children():
            w.destroy()
        fig = Figure(figsize=(10, 3.2), facecolor=BG_CARD)
        epochs = list(range(1, len(acc) + 1))

        ax1 = fig.add_subplot(121)
        ax1.set_facecolor(BG_CARD)
        ax1.plot(epochs, acc,     color=ACCENT,  label="Train", lw=2)
        ax1.plot(epochs, val_acc, color=ACCENT2, label="Val",   lw=2, linestyle="--")
        ax1.axhline(0.97, color=SUCCESS, lw=1, linestyle=":", alpha=0.7, label="97% target")
        ax1.set_title("Accuracy", color=TEXT_S, fontsize=11)
        ax1.set_xlabel("Epoch", color=TEXT_S, fontsize=9)
        ax1.tick_params(colors=TEXT_S, labelsize=8)
        ax1.legend(fontsize=8, labelcolor=TEXT_S, facecolor=BG_INPUT, edgecolor=BORDER)
        for sp in ax1.spines.values(): sp.set_color(BORDER)
        ax1.grid(color=BORDER, linestyle="--", lw=0.5)

        ax2 = fig.add_subplot(122)
        ax2.set_facecolor(BG_CARD)
        ax2.plot(epochs, loss,     color=DANGER,  label="Train", lw=2)
        ax2.plot(epochs, val_loss, color=WARNING, label="Val",   lw=2, linestyle="--")
        ax2.set_title("Loss", color=TEXT_S, fontsize=11)
        ax2.set_xlabel("Epoch", color=TEXT_S, fontsize=9)
        ax2.tick_params(colors=TEXT_S, labelsize=8)
        ax2.legend(fontsize=8, labelcolor=TEXT_S, facecolor=BG_INPUT, edgecolor=BORDER)
        for sp in ax2.spines.values(): sp.set_color(BORDER)
        ax2.grid(color=BORDER, linestyle="--", lw=0.5)

        fig.patch.set_facecolor(BG_CARD)
        fig.tight_layout(pad=1.5)
        canvas = FigureCanvasTkAgg(fig, master=self.train_fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # ── PAGE: EXPERIMENTS ──────────────────────────────────────────────────────
    def _build_page_experiments(self):
        page = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        self.pages["experiments"] = page
        self._section_title(page, "Experiments",
                            "Systematically vary one parameter at a time — satisfies Question f.")

        info = self._card(page, title="How it works")
        info.pack(fill="x", pady=(0, 16))
        ctk.CTkLabel(info,
                     text="Each experiment trains a fresh model, varying ONE parameter while keeping all others constant.\n"
                          "Results are plotted as a bar chart so you can clearly see the effect on accuracy.",
                     font=ctk.CTkFont(size=12), text_color=TEXT_S,
                     wraplength=900, justify="left").pack(padx=16, pady=(0, 14))

        # Which experiment to run
        sel_card = self._card(page, title="Select Experiment")
        sel_card.pack(fill="x", pady=(0, 16))
        sel = ctk.CTkFrame(sel_card, fg_color="transparent")
        sel.pack(fill="x", padx=16, pady=(0, 16))

        self.exp_choice = ctk.StringVar(value="Epochs")
        for i, name in enumerate(["Epochs", "Learning Rate", "Activation Function",
                                   "Optimiser", "Loss Function", "Number of Layers"]):
            ctk.CTkRadioButton(sel, text=name, variable=self.exp_choice, value=name,
                               text_color=TEXT_P, font=ctk.CTkFont(size=12)).grid(
                row=i // 3, column=i % 3, padx=16, pady=6, sticky="w")

        run_row = ctk.CTkFrame(page, fg_color="transparent")
        run_row.pack(fill="x", pady=(0, 16))
        self.exp_run_btn = ctk.CTkButton(
            run_row, text="▶  Run Experiment",
            fg_color=ACCENT2, hover_color="#8B6FE8",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=44, width=200, corner_radius=8,
            command=self._run_experiment)
        self.exp_run_btn.pack(side="left", padx=(0, 12))
        self.exp_status = ctk.CTkLabel(run_row, text="Select an experiment and click Run",
                                        text_color=TEXT_S, font=ctk.CTkFont(size=12))
        self.exp_status.pack(side="left")

        self.exp_progress = ctk.CTkProgressBar(page, height=6,
                                                fg_color=BG_INPUT, progress_color=ACCENT2)
        self.exp_progress.pack(fill="x", pady=(0, 16))
        self.exp_progress.set(0)

        self.exp_log = ctk.CTkTextbox(page, height=160, fg_color=BG_INPUT,
                                       font=ctk.CTkFont("Courier", 11),
                                       text_color=TEXT_P, border_width=0)
        self.exp_log.pack(fill="x", pady=(0, 16))

        self.exp_fig_frame = ctk.CTkFrame(page, fg_color="transparent")
        self.exp_fig_frame.pack(fill="both", expand=True)
        ctk.CTkLabel(self.exp_fig_frame, text="Results will appear here",
                     text_color=TEXT_S).pack(pady=40)

    def _exp_log(self, msg):
        self.exp_log.insert("end", msg + "\n")
        self.exp_log.see("end")

    def _run_experiment(self):
        if not self.data_loaded:
            messagebox.showerror("Error", "Load dataset first.")
            return
        choice = self.exp_choice.get()
        self.exp_run_btn.configure(state="disabled")
        self.exp_log.delete("1.0", "end")
        self.exp_progress.set(0)
        threading.Thread(target=self._experiment_worker,
                         args=(choice,), daemon=True).start()

    def _experiment_worker(self, choice):
        try:
            load_tf()
            self.exp_status.configure(text=f"⟳  Running: {choice}...", text_color=WARNING)
            self._exp_log(f"═══ Experiment: {choice} ═══\n")

            # Base config
            BASE = dict(h1=128, h2=64, h3=0, act="relu", dropout=0.2,
                        lr=0.001, opt="Adam", loss="categorical_crossentropy",
                        epochs=10, batch=32)

            if choice == "Epochs":
                variants = [3, 5, 10, 15, 20]
                label = "Epochs"
                configs = [{**BASE, "epochs": v} for v in variants]
                x_labels = [str(v) for v in variants]

            elif choice == "Learning Rate":
                variants = [0.1, 0.01, 0.001, 0.0001, 0.00001]
                label = "Learning Rate"
                configs = [{**BASE, "lr": v} for v in variants]
                x_labels = [str(v) for v in variants]

            elif choice == "Activation Function":
                variants = ["relu", "sigmoid", "tanh", "elu", "selu"]
                label = "Activation"
                configs = [{**BASE, "act": v} for v in variants]
                x_labels = variants

            elif choice == "Optimiser":
                variants = ["Adam", "SGD", "RMSprop"]
                label = "Optimiser"
                configs = [{**BASE, "opt": v} for v in variants]
                x_labels = variants

            elif choice == "Loss Function":
                variants = ["categorical_crossentropy",
                            "mean_squared_error",
                            "kullback_leibler_divergence"]
                label = "Loss"
                configs = [{**BASE, "loss": v} for v in variants]
                x_labels = ["crossentropy", "MSE", "KLD"]

            elif choice == "Number of Layers":
                variants = [(128, 0, 0), (128, 64, 0), (128, 64, 32)]
                label = "Layers"
                configs = [{**BASE, "h1": h1, "h2": h2, "h3": h3}
                           for h1, h2, h3 in variants]
                x_labels = ["1 hidden", "2 hidden", "3 hidden"]

            results = []
            n = len(configs)
            for i, cfg in enumerate(configs):
                self._exp_log(f"  [{i+1}/{n}] {label}={x_labels[i]}  →  training...")
                m = self._build_model(cfg["h1"], cfg["h2"], cfg["h3"],
                                      cfg["act"], cfg["dropout"])
                m.compile(optimizer=self._get_optimizer(cfg["opt"], cfg["lr"]),
                           loss=cfg["loss"], metrics=["accuracy"])
                m.fit(self.x_train, self.y_train,
                      epochs=cfg["epochs"], batch_size=cfg["batch"],
                      validation_split=0.2, verbose=0)
                _, acc = m.evaluate(self.x_test, self.y_test, verbose=0)
                results.append(acc)
                self._exp_log(f"         accuracy = {acc*100:.2f}%")
                self.exp_progress.set((i + 1) / n)

            self._plot_experiment(x_labels, results, label, choice)
            best = x_labels[np.argmax(results)]
            self._exp_log(f"\n✓  Best {label}: {best}  ({max(results)*100:.2f}%)")
            self.exp_status.configure(text=f"✓  Done — best {label}: {best}", text_color=SUCCESS)

        except Exception as e:
            self.exp_status.configure(text=f"✗  {e}", text_color=DANGER)
            self._exp_log(f"Error: {e}")
        finally:
            self.exp_run_btn.configure(state="normal")

    def _plot_experiment(self, labels, results, x_label, title):
        for w in self.exp_fig_frame.winfo_children():
            w.destroy()
        fig = Figure(figsize=(10, 3.8), facecolor=BG_CARD)
        ax  = fig.add_subplot(111)
        ax.set_facecolor(BG_CARD)

        colors = [SUCCESS if r == max(results) else ACCENT for r in results]
        bars = ax.bar(labels, [r * 100 for r in results], color=colors,
                      edgecolor=BORDER, linewidth=0.8, width=0.5)
        ax.axhline(97, color=SUCCESS, lw=1.2, linestyle="--", alpha=0.8, label="97% target")
        ax.axhline(90, color=WARNING, lw=1.0, linestyle=":",  alpha=0.7, label="90% target")

        for bar, r in zip(bars, results):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{r*100:.1f}%", ha="center", va="bottom", fontsize=9, color=TEXT_P)

        ax.set_title(f"Effect of {title} on Test Accuracy", color=TEXT_P, fontsize=12, pad=10)
        ax.set_xlabel(x_label, color=TEXT_S, fontsize=10)
        ax.set_ylabel("Accuracy (%)", color=TEXT_S, fontsize=10)
        ax.tick_params(colors=TEXT_S, labelsize=9)
        ax.legend(fontsize=9, labelcolor=TEXT_S, facecolor=BG_INPUT, edgecolor=BORDER)
        ax.set_ylim(max(0, min(results) * 100 - 10), 100)
        for sp in ax.spines.values(): sp.set_color(BORDER)
        ax.grid(axis="y", color=BORDER, linestyle="--", lw=0.5)

        fig.patch.set_facecolor(BG_CARD)
        fig.tight_layout(pad=1.5)
        canvas = FigureCanvasTkAgg(fig, master=self.exp_fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # ── PAGE: EVALUATE ─────────────────────────────────────────────────────────
    def _build_page_evaluate(self):
        page = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        self.pages["evaluate"] = page
        self._section_title(page, "Evaluate", "Test the trained model on the MNIST test set.")

        btn_row = ctk.CTkFrame(page, fg_color="transparent")
        btn_row.pack(fill="x", pady=(0, 16))
        ctk.CTkButton(btn_row, text="⬡  Run Evaluation",
                      fg_color=ACCENT, hover_color="#3A7DE8",
                      font=ctk.CTkFont(size=13, weight="bold"),
                      height=44, width=200, corner_radius=8,
                      command=self._run_evaluate).pack(side="left", padx=(0, 12))
        self.eval_status = ctk.CTkLabel(btn_row, text="Train a model first",
                                         text_color=TEXT_S, font=ctk.CTkFont(size=12))
        self.eval_status.pack(side="left")

        self.metrics_row = ctk.CTkFrame(page, fg_color="transparent")
        self.metrics_row.pack(fill="x", pady=(0, 16))

        self.cm_card = self._card(page, title="Confusion Matrix")
        self.cm_card.pack(fill="both", expand=True, pady=(0, 16))
        self.cm_frame = ctk.CTkFrame(self.cm_card, fg_color="transparent")
        self.cm_frame.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        ctk.CTkLabel(self.cm_frame, text="Run evaluation to see confusion matrix",
                     text_color=TEXT_S).pack(pady=40)

        self.report_card = self._card(page, title="Classification Report (Precision / Recall / F1)")
        self.report_card.pack(fill="x", pady=(0, 16))
        self.report_box = ctk.CTkTextbox(self.report_card, height=220, fg_color=BG_INPUT,
                                          font=ctk.CTkFont("Courier", 11),
                                          text_color=TEXT_P, border_width=0)
        self.report_box.pack(fill="x", padx=16, pady=(0, 16))

    def _run_evaluate(self):
        if self.model is None:
            messagebox.showerror("Error", "Train a model first.")
            return
        def _worker():
            self.eval_status.configure(text="⟳  Evaluating...", text_color=WARNING)
            try:
                import seaborn as sns
                loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)

                for w in self.metrics_row.winfo_children():
                    w.destroy()
                for val, lbl, col in [
                    (f"{acc*100:.2f}%", "Test Accuracy",  SUCCESS if acc > 0.97 else (WARNING if acc > 0.9 else DANGER)),
                    (f"{loss:.4f}",     "Test Loss",       TEXT_P),
                    ("10,000",          "Test Samples",    TEXT_P),
                    ("✓" if acc >= 0.97 else ("~" if acc >= 0.90 else "✗"),
                     ">97% Benchmark",  SUCCESS if acc >= 0.97 else (WARNING if acc >= 0.90 else DANGER))]:
                    c = self._card(self.metrics_row)
                    c.pack(side="left", fill="x", expand=True, padx=(0, 10))
                    ctk.CTkLabel(c, text=val, font=ctk.CTkFont("Courier", 22, "bold"),
                                 text_color=col).pack(pady=(12, 2))
                    ctk.CTkLabel(c, text=lbl, font=ctk.CTkFont(size=11),
                                 text_color=TEXT_S).pack(pady=(0, 12))

                preds = np.argmax(self.model.predict(self.x_test, verbose=0), axis=1)
                true  = np.argmax(self.y_test, axis=1)
                cm    = confusion_matrix_fn(true, preds)

                for w in self.cm_frame.winfo_children():
                    w.destroy()
                fig = Figure(figsize=(7.5, 5.8), facecolor=BG_CARD)
                ax  = fig.add_subplot(111)
                ax.set_facecolor(BG_CARD)
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=range(10), yticklabels=range(10),
                            ax=ax, cbar=True, linewidths=0.3)
                ax.set_title("Confusion Matrix — rows=Actual, cols=Predicted",
                             color=TEXT_S, fontsize=11, pad=10)
                ax.set_xlabel("Predicted", color=TEXT_S)
                ax.set_ylabel("Actual", color=TEXT_S)
                ax.tick_params(colors=TEXT_S)
                fig.patch.set_facecolor(BG_CARD)
                fig.tight_layout()
                canvas = FigureCanvasTkAgg(fig, master=self.cm_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True)

                report = classification_report_fn(true, preds,
                                                   target_names=[str(i) for i in range(10)])
                self.report_box.delete("1.0", "end")
                self.report_box.insert("1.0", report)
                self.eval_status.configure(
                    text=f"✓  Done — {acc*100:.2f}% accuracy", text_color=SUCCESS)
            except Exception as e:
                self.eval_status.configure(text=f"✗  {e}", text_color=DANGER)
        threading.Thread(target=_worker, daemon=True).start()

    # ── PAGE: PREDICT ──────────────────────────────────────────────────────────
    def _build_page_predict(self):
        page = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        self.pages["predict"] = page
        self._section_title(page, "Predict",
                            "Draw a digit or upload a hand-drawn image file (Part 2 of assignment).")

        top = ctk.CTkFrame(page, fg_color="transparent")
        top.pack(fill="x", pady=(0, 16))

        draw_card = self._card(top, title="Draw a digit")
        draw_card.pack(side="left", fill="both", expand=True, padx=(0, 12))
        self.canvas_size = 280
        self.draw_canvas = tk.Canvas(draw_card, width=self.canvas_size,
                                      height=self.canvas_size, bg="black",
                                      highlightthickness=1, highlightbackground=BORDER)
        self.draw_canvas.pack(padx=16, pady=8)
        self.draw_canvas.bind("<B1-Motion>",      self._paint)
        self.draw_canvas.bind("<ButtonRelease-1>", self._reset_prev)

        btn_row = ctk.CTkFrame(draw_card, fg_color="transparent")
        btn_row.pack(fill="x", padx=16, pady=(0, 12))
        ctk.CTkButton(btn_row, text="Clear", fg_color=BG_INPUT, hover_color=BORDER,
                      text_color=TEXT_P, width=80, height=32,
                      command=lambda: self.draw_canvas.delete("all")).pack(side="left", padx=(0, 8))
        ctk.CTkButton(btn_row, text="⬡  Predict",
                      fg_color=ACCENT, hover_color="#3A7DE8",
                      text_color=TEXT_P, width=100, height=32,
                      command=self._predict_drawing).pack(side="left")

        result_card = self._card(top, title="Prediction Result")
        result_card.pack(side="left", fill="both", expand=True)
        self.pred_digit = ctk.CTkLabel(result_card, text="?",
                                        font=ctk.CTkFont("Courier", 80, "bold"),
                                        text_color=ACCENT)
        self.pred_digit.pack(pady=(20, 4))
        self.pred_conf = ctk.CTkLabel(result_card, text="Draw and click Predict",
                                       font=ctk.CTkFont(size=12), text_color=TEXT_S)
        self.pred_conf.pack()
        self.bar_frame = ctk.CTkFrame(result_card, fg_color="transparent")
        self.bar_frame.pack(fill="x", padx=16, pady=16)

        # Upload from file (Part 2)
        file_card = self._card(page, title="📂  Predict from Image File  (Part 2 — own data)")
        file_card.pack(fill="x", pady=(0, 16))
        file_inner = ctk.CTkFrame(file_card, fg_color="transparent")
        file_inner.pack(fill="x", padx=16, pady=(0, 16))
        ctk.CTkButton(file_inner, text="Browse Image(s)",
                      fg_color=BG_INPUT, hover_color=BORDER, text_color=TEXT_P,
                      height=40, width=160, command=self._predict_file).pack(side="left", padx=(0, 12))
        ctk.CTkButton(file_inner, text="Batch Predict Folder",
                      fg_color=BG_INPUT, hover_color=BORDER, text_color=TEXT_P,
                      height=40, width=180, command=self._batch_predict).pack(side="left", padx=(0, 12))
        self.file_label = ctk.CTkLabel(file_inner, text="No file selected",
                                        text_color=TEXT_S, font=ctk.CTkFont(size=12))
        self.file_label.pack(side="left")

        # Batch results table
        self.batch_card = self._card(page, title="Batch Prediction Results")
        self.batch_card.pack(fill="x", pady=(0, 16))
        self.batch_box = ctk.CTkTextbox(self.batch_card, height=200, fg_color=BG_INPUT,
                                         font=ctk.CTkFont("Courier", 11),
                                         text_color=TEXT_P, border_width=0)
        self.batch_box.pack(fill="x", padx=16, pady=(0, 16))
        self.batch_box.insert("1.0", "Batch predict a folder of digit images here.\n"
                               "Images should be JPEG/PNG with a black background and white digit.")

        self._prev_x = self._prev_y = None

    def _paint(self, event):
        r = 14
        x, y = event.x, event.y
        if self._prev_x and self._prev_y:
            self.draw_canvas.create_line(self._prev_x, self._prev_y, x, y,
                                          fill="white", width=r * 2,
                                          capstyle=tk.ROUND, smooth=True)
        self.draw_canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
        self._prev_x, self._prev_y = x, y

    def _reset_prev(self, event):
        self._prev_x = self._prev_y = None

    def _preprocess_image(self, img):
        """Convert a PIL image to a normalised (1, 784) numpy array."""
        from PIL import Image
        img = img.convert("L").resize((28, 28), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr.reshape(1, 784)

    def _predict_drawing(self):
        if self.model is None:
            messagebox.showerror("Error", "Train a model first.")
            return
        ps = self.draw_canvas.postscript(colormode="gray")
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(ps.encode("utf-8")))
        except:
            try:
                import PIL.ImageGrab
                x0 = self.draw_canvas.winfo_rootx()
                y0 = self.draw_canvas.winfo_rooty()
                img = PIL.ImageGrab.grab(bbox=(x0, y0,
                                               x0 + self.canvas_size,
                                               y0 + self.canvas_size)).convert("L")
            except:
                messagebox.showerror("Error", "Could not capture drawing. Try uploading a file.")
                return
        self._show_prediction(self._preprocess_image(img))

    def _predict_file(self):
        if self.model is None:
            messagebox.showerror("Error", "Train a model first.")
            return
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        try:
            from PIL import Image
            img = Image.open(path)
            arr = self._preprocess_image(img)
            self.file_label.configure(text=os.path.basename(path))
            self._show_prediction(arr)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _batch_predict(self):
        if self.model is None:
            messagebox.showerror("Error", "Train a model first.")
            return
        folder = filedialog.askdirectory(title="Select folder with digit images")
        if not folder:
            return
        try:
            from PIL import Image
            exts = {".jpg", ".jpeg", ".png", ".bmp"}
            files = [f for f in sorted(os.listdir(folder))
                     if os.path.splitext(f)[1].lower() in exts]
            if not files:
                messagebox.showinfo("No images", "No image files found in that folder.")
                return
            self.batch_box.delete("1.0", "end")
            self.batch_box.insert("end",
                f"Batch predicting {len(files)} images from: {folder}\n"
                f"{'─'*60}\n")
            correct = 0
            for fname in files:
                path = os.path.join(folder, fname)
                img  = Image.open(path)
                arr  = self._preprocess_image(img)
                probs  = self.model.predict(arr, verbose=0)[0]
                pred   = np.argmax(probs)
                conf   = probs[pred] * 100
                self.batch_box.insert("end",
                    f"  {fname:<30}  →  Predicted: {pred}  (conf: {conf:.1f}%)\n")
            self.batch_box.insert("end",
                f"{'─'*60}\nTotal images processed: {len(files)}\n")
            self.batch_box.see("end")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _show_prediction(self, arr):
        probs = self.model.predict(arr, verbose=0)[0]
        digit = np.argmax(probs)
        conf  = probs[digit] * 100
        self.pred_digit.configure(text=str(digit))
        self.pred_conf.configure(
            text=f"Confidence: {conf:.1f}%",
            text_color=SUCCESS if conf > 70 else WARNING)
        for w in self.bar_frame.winfo_children():
            w.destroy()
        for i, p in enumerate(probs):
            row = ctk.CTkFrame(self.bar_frame, fg_color="transparent")
            row.pack(fill="x", pady=1)
            ctk.CTkLabel(row, text=str(i), width=20,
                         font=ctk.CTkFont("Courier", 11),
                         text_color=ACCENT if i == digit else TEXT_S).pack(side="left")
            bar = ctk.CTkProgressBar(row, height=10, fg_color=BG_INPUT,
                                      progress_color=ACCENT if i == digit else BORDER)
            bar.pack(side="left", fill="x", expand=True, padx=6)
            bar.set(float(p))
            ctk.CTkLabel(row, text=f"{p*100:.1f}%", width=50,
                         font=ctk.CTkFont(size=10),
                         text_color=TEXT_P if i == digit else TEXT_S).pack(side="left")

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = ANNApp()
    app.mainloop()
