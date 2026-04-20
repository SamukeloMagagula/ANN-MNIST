"""
ANN MNIST GUI — Full-featured desktop application
Requires: pip install tensorflow customtkinter pillow scikit-learn seaborn matplotlib numpy
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

# ── Lazy imports (heavy, loaded on demand) ────────────────────────────────────
tf = None
mnist = None
Sequential = None
Dense = None
to_categorical = None
Adam = None
confusion_matrix_fn = None
classification_report_fn = None

def load_tf():
    global tf, mnist, Sequential, Dense, to_categorical, Adam
    global confusion_matrix_fn, classification_report_fn
    if tf is None:
        import tensorflow as _tf
        tf = _tf
        from tensorflow.keras.datasets import mnist as _mnist
        from tensorflow.keras.models import Sequential as _Seq
        from tensorflow.keras.layers import Dense as _Dense
        from tensorflow.keras.utils import to_categorical as _tc
        from tensorflow.keras.optimizers import Adam as _Adam
        from sklearn.metrics import confusion_matrix as _cm, classification_report as _cr
        mnist = _mnist
        Sequential = _Seq
        Dense = _Dense
        to_categorical = _tc
        Adam = _Adam
        confusion_matrix_fn = _cm
        classification_report_fn = _cr

# ── Theme ─────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

ACCENT   = "#4F8EF7"
ACCENT2  = "#A78BFA"
BG_DARK  = "#0F1117"
BG_CARD  = "#1A1D27"
BG_INPUT = "#242736"
TEXT_P   = "#E8EAF0"
TEXT_S   = "#8B91A8"
SUCCESS  = "#34D399"
WARNING  = "#FBBF24"
DANGER   = "#F87171"
BORDER   = "#2E3245"

# ── Main App ──────────────────────────────────────────────────────────────────
class ANNApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("ANN MNIST — Neural Network Studio")
        self.geometry("1280x820")
        self.minsize(1100, 700)
        self.configure(fg_color=BG_DARK)

        # State
        self.model = None
        self.history = None
        self.x_train = self.y_train = self.x_test = self.y_test = None
        self.x_train_raw = self.x_test_raw = None
        self.y_train_int = self.y_test_int = None
        self.data_loaded = False
        self.training = False
        self._stop_training = False

        self._build_ui()

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=200, fg_color=BG_CARD,
                                     corner_radius=0, border_width=0)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # Logo
        logo = ctk.CTkLabel(self.sidebar, text="ANN\nSTUDIO",
                             font=ctk.CTkFont("Courier", 22, "bold"),
                             text_color=ACCENT)
        logo.pack(pady=(28, 4))
        ctk.CTkLabel(self.sidebar, text="MNIST Neural Network",
                     font=ctk.CTkFont(size=10), text_color=TEXT_S).pack(pady=(0, 24))

        ctk.CTkFrame(self.sidebar, height=1, fg_color=BORDER).pack(fill="x", padx=16, pady=4)

        # Nav buttons
        self.nav_buttons = {}
        pages = [
            ("⬡  Dataset",   "dataset"),
            ("⬡  Architecture", "arch"),
            ("⬡  Train",     "train"),
            ("⬡  Evaluate",  "evaluate"),
            ("⬡  Predict",   "predict"),
        ]
        for label, key in pages:
            btn = ctk.CTkButton(
                self.sidebar, text=label, anchor="w",
                fg_color="transparent", hover_color=BG_INPUT,
                text_color=TEXT_S, font=ctk.CTkFont(size=13),
                height=40, corner_radius=8,
                command=lambda k=key: self._show_page(k)
            )
            btn.pack(fill="x", padx=12, pady=2)
            self.nav_buttons[key] = btn

        # Status indicator at bottom of sidebar
        ctk.CTkFrame(self.sidebar, height=1, fg_color=BORDER).pack(fill="x", padx=16, pady=(12,4), side="bottom")
        self.status_dot = ctk.CTkLabel(self.sidebar, text="● No model loaded",
                                        font=ctk.CTkFont(size=11), text_color=TEXT_S)
        self.status_dot.pack(pady=8, side="bottom")

        # Main content
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.pack(side="left", fill="both", expand=True, padx=0)

        # Pages
        self.pages = {}
        self._build_page_dataset()
        self._build_page_arch()
        self._build_page_train()
        self._build_page_evaluate()
        self._build_page_predict()

        self._show_page("dataset")

    def _show_page(self, key):
        for k, frame in self.pages.items():
            frame.pack_forget()
        self.pages[key].pack(fill="both", expand=True, padx=24, pady=24)
        for k, btn in self.nav_buttons.items():
            btn.configure(
                fg_color=BG_INPUT if k == key else "transparent",
                text_color=TEXT_P if k == key else TEXT_S
            )

    def _card(self, parent, title=None, **kwargs):
        outer = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=12,
                              border_width=1, border_color=BORDER, **kwargs)
        if title:
            ctk.CTkLabel(outer, text=title,
                         font=ctk.CTkFont(size=12, weight="bold"),
                         text_color=TEXT_S).pack(anchor="w", padx=16, pady=(14,2))
            ctk.CTkFrame(outer, height=1, fg_color=BORDER).pack(fill="x", padx=16, pady=(0,10))
        return outer

    def _section_title(self, parent, text):
        ctk.CTkLabel(parent, text=text,
                     font=ctk.CTkFont(size=20, weight="bold"),
                     text_color=TEXT_P).pack(anchor="w", pady=(0, 16))

    # ── PAGE: DATASET ─────────────────────────────────────────────────────────

    def _build_page_dataset(self):
        page = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        self.pages["dataset"] = page

        self._section_title(page, "Dataset")

        # Info cards row
        row = ctk.CTkFrame(page, fg_color="transparent")
        row.pack(fill="x", pady=(0, 16))
        stats = [("60,000", "Training images"), ("10,000", "Test images"),
                 ("28 × 28", "Image size"), ("10", "Classes (0–9)")]
        for val, lbl in stats:
            c = self._card(row)
            c.pack(side="left", fill="x", expand=True, padx=(0,10))
            ctk.CTkLabel(c, text=val, font=ctk.CTkFont("Courier", 22, "bold"),
                         text_color=ACCENT).pack(pady=(14,2))
            ctk.CTkLabel(c, text=lbl, font=ctk.CTkFont(size=11),
                         text_color=TEXT_S).pack(pady=(0,14))

        # Load button + output
        btn_card = self._card(page, title="Load MNIST Dataset")
        btn_card.pack(fill="x", pady=(0,16))
        inner = ctk.CTkFrame(btn_card, fg_color="transparent")
        inner.pack(fill="x", padx=16, pady=(0,16))

        ctk.CTkButton(inner, text="⬇  Download & Load MNIST",
                      fg_color=ACCENT, hover_color="#3A7DE8",
                      font=ctk.CTkFont(size=13, weight="bold"),
                      height=40, corner_radius=8,
                      command=self._load_data).pack(side="left", padx=(0,12))

        self.data_status = ctk.CTkLabel(inner, text="Not loaded",
                                         text_color=TEXT_S, font=ctk.CTkFont(size=12))
        self.data_status.pack(side="left")

        # Sample grid
        self.sample_card = self._card(page, title="Sample Images")
        self.sample_card.pack(fill="both", expand=True)
        self.sample_fig_frame = ctk.CTkFrame(self.sample_card, fg_color="transparent")
        self.sample_fig_frame.pack(fill="both", expand=True, padx=16, pady=(0,16))
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
                # Preprocess
                self.x_train = x_tr.reshape(60000, 784) / 255.0
                self.x_test  = x_te.reshape(10000, 784)  / 255.0
                self.y_train = to_categorical(y_tr, 10)
                self.y_test  = to_categorical(y_te, 10)
                self.data_loaded = True
                self.data_status.configure(text="✓  Loaded — 70,000 images ready", text_color=SUCCESS)
                self._draw_samples()
            except Exception as e:
                self.data_status.configure(text=f"✗  Error: {e}", text_color=DANGER)
        threading.Thread(target=_worker, daemon=True).start()

    def _draw_samples(self):
        for w in self.sample_fig_frame.winfo_children():
            w.destroy()
        fig = Figure(figsize=(9, 2.8), facecolor=BG_CARD)
        for i in range(20):
            ax = fig.add_subplot(2, 10, i+1)
            ax.imshow(self.x_train_raw[i], cmap="gray")
            ax.set_title(str(self.y_train_int[i]), fontsize=8, color=TEXT_S, pad=2)
            ax.axis("off")
        fig.tight_layout(pad=0.3)
        canvas = FigureCanvasTkAgg(fig, master=self.sample_fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # ── PAGE: ARCHITECTURE ────────────────────────────────────────────────────

    def _build_page_arch(self):
        page = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        self.pages["arch"] = page
        self._section_title(page, "Architecture")

        # Layer config
        config_card = self._card(page, title="Layer Configuration")
        config_card.pack(fill="x", pady=(0,16))
        cfg = ctk.CTkFrame(config_card, fg_color="transparent")
        cfg.pack(fill="x", padx=16, pady=(0,16))

        def _row(parent, label, default, row):
            ctk.CTkLabel(parent, text=label, text_color=TEXT_S,
                         font=ctk.CTkFont(size=12), width=160, anchor="w").grid(
                row=row, column=0, padx=(0,12), pady=6, sticky="w")
            var = ctk.StringVar(value=default)
            entry = ctk.CTkEntry(parent, textvariable=var, width=120,
                                  fg_color=BG_INPUT, border_color=BORDER)
            entry.grid(row=row, column=1, padx=(0,16), pady=6, sticky="w")
            return var

        self.var_h1     = _row(cfg, "Hidden Layer 1 (neurons)", "128", 0)
        self.var_h2     = _row(cfg, "Hidden Layer 2 (neurons)", "64",  1)
        self.var_h3     = _row(cfg, "Hidden Layer 3 (optional)", "0",  2)
        self.var_act    = _row(cfg, "Hidden activation",         "relu", 3)
        self.var_dropout= _row(cfg, "Dropout rate (0 = off)",    "0.0", 4)

        # Visual diagram
        diag_card = self._card(page, title="Network Diagram")
        diag_card.pack(fill="x", pady=(0,16))
        self.diag_frame = ctk.CTkFrame(diag_card, fg_color="transparent")
        self.diag_frame.pack(fill="x", padx=16, pady=(0,16))
        ctk.CTkButton(diag_card, text="Refresh Diagram", fg_color=BG_INPUT,
                      hover_color=BORDER, text_color=TEXT_P, height=32,
                      command=self._draw_diagram).pack(padx=16, pady=(0,14))
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
        if h1 > 0: layers.append((f"Hidden 1\n{h1}", h1, ACCENT2))
        if h2 > 0: layers.append((f"Hidden 2\n{h2}", h2, ACCENT2))
        if h3 > 0: layers.append((f"Hidden 3\n{h3}", h3, ACCENT2))
        layers.append(("Output\n10", 10, SUCCESS))

        fig = Figure(figsize=(9, 3), facecolor=BG_CARD)
        ax  = fig.add_subplot(111)
        ax.set_facecolor(BG_CARD)
        ax.axis("off")

        n_layers = len(layers)
        x_positions = np.linspace(0.1, 0.9, n_layers)
        max_neurons = 12  # visual cap

        for i, (name, size, color) in enumerate(layers):
            n_vis = min(size, max_neurons)
            y_pos = np.linspace(0.1, 0.9, n_vis)
            # Draw connections to next layer
            if i < n_layers - 1:
                n_vis_next = min(layers[i+1][1], max_neurons)
                y_next = np.linspace(0.1, 0.9, n_vis_next)
                for y in y_pos:
                    for yn in y_next:
                        ax.plot([x_positions[i], x_positions[i+1]], [y, yn],
                                color=BORDER, linewidth=0.4, alpha=0.6, zorder=1)
            # Draw neurons
            for y in y_pos:
                circ = plt.Circle((x_positions[i], y), 0.025,
                                   color=color, zorder=2)
                ax.add_patch(circ)
            ax.text(x_positions[i], -0.02, name, ha="center", va="top",
                    fontsize=8, color=TEXT_S, transform=ax.transData)

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.12, 1.02)
        fig.tight_layout(pad=0.5)
        canvas = FigureCanvasTkAgg(fig, master=self.diag_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="x")

    # ── PAGE: TRAIN ───────────────────────────────────────────────────────────

    def _build_page_train(self):
        page = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        self.pages["train"] = page
        self._section_title(page, "Train")

        top = ctk.CTkFrame(page, fg_color="transparent")
        top.pack(fill="x", pady=(0,16))

        # Hyperparams
        hp_card = self._card(top, title="Hyperparameters")
        hp_card.pack(side="left", fill="both", expand=True, padx=(0,12))
        hp = ctk.CTkFrame(hp_card, fg_color="transparent")
        hp.pack(fill="x", padx=16, pady=(0,16))

        def _param(parent, label, default, row, col_offset=0):
            ctk.CTkLabel(parent, text=label, text_color=TEXT_S,
                         font=ctk.CTkFont(size=12), anchor="w").grid(
                row=row, column=col_offset, padx=(0,10), pady=6, sticky="w")
            var = ctk.StringVar(value=default)
            ctk.CTkEntry(parent, textvariable=var, width=100,
                          fg_color=BG_INPUT, border_color=BORDER).grid(
                row=row, column=col_offset+1, pady=6, sticky="w")
            return var

        self.var_epochs  = _param(hp, "Epochs",       "10", 0)
        self.var_lr      = _param(hp, "Learning rate","0.001", 1)
        self.var_batch   = _param(hp, "Batch size",   "32", 2)
        self.var_valsplit= _param(hp, "Validation %", "20", 3)

        # Param experiment card
        exp_card = self._card(top, title="Experiment Mode")
        exp_card.pack(side="left", fill="both", expand=True)
        exp = ctk.CTkFrame(exp_card, fg_color="transparent")
        exp.pack(fill="x", padx=16, pady=(0,16))

        ctk.CTkLabel(exp, text="Run 3 epoch experiments automatically",
                     text_color=TEXT_S, font=ctk.CTkFont(size=12)).pack(anchor="w")
        self.exp_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(exp, text="Auto-run epoch experiments (5/10/20)",
                        variable=self.exp_var, text_color=TEXT_P).pack(anchor="w", pady=8)

        # Train button + progress
        ctrl_card = self._card(page, title="Training Control")
        ctrl_card.pack(fill="x", pady=(0,16))
        ctrl = ctk.CTkFrame(ctrl_card, fg_color="transparent")
        ctrl.pack(fill="x", padx=16, pady=(0,16))

        self.train_btn = ctk.CTkButton(ctrl, text="▶  Start Training",
                                        fg_color=ACCENT, hover_color="#3A7DE8",
                                        font=ctk.CTkFont(size=13, weight="bold"),
                                        height=44, width=180, corner_radius=8,
                                        command=self._start_training)
        self.train_btn.pack(side="left", padx=(0,12))

        self.stop_btn = ctk.CTkButton(ctrl, text="■  Stop",
                                       fg_color=DANGER, hover_color="#D45050",
                                       height=44, width=100, corner_radius=8,
                                       state="disabled", command=self._stop_training_fn)
        self.stop_btn.pack(side="left", padx=(0,16))

        self.train_status = ctk.CTkLabel(ctrl, text="Ready", text_color=TEXT_S,
                                          font=ctk.CTkFont(size=12))
        self.train_status.pack(side="left")

        self.progress_bar = ctk.CTkProgressBar(ctrl_card, height=6,
                                                fg_color=BG_INPUT, progress_color=ACCENT)
        self.progress_bar.pack(fill="x", padx=16, pady=(0,6))
        self.progress_bar.set(0)

        # Log output
        log_card = self._card(page, title="Training Log")
        log_card.pack(fill="x", pady=(0,16))
        self.log_box = ctk.CTkTextbox(log_card, height=150, fg_color=BG_INPUT,
                                       font=ctk.CTkFont("Courier", 11),
                                       text_color=TEXT_P, border_width=0)
        self.log_box.pack(fill="x", padx=16, pady=(0,16))

        # Live plot
        plot_card = self._card(page, title="Live Training Curves")
        plot_card.pack(fill="both", expand=True)
        self.train_fig_frame = ctk.CTkFrame(plot_card, fg_color="transparent")
        self.train_fig_frame.pack(fill="both", expand=True, padx=16, pady=(0,16))
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

    def _train_worker(self):
        try:
            load_tf()
            epochs   = int(self.var_epochs.get())
            lr       = float(self.var_lr.get())
            batch    = int(self.var_batch.get())
            val_pct  = float(self.var_valsplit.get()) / 100
            h1       = int(self.var_h1.get() or 0)
            h2       = int(self.var_h2.get() or 0)
            h3       = int(self.var_h3.get() or 0)
            act      = self.var_act.get() or "relu"
            dropout  = float(self.var_dropout.get() or 0)

            self._log(f"Building model:  Input(784) → {h1} → {h2}" +
                      (f" → {h3}" if h3 else "") + " → Output(10)")

            layers = [Dense(h1, activation=act, input_shape=(784,))]
            if dropout > 0:
                from tensorflow.keras.layers import Dropout
                layers.append(Dropout(dropout))
            if h2 > 0:
                layers.append(Dense(h2, activation=act))
                if dropout > 0:
                    from tensorflow.keras.layers import Dropout
                    layers.append(Dropout(dropout))
            if h3 > 0:
                layers.append(Dense(h3, activation=act))
            layers.append(Dense(10, activation="softmax"))

            self.model = Sequential(layers)
            self.model.compile(optimizer=Adam(learning_rate=lr),
                                loss="categorical_crossentropy",
                                metrics=["accuracy"])

            total_params = self.model.count_params()
            self._log(f"Total parameters: {total_params:,}")
            self._log(f"Training for {epochs} epochs, batch={batch}, lr={lr}\n")

            acc_hist, val_hist, loss_hist, val_loss_hist = [], [], [], []

            class GUICallback(tf.keras.callbacks.Callback):
                def __init__(cb):
                    super().__init__()
                def on_epoch_end(cb, epoch, logs=None):
                    if self._stop_training:
                        cb.model.stop_training = True
                    a  = logs.get("accuracy", 0)
                    va = logs.get("val_accuracy", 0)
                    l  = logs.get("loss", 0)
                    vl = logs.get("val_loss", 0)
                    acc_hist.append(a); val_hist.append(va)
                    loss_hist.append(l); val_loss_hist.append(vl)
                    self._log(f"Epoch {epoch+1:2d}/{epochs}  "
                              f"acc={a:.4f}  val_acc={va:.4f}  "
                              f"loss={l:.4f}  val_loss={vl:.4f}")
                    self.progress_bar.set((epoch+1)/epochs)
                    self._update_train_plot(acc_hist, val_hist, loss_hist, val_loss_hist)

            self.model.fit(self.x_train, self.y_train,
                           epochs=epochs, batch_size=batch,
                           validation_split=val_pct,
                           callbacks=[GUICallback()],
                           verbose=0)

            # Quick eval
            _, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
            self._log(f"\n✓  Training complete!  Test accuracy: {acc*100:.2f}%")
            self.status_dot.configure(text=f"● {acc*100:.1f}% accuracy", text_color=SUCCESS)

            if self.exp_var.get():
                self._run_experiments()

        except Exception as e:
            self._log(f"✗  Error: {e}")
        finally:
            self.training = False
            self.train_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")

    def _update_train_plot(self, acc, val_acc, loss, val_loss):
        for w in self.train_fig_frame.winfo_children():
            w.destroy()
        fig = Figure(figsize=(9, 3), facecolor=BG_CARD)
        epochs = list(range(1, len(acc)+1))

        ax1 = fig.add_subplot(121)
        ax1.set_facecolor(BG_CARD)
        ax1.plot(epochs, acc, color=ACCENT, label="Train", linewidth=2)
        ax1.plot(epochs, val_acc, color=ACCENT2, label="Val", linewidth=2, linestyle="--")
        ax1.set_title("Accuracy", color=TEXT_S, fontsize=11)
        ax1.set_xlabel("Epoch", color=TEXT_S, fontsize=9)
        ax1.tick_params(colors=TEXT_S, labelsize=8)
        ax1.legend(fontsize=8, labelcolor=TEXT_S, facecolor=BG_INPUT, edgecolor=BORDER)
        ax1.set_facecolor(BG_CARD)
        for spine in ax1.spines.values(): spine.set_color(BORDER)
        ax1.grid(color=BORDER, linestyle="--", linewidth=0.5)

        ax2 = fig.add_subplot(122)
        ax2.plot(epochs, loss, color=DANGER, label="Train", linewidth=2)
        ax2.plot(epochs, val_loss, color=WARNING, label="Val", linewidth=2, linestyle="--")
        ax2.set_title("Loss", color=TEXT_S, fontsize=11)
        ax2.set_xlabel("Epoch", color=TEXT_S, fontsize=9)
        ax2.tick_params(colors=TEXT_S, labelsize=8)
        ax2.legend(fontsize=8, labelcolor=TEXT_S, facecolor=BG_INPUT, edgecolor=BORDER)
        ax2.set_facecolor(BG_CARD)
        for spine in ax2.spines.values(): spine.set_color(BORDER)
        ax2.grid(color=BORDER, linestyle="--", linewidth=0.5)

        fig.patch.set_facecolor(BG_CARD)
        fig.tight_layout(pad=1.5)
        canvas = FigureCanvasTkAgg(fig, master=self.train_fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _run_experiments(self):
        self._log("\n═══ Parameter Experiments ═══")
        lr = float(self.var_lr.get())
        batch = int(self.var_batch.get())
        h1 = int(self.var_h1.get() or 128)
        h2 = int(self.var_h2.get() or 64)
        act = self.var_act.get() or "relu"

        results = {}
        for ep in [5, 10, 20]:
            m = Sequential([Dense(h1, activation=act, input_shape=(784,)),
                             Dense(h2, activation=act), Dense(10, activation="softmax")])
            m.compile(optimizer=Adam(lr), loss="categorical_crossentropy", metrics=["accuracy"])
            m.fit(self.x_train, self.y_train, epochs=ep, batch_size=batch,
                  validation_split=0.2, verbose=0)
            _, acc = m.evaluate(self.x_test, self.y_test, verbose=0)
            results[f"{ep} epochs"] = acc
            self._log(f"  {ep:2d} epochs → {acc*100:.2f}%")
        self._log("\nExperiment complete!")

    # ── PAGE: EVALUATE ────────────────────────────────────────────────────────

    def _build_page_evaluate(self):
        page = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        self.pages["evaluate"] = page
        self._section_title(page, "Evaluate")

        btn_row = ctk.CTkFrame(page, fg_color="transparent")
        btn_row.pack(fill="x", pady=(0,16))

        ctk.CTkButton(btn_row, text="⬡  Run Evaluation",
                      fg_color=ACCENT, hover_color="#3A7DE8",
                      font=ctk.CTkFont(size=13, weight="bold"),
                      height=44, width=200, corner_radius=8,
                      command=self._run_evaluate).pack(side="left", padx=(0,12))

        self.eval_status = ctk.CTkLabel(btn_row, text="Train a model first",
                                         text_color=TEXT_S, font=ctk.CTkFont(size=12))
        self.eval_status.pack(side="left")

        # Metrics row
        self.metrics_row = ctk.CTkFrame(page, fg_color="transparent")
        self.metrics_row.pack(fill="x", pady=(0,16))

        # Confusion matrix
        self.cm_card = self._card(page, title="Confusion Matrix")
        self.cm_card.pack(fill="both", expand=True, pady=(0,16))
        self.cm_frame = ctk.CTkFrame(self.cm_card, fg_color="transparent")
        self.cm_frame.pack(fill="both", expand=True, padx=16, pady=(0,16))
        ctk.CTkLabel(self.cm_frame, text="Run evaluation to see confusion matrix",
                     text_color=TEXT_S).pack(pady=40)

        # Report
        self.report_card = self._card(page, title="Classification Report")
        self.report_card.pack(fill="x", pady=(0,16))
        self.report_box = ctk.CTkTextbox(self.report_card, height=200, fg_color=BG_INPUT,
                                          font=ctk.CTkFont("Courier", 11),
                                          text_color=TEXT_P, border_width=0)
        self.report_box.pack(fill="x", padx=16, pady=(0,16))

    def _run_evaluate(self):
        if self.model is None:
            messagebox.showerror("Error", "Train a model first.")
            return
        def _worker():
            self.eval_status.configure(text="⟳  Evaluating...", text_color=WARNING)
            try:
                loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)

                for w in self.metrics_row.winfo_children():
                    w.destroy()
                stats = [(f"{acc*100:.2f}%", "Test Accuracy", SUCCESS if acc > 0.9 else WARNING),
                         (f"{loss:.4f}", "Test Loss", TEXT_P),
                         (f"{10000:,}", "Test Samples", TEXT_P),
                         ("✓" if acc >= 0.9 else "✗", ">90% Target", SUCCESS if acc >= 0.9 else DANGER)]
                for val, lbl, col in stats:
                    c = self._card(self.metrics_row)
                    c.pack(side="left", fill="x", expand=True, padx=(0,10))
                    ctk.CTkLabel(c, text=val, font=ctk.CTkFont("Courier", 20, "bold"),
                                 text_color=col).pack(pady=(12,2))
                    ctk.CTkLabel(c, text=lbl, font=ctk.CTkFont(size=11),
                                 text_color=TEXT_S).pack(pady=(0,12))

                # Confusion matrix
                preds = np.argmax(self.model.predict(self.x_test, verbose=0), axis=1)
                true  = np.argmax(self.y_test, axis=1)
                cm    = confusion_matrix_fn(true, preds)

                for w in self.cm_frame.winfo_children():
                    w.destroy()

                import seaborn as sns
                fig = Figure(figsize=(7, 5.5), facecolor=BG_CARD)
                ax  = fig.add_subplot(111)
                ax.set_facecolor(BG_CARD)
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=range(10), yticklabels=range(10),
                            ax=ax, cbar=True, linewidths=0.3)
                ax.set_title("Confusion Matrix", color=TEXT_S, fontsize=12, pad=10)
                ax.set_xlabel("Predicted", color=TEXT_S); ax.set_ylabel("True", color=TEXT_S)
                ax.tick_params(colors=TEXT_S)
                fig.patch.set_facecolor(BG_CARD)
                fig.tight_layout()
                canvas = FigureCanvasTkAgg(fig, master=self.cm_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True)

                # Report
                report = classification_report_fn(true, preds,
                                                   target_names=[str(i) for i in range(10)])
                self.report_box.delete("1.0", "end")
                self.report_box.insert("1.0", report)

                self.eval_status.configure(text=f"✓  Done — {acc*100:.2f}% accuracy", text_color=SUCCESS)
            except Exception as e:
                self.eval_status.configure(text=f"✗  {e}", text_color=DANGER)
        threading.Thread(target=_worker, daemon=True).start()

    # ── PAGE: PREDICT ─────────────────────────────────────────────────────────

    def _build_page_predict(self):
        page = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        self.pages["predict"] = page
        self._section_title(page, "Predict")

        top = ctk.CTkFrame(page, fg_color="transparent")
        top.pack(fill="x", pady=(0,16))

        # Draw canvas
        draw_card = self._card(top, title="Draw a digit")
        draw_card.pack(side="left", fill="both", expand=True, padx=(0,12))

        self.canvas_size = 280
        self.draw_canvas = tk.Canvas(draw_card, width=self.canvas_size,
                                      height=self.canvas_size, bg="black",
                                      highlightthickness=1, highlightbackground=BORDER)
        self.draw_canvas.pack(padx=16, pady=8)
        self.draw_canvas.bind("<B1-Motion>", self._paint)
        self.draw_canvas.bind("<ButtonRelease-1>", self._reset_prev)

        btn_row = ctk.CTkFrame(draw_card, fg_color="transparent")
        btn_row.pack(fill="x", padx=16, pady=(0,12))
        ctk.CTkButton(btn_row, text="Clear", fg_color=BG_INPUT, hover_color=BORDER,
                      text_color=TEXT_P, width=80, height=32,
                      command=lambda: self.draw_canvas.delete("all")).pack(side="left", padx=(0,8))
        ctk.CTkButton(btn_row, text="⬡  Predict",
                      fg_color=ACCENT, hover_color="#3A7DE8",
                      text_color=TEXT_P, width=100, height=32,
                      command=self._predict_drawing).pack(side="left")

        # Result panel
        result_card = self._card(top, title="Prediction")
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

        # Upload from file
        file_card = self._card(page, title="Predict from Image File")
        file_card.pack(fill="x", pady=(0,16))
        file_inner = ctk.CTkFrame(file_card, fg_color="transparent")
        file_inner.pack(fill="x", padx=16, pady=(0,16))

        ctk.CTkButton(file_inner, text="📂  Browse Image",
                      fg_color=BG_INPUT, hover_color=BORDER, text_color=TEXT_P,
                      height=40, width=160, command=self._predict_file).pack(side="left", padx=(0,12))
        self.file_label = ctk.CTkLabel(file_inner, text="No file selected",
                                        text_color=TEXT_S, font=ctk.CTkFont(size=12))
        self.file_label.pack(side="left")

        self._prev_x = self._prev_y = None

    def _paint(self, event):
        r = 14
        x, y = event.x, event.y
        if self._prev_x and self._prev_y:
            self.draw_canvas.create_line(self._prev_x, self._prev_y, x, y,
                                          fill="white", width=r*2, capstyle=tk.ROUND, smooth=True)
        self.draw_canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
        self._prev_x, self._prev_y = x, y

    def _reset_prev(self, event):
        self._prev_x = self._prev_y = None

    def _predict_drawing(self):
        if self.model is None:
            messagebox.showerror("Error", "Train a model first.")
            return
        # Capture canvas as image
        ps = self.draw_canvas.postscript(colormode="gray")
        from PIL import Image, EpsImagePlugin
        try:
            img = Image.open(io.BytesIO(ps.encode("utf-8")))
        except:
            # Fallback: read pixel data directly via ImageGrab
            x0 = self.draw_canvas.winfo_rootx()
            y0 = self.draw_canvas.winfo_rooty()
            x1 = x0 + self.canvas_size
            y1 = y0 + self.canvas_size
            try:
                import PIL.ImageGrab
                img = PIL.ImageGrab.grab(bbox=(x0, y0, x1, y1)).convert("L")
            except:
                messagebox.showerror("Error", "Could not capture drawing. Try uploading a file.")
                return

        img = img.convert("L").resize((28, 28))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr.reshape(1, 784)
        self._show_prediction(arr)

    def _predict_file(self):
        if self.model is None:
            messagebox.showerror("Error", "Train a model first.")
            return
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")])
        if not path:
            return
        try:
            from PIL import Image
            img = Image.open(path).convert("L").resize((28, 28))
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = arr.reshape(1, 784)
            self.file_label.configure(text=os.path.basename(path))
            self._show_prediction(arr)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _show_prediction(self, arr):
        probs  = self.model.predict(arr, verbose=0)[0]
        digit  = np.argmax(probs)
        conf   = probs[digit] * 100
        self.pred_digit.configure(text=str(digit))
        self.pred_conf.configure(text=f"Confidence: {conf:.1f}%",
                                  text_color=SUCCESS if conf > 70 else WARNING)
        # Probability bars
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
            ctk.CTkLabel(row, text=f"{p*100:.1f}%", width=48,
                         font=ctk.CTkFont(size=10),
                         text_color=TEXT_P if i == digit else TEXT_S).pack(side="left")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = ANNApp()
    app.mainloop()