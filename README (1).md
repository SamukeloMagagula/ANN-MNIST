# AIN58XS — ANN MNIST Notebook

**Cape Peninsula University of Technology**  
Artificial Neural Network Analysis — Handwritten Digit Recognition

---

## Requirements

- Python **3.9 – 3.12**
- pip (comes bundled with Python)

---

## Installation

### 1. Clone or download the project

```bash
git clone <your-repo-url>
cd <project-folder>
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv venv
```

Activate it:

| Platform | Command |
|----------|---------|
| Windows | `venv\Scripts\activate` |
| macOS / Linux | `source venv/bin/activate` |

### 3. Install dependencies

```bash
pip install tensorflow scikit-learn seaborn matplotlib numpy pillow scipy
```

Or install from the requirements file:

```bash
pip install -r requirements.txt
```

### 4. Launch the notebook

```bash
pip install jupyter
jupyter notebook ann_mnist_fixed.ipynb
```

---

## requirements.txt

```
tensorflow>=2.13
scikit-learn>=1.3
seaborn>=0.12
matplotlib>=3.7
numpy>=1.24
pillow>=10.0
scipy>=1.11
jupyter>=1.0
```

> Copy the block above into a file named `requirements.txt` in your project folder if you don't have one yet.

---

## Predicting your own hand-drawn digits (Part 2)

1. Draw digits **0–9** in MS Paint (or any image editor).  
2. Save each image as a `.jpg` or `.png` — any size is fine, the notebook resizes automatically.  
3. Place all images in a folder called `own_digits/` next to the notebook.  
4. Run **Cell 33** — the preprocessing pipeline will handle the rest:

   | Step | What it does |
   |------|-------------|
   | Greyscale conversion | Strips colour channels |
   | Median filter | Removes scan / camera noise |
   | Auto-contrast | Boosts faint strokes |
   | Resize to 28 × 28 | Matches MNIST format |
   | Auto-invert | Flips black-on-white → white-on-black |
   | Centre-of-mass shift | Centres the digit like MNIST training data |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'tensorflow'`**  
Run `pip install tensorflow` and make sure your virtual environment is activated.

**TensorFlow install fails on Apple Silicon (M1/M2/M3)**  
Use the Apple-optimised build instead:
```bash
pip install tensorflow-macos tensorflow-metal
```

**`pip` not found**  
Try `pip3` instead of `pip`, or run `python -m pip install ...`.

**Jupyter notebook doesn't open**  
Make sure Jupyter is installed (`pip install jupyter`) and run `jupyter notebook` from the project folder.
