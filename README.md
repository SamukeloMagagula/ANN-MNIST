# ANN MNIST Neural Network Studio

A full desktop GUI application for building, training, and testing an Artificial Neural Network on the MNIST handwritten digit dataset. Built with Python, TensorFlow/Keras, and CustomTkinter.

---

## What It Does

This app lets you load the MNIST dataset, configure a neural network architecture, train it with live feedback, evaluate its performance, and test it by drawing digits with your mouse or uploading your own images — all from a single dark-themed desktop interface.

---

## Requirements

- Python 3.8 or higher
- pip

Install all dependencies with:

```bash
pip install tensorflow customtkinter pillow scikit-learn seaborn matplotlib numpy
```

---

## Running the App

```bash
python ann_mnist_gui.py
```

If you also want to use the plain code version (no GUI):

```bash
python ann_mnist.py
```

---

## App Tabs

### Dataset
Loads the MNIST dataset directly from Keras (downloads automatically on first run). Displays a grid of 20 sample images with their labels and shows dataset statistics — 60,000 training images, 10,000 test images, all 28×28 pixels across 10 digit classes.

### Architecture
Configure your network before training:
- Number of hidden layers (1, 2, or 3)
- Neurons per layer
- Activation function (relu, sigmoid, tanh)
- Dropout rate (set to 0 to disable)

A network diagram updates live as you change the settings.

### Train
Set your hyperparameters and start training:
- **Epochs** — how many full passes over the training data (default: 10)
- **Learning rate** — step size for the Adam optimizer (default: 0.001)
- **Batch size** — samples per weight update (default: 32)
- **Validation split** — percentage of training data held out for validation (default: 20%)

A scrolling log shows live epoch-by-epoch output. Accuracy and loss curves update in real time as each epoch completes. You can stop training early at any point. Enable **Experiment Mode** to automatically run three training runs at 5, 10, and 20 epochs and compare results.

### Evaluate
Runs a full evaluation on the 10,000 test images. Shows:
- Test accuracy and loss
- Whether the >90% target was met
- A colour-coded confusion matrix heatmap
- A per-class classification report (precision, recall, F1-score)

### Predict
Two ways to test the model on real input:

**Draw mode** — draw a digit on the black canvas using your mouse, then click Predict. The app preprocesses your drawing to match MNIST format and displays the predicted digit with a confidence score and probability bars for all 10 classes.

**File mode** — browse and load any JPG or PNG image of a handwritten digit. The app converts it to grayscale, resizes it to 28×28, and runs a prediction.

---

## File Structure

```
ann_mnist_gui.py    ← Full GUI application (run this)
ann_mnist.py        ← Plain script version (no GUI, good for reference)
README.md           ← This file
```

Generated output files (created when you run `ann_mnist.py`):
```
sample_images.png       ← Grid of 20 MNIST sample images
training_history.png    ← Accuracy and loss curves
confusion_matrix.png    ← Confusion matrix heatmap
```

---

## Expected Results

| Dataset       | Expected Accuracy |
|---------------|-------------------|
| MNIST test    | ~97%              |
| Custom images | ~60–85%           |
| Target        | > 90% ✓           |

Custom image accuracy is lower because hand-drawn digits differ from the standardised MNIST format in stroke thickness, centering, background colour, and style. This is a normal and expected result.

---

## Troubleshooting

**App window doesn't open**
Make sure `customtkinter` is installed: `pip install customtkinter`

**TensorFlow errors on first run**
TensorFlow can take a few seconds to initialise. Any warnings about CUDA or GPU drivers are normal on machines without a GPU — training will run on CPU.

**Drawing prediction is wrong**
Try drawing the digit larger, centred, and with a thick stroke. The model expects white digits on a black background, which is what the canvas uses by default.

**Low accuracy on custom images**
This is expected. See the Results Comparison section of the report for a full explanation of domain shift.

---

## How the Neural Network Works

1. Each 28×28 image is flattened into a 784-element vector
2. Pixel values are normalised from 0–255 to 0.0–1.0
3. Labels are one-hot encoded into 10-element vectors
4. The network passes input through Dense layers with ReLU activations
5. The output layer uses Softmax to produce a probability distribution over 10 classes
6. The Adam optimizer minimises categorical crossentropy loss via backpropagation
7. After training, the class with the highest output probability is the predicted digit

---

## Built With

- [TensorFlow / Keras](https://www.tensorflow.org/) — neural network framework
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) — modern dark-themed GUI
- [scikit-learn](https://scikit-learn.org/) — confusion matrix and classification metrics
- [Matplotlib](https://matplotlib.org/) — training curves and sample image display
- [Seaborn](https://seaborn.pydata.org/) — confusion matrix heatmap
- [Pillow](https://python-pillow.org/) — image loading and preprocessing
