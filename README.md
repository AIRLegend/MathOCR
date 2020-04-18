# Math OCR

Latex formula recognizer using TensorFlow 2.

**NOTE: This is currently a work in progress.**

## Steps to run

1. Download Im2Latex dataset:

```
cd MathOCR/src/data &&\
    python im2latex_downloader.py
```
2. Preprocess the data:

```
cd MathOCR/src/data &&\
    python im2latex_proccess.py
```

3. Train the model:

```
cd MathOCR/src/models/ &&\
    python train.py --processed_data ../../data/processed --savedir ../../models --restore --epochs 10
```

