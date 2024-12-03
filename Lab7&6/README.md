
# Image Captioning with RNN & Transfer Learning

This project implements an **Image Captioning System** using **Transfer Learning** for feature extraction and a simple **Recurrent Neural Network (RNN)** for caption generation. The system describes images in natural language based on the **Flickr8k dataset**.

---

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Steps](#steps)
4. [Data Preprocessing](#data-preprocessing)
5. [Feature Extraction](#feature-extraction)
6. [Caption Generation](#caption-generation)
7. [Model Training](#model-training)
8. [Evaluation](#evaluation)
9. [Real-Time Implementation](#real-time-implementation)
10. [Results](#results)
11. [References](#references)

---

## 1. Overview
The system generates captions for images using:
- **Transfer Learning**: Pre-trained CNN models (InceptionV3) for feature extraction.
- **RNN Decoder**: Predicts captions based on extracted features.

---

## 2. Dataset
**Flickr8k Dataset**:
- Available on [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k).
- Contains 8,000 images and multiple captions per image.

---

## 3. Steps
### Phase 1: Feature Extraction
- Use **InceptionV3** (pre-trained on ImageNet) to extract high-level image features.

### Phase 2: Caption Generation
- Build an RNN-based model to generate captions.
- Train using processed image features and captions.

---

## 4. Data Preprocessing
### Image Preprocessing:
- Resize images to 299x299 pixels to match **InceptionV3** input size.
- Normalize pixel values and preprocess using `preprocess_input`.

### Caption Preprocessing:
- Convert captions to lowercase.
- Remove special characters and numbers.
- Add `<start>` and `<end>` tokens to each caption.
- Tokenize captions and create padded sequences.

### Tokenizer Creation:
- Limit vocabulary size to 10,000 most frequent words.
- Store the tokenizer for use during evaluation.

---

## 5. Feature Extraction
- Use **InceptionV3** to extract feature vectors for all images.
- Features saved as `.npy` files for efficient reuse.

```python
features = extract_features_in_batches(image_paths, batch_size=1000, save_path='image_features.npy')
```

---

## 6. Caption Generation
### Model Architecture:
1. **Image Features**:
   - Input shape: `(2048,)` (from InceptionV3).
   - Dense layer to reduce dimensionality.

2. **Text Input**:
   - Embedding layer for tokenized captions.
   - LSTM layer to learn sequences.

3. **Decoder**:
   - Merge image features and text embeddings.
   - Dense output layer for word prediction.

### Loss and Optimization:
- **Loss Function**: Categorical Cross-Entropy.
- **Optimizer**: Adam.

---

## 7. Model Training
- Train using **DataGenerator** for efficient batching.
- Apply callbacks for:
  - Early stopping.
  - Saving the best model.

```python
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]
)
```

### Training Graphs:
- Plot accuracy and loss over epochs.

---

## 8. Evaluation
### Metrics:
- **BLEU Scores**: Evaluate relevance of generated captions.
- Compute BLEU-1, BLEU-2, BLEU-3, and BLEU-4 scores.

```python
results = evaluate_model(model, test_data, tokenizer, max_length=40)
print(results)
```

---

## 9. Real-Time Implementation
### Steps:
1. Use OpenCV to capture live camera feed.
2. Extract features from the camera frames.
3. Generate captions using the trained model.

```python
real_time_caption(model, tokenizer, max_length=40)
```

---

## 10. Results
### Sample Captions:
- Example image captions from the test set and live feed.
- accuracy: 0.8337 - loss: 0.6811
- val_accuracy: 0.8116 - val_loss: 0.9897
---

## 11. References
- [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- Pre-trained model: [InceptionV3](https://keras.io/api/applications/inceptionv3/)

---
