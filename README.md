# Multiclass Tissue Classification (PathMNIST)

A multiclass classification pipeline built with Scikit-Learn to classify colorectal histopathology images into 9 tissue types, using the **PathMNIST** dataset (part of the [MedMNIST v2](https://medmnist.com/) benchmark).

This project is a hands-on exercise inspired by *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (Aurélien Géron) — especially the chapters on classification, multiclass strategies (One-vs-Rest / One-vs-One), and error analysis.

## Biological Background

PathMNIST is derived from real H&E-stained (Hematoxylin & Eosin) histological slides of colorectal cancer tissue. Each 28x28 RGB patch belongs to one of 9 tissue types, ranging from healthy structures (normal colon mucosa, adipose tissue) to tumor-related tissue (colorectal adenocarcinoma epithelium, cancer-associated stroma).

Automatically classifying tissue type from a small image patch is a real diagnostic-support task in digital pathology, and a good testbed for comparing classical ML approaches before moving to deep learning.

## Dataset

- Source: PathMNIST (MedMNIST v2)
- Train samples: 89,996 (subsampled to 10,000 for faster experimentation)
- Validation samples: 10,004
- Test samples: 7,180
- Feature representation: flattened from `(N, 28, 28, 3)` to `(N, 2352)`

| Class | Tissue | Train count (full) |
|---|---|---|
| 0 | Adipose | 9,366 |
| 1 | Background | 9,509 |
| 2 | Debris | 10,360 |
| 3 | Lymphocytes | 10,401 |
| 4 | Mucus | 8,006 |
| 5 | Smooth muscle | 12,182 |
| 6 | Normal colon mucosa | 7,886 |
| 7 | Cancer-associated stroma | 9,401 |
| 8 | Colorectal adenocarcinoma epithelium | 12,885 |

Classes are reasonably balanced compared to a typical rare-disease detection task, so **accuracy is a meaningful metric here** (unlike a minority-class detection setup).

## Pipeline

Three classic multiclass classifiers are trained and compared on the same train/test split:

**Model 1 — SGDClassifier (linear baseline)**
- `StandardScaler` → `SGDClassifier`
- Linear model: assigns one weight per pixel per class, decision = weighted sum of pixel intensities
- No native concept of shape or spatial structure
- 3-fold cross-validation for evaluation

**Model 2 — SVC (RBF kernel)**
- `StandardScaler` → `PCA(n_components=500)` → `SVC` (RBF kernel)
- Scikit-Learn automatically applies **One-vs-One (OvO)** for multiclass SVM
- PCA is used to make the RBF kernel computationally tractable on 2352-dimensional raw pixels
- 3-fold cross-validation for evaluation

**Model 3 — Random Forest**
- `RandomForestClassifier` (200 trees)
- Natively handles multiclass classification — no OvR/OvO needed
- Captures non-linear feature interactions directly on raw pixels, without PCA
- 3-fold cross-validation for evaluation

## Results

| Model | Test Accuracy | Mean CV Accuracy | Training Time |
|---|---|---|---|
| **SVC (RBF + PCA)** | **64.42%** | 61.40% | 95.9s |
| Random Forest | 62.94% | 62.17% | 31.8s |
| SGDClassifier (linear) | 40.21% | 32.22% | 113.1s |

### SGDClassifier — per-class detail

| Tissue | Precision | Recall | F1 |
|---|---|---|---|
| Adipose | 0.9932 | 0.4357 | 0.6057 |
| Background | 0.9411 | 1.0000 | 0.9697 |
| Debris | 0.0655 | 0.1150 | 0.0835 |
| Lymphocytes | 0.1632 | 0.2240 | 0.1888 |
| Mucus | 0.4494 | 0.5401 | 0.4906 |
| Smooth muscle | 0.3259 | 0.2956 | 0.3100 |
| Normal colon mucosa | 0.1611 | 0.1633 | 0.1622 |
| Cancer-associated stroma | 0.0994 | 0.1853 | 0.1294 |
| Colorectal adenocarcinoma epithelium | 0.3765 | 0.2782 | 0.3200 |

### SVC (RBF + PCA) — per-class detail (best model)

| Tissue | Precision | Recall | F1 |
|---|---|---|---|
| Adipose | 0.8679 | 0.9185 | 0.8925 |
| Background | 0.6870 | 0.9976 | 0.8137 |
| Debris | 0.4367 | 0.7729 | 0.5580 |
| Lymphocytes | 0.5413 | 0.2792 | 0.3684 |
| Mucus | 0.6343 | 0.4609 | 0.5339 |
| Smooth muscle | 0.6278 | 0.4730 | 0.5395 |
| Normal colon mucosa | 0.5024 | 0.2794 | 0.3591 |
| Cancer-associated stroma | 0.6491 | 0.3515 | 0.4561 |
| Colorectal adenocarcinoma epithelium | 0.5653 | 0.8110 | 0.6662 |

### Random Forest — per-class detail

| Tissue | Precision | Recall | F1 |
|---|---|---|---|
| Adipose | 0.8578 | 0.9604 | 0.9062 |
| Background | 0.9627 | 0.9764 | 0.9695 |
| Debris | 0.5056 | 0.2655 | 0.3482 |
| Lymphocytes | 0.2815 | 0.2571 | 0.2688 |
| Mucus | 0.5800 | 0.4937 | 0.5334 |
| Smooth muscle | 0.4657 | 0.5845 | 0.5184 |
| Normal colon mucosa | 0.7143 | 0.1687 | 0.2729 |
| Cancer-associated stroma | 0.5194 | 0.2542 | 0.3413 |
| Colorectal adenocarcinoma epithelium | 0.5167 | 0.8637 | 0.6466 |

## Error Analysis

Confusion matrix analysis was run on the best-performing model (SVC).

**Biggest mutual confusion pair:** `normal colon mucosa` (class 6) ↔ `colorectal adenocarcinoma epithelium` (class 8) — **584 total confused samples**.

This is consistent across all three models: normal colon mucosa is repeatedly predicted as adenocarcinoma epithelium (e.g. 443/480 misclassifications for SVC/RF respectively going in that direction). Visual inspection of example patches (see `error_analysis_worst_pair_*.png`) shows both classes share the same glandular, ring-shaped structure and pink/purple H&E staining. The main visual difference — gland regularity and spacing — is subtle at 28x28 resolution and is not something raw flattened pixels + a linear or kernel-distance model can reliably capture.

Two other consistent confusion patterns across models:
- `lymphocytes` ↔ `mucus` (and `colorectal adenocarcinoma epithelium`) — small, densely packed structures are visually ambiguous at this resolution.
- `debris` ↔ `smooth muscle` / `cancer-associated stroma` — irregular textures without a clear shape signature.

See the saved figures for the confusion matrix heatmaps (raw counts and row-normalized error rates) and side-by-side example images for the most-confused pair.

## Why the Models Differ So Much

- **SGDClassifier (40.2% accuracy)** is a linear model: it only weighs raw pixel intensities per class, with no notion of shape, texture, or spatial structure. Since several tissue classes differ mainly in spatial arrangement rather than overall color/intensity, this model struggles badly — it even failed to fully converge (`ConvergenceWarning`) within the iteration budget used here.
- **SVC with RBF kernel + PCA (64.4% accuracy)** can model non-linear decision boundaries in a reduced 500-dimensional PCA space, giving it the best test accuracy of the three, at the cost of the longest inference/training pipeline.
- **Random Forest (62.9% accuracy)** performs close to SVC without needing PCA at all, and trains ~3x faster than SVC. It struggles most on classes with fine internal texture (`normal colon mucosa` recall = 16.9%).

## Limitations

- **Flattened pixels lose spatial structure.** None of these models see the image as a 2D grid — they only see a 1D vector of pixel values, so local texture, shape, and spatial layout are not represented as such.
- **Very small image size.** Each patch is only 28x28, low resolution for histopathology; fine-grained tissue detail may be lost.
- **Visual similarity between classes.** As shown in the error analysis, several tissue types are genuinely hard to separate at this resolution, even with a strong classifier.
- **Subsampled training set.** Training was run on 10,000 of the ~90,000 available training images to keep runtime practical on a laptop; using the full training set would likely improve all three models further.
- **SGD did not fully converge** within `max_iter=1000`; results might improve somewhat with more iterations or a smaller learning rate, but the underlying linear-model limitation would remain.

## Interpretation

The main takeaway is that a linear model is not expressive enough for this task, while both a non-linear kernel method (SVM) and an ensemble of trees (Random Forest) reach similar, meaningfully better accuracy — with Random Forest being far cheaper to train. The dominant error source across all models is the confusion between **normal** and **cancerous** colon tissue, which is exactly the distinction that matters most clinically. This suggests that classical pixel-based models, even non-linear ones, are hitting a representation ceiling: a CNN-based approach that can learn local texture and shape features directly would likely close much of this gap.

## Next Steps

- Train on the full 90,000-image training set instead of a 10,000-sample subset.
- Replace flattened pixel features with a CNN or transfer-learning model (e.g. a small ResNet).
- Add data augmentation (rotation, flips, color jitter) to improve robustness to staining variation.
- Try texture-based features (GLCM, LBP) as a middle ground between raw pixels and deep features.
- Apply stain normalization to reduce color variation between slides.
- Tune SVM (`C`, `gamma`) and Random Forest (`max_depth`, `min_samples_leaf`) hyperparameters with `GridSearchCV`.

## Author

Amirhossein
Email: amirhossein070905@gmail.com
Telegram: [@itsamirhosseingadimi](https://t.me/itsamirhosseingadimi)
