# Speech Command Recognition

An educational PyTorch project for audio classification, developed for the [TensorFlow Speech Recognition Challenge on Kaggle](https://www.kaggle.com/competitions/tensorflow-speech-recognition-challenge/). The repository compares several approaches to speech-command recognition, from Mel-spectrogram CNNs to learnable waveform filters and a Transformer-based encoder.

The main objective is to investigate audio representations and model architectures, with a particular focus on knowledge distillation between models that process the same audio in different ways.

## Table of Contents

- [Project Overview](#project-overview)
- [Kaggle Results](#kaggle-results)
- [Repository Structure](#repository-structure)
- [Dataset and Preprocessing](#dataset-and-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Architectures](#model-architectures)
- [Knowledge Distillation](#knowledge-distillation)
- [Training and Evaluation](#training-and-evaluation)
- [Running the Notebooks](#running-the-notebooks)
- [Requirements](#requirements)
- [License](#license)

## Project Overview

The project explores several approaches to classifying short audio recordings:

- Exploratory analysis of waveforms, spectrograms, class distributions and recording properties.
- A convolutional neural network operating on Mel power spectrograms.
- A SincNet-inspired model that learns frequency-selective filters directly from waveforms.
- A compact convolutional and Transformer encoder inspired by wav2vec 2.0.
- Knowledge distillation using both intermediate-feature alignment and teacher-prediction matching.
- Evaluation and preparation of predictions for Kaggle submission.

Experiments are organized as Jupyter notebooks. Model components, dataset loading, training utilities and visualization helpers are implemented in `src/`.

## Kaggle Results

The reported competition scores are:

| Metric | Score |
|---|---:|
| Public score | **0.86842** |
| Private score | **0.87313** |

No audio augmentation was used. Training was performed only on the training split of the provided training dataset. Given these limitations, the result is a useful outcome for an educational implementation, rather than a claim of a fully optimized competition solution.

## Repository Structure

```text
.
├── LICENSE
├── notebooks/
│   ├── distillation.ipynb
│   ├── eda.ipynb
│   ├── mel.ipynb
│   ├── sincnet.ipynb
│   ├── submission.ipynb
│   └── wav2vec.ipynb
└── src/
    ├── dataset.py
    ├── train.py
    ├── visualization.py
    └── model/
        ├── __init__.py
        ├── attentionpooling1d.py
        ├── downsample/
        │   ├── downsample1d.py
        │   └── downsample2d.py
        ├── featureencoder.py
        ├── featureprojection.py
        ├── head.py
        ├── mel.py
        ├── positionalembedding.py
        ├── res/
        │   ├── resblock1d.py
        │   └── resblock2d.py
        ├── se/
        │   ├── seblock1d.py
        │   └── seblock2d.py
        ├── sincconv1d.py
        ├── sincnet.py
        ├── transformerblock.py
        ├── transformerencoder.py
        └── wav2vec2.py
```

## Dataset and Preprocessing

The dataset loader in `src/dataset.py` supports the labeled Speech Commands directory layout and the flat, unlabeled Kaggle test directory. Audio is converted to mono when needed and padded or truncated to **16,000 samples**, corresponding to one second at 16 kHz.

The classifier uses the ten target commands:

`yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`.

Two additional classes are included:

- **`unknown`** - recordings belonging to other word classes.
- **`silence`** - examples derived from background noise, or low-amplitude synthetic noise when background recordings are unavailable.

The dataset factory uses the dataset's validation and testing file lists to create the corresponding splits. Training notebooks can exclude recordings flagged as anomalies. The experiments documented here do not use audio augmentation.

## Exploratory Data Analysis

The `eda.ipynb` notebook investigates the dataset before model training. It examines the label distribution and recording properties, and compares audio representations that expose different aspects of the signal.

The analysis includes:

- Raw waveform visualization.
- Short-time Fourier transform (STFT) spectrograms.
- Mel spectrograms and MFCCs.
- Spectral comparisons across commands and speakers.
- Sampling-rate and recording checks.
- Inspection of class distribution and anomalous recordings.

The purpose of this stage is to understand the input data and identify potential data-quality issues before choosing model architectures. Recordings flagged as anomalies can be excluded by the training notebooks; this is separate from audio augmentation, which is not used in these experiments.

## Model Architectures

### 1. MelResNetKWS - a time-frequency CNN

Implemented in `src/model/mel.py`, `MelResNetKWS` converts the waveform into a **64-bin Mel power spectrogram** using a 400-sample FFT/window and a 160-sample hop. The power spectrogram is converted to decibels and normalized per example.

A two-dimensional residual CNN learns patterns over time and frequency. Its main stages are:

1. A convolutional stem.
2. Residual stages with increasing channel counts and downsampling.
3. Attention pooling over the remaining temporal dimension.
4. A fully connected classification head.

This architecture uses a hand-designed, perceptually motivated time-frequency representation. It is also the teacher model used in the main distillation experiment.

### 2. SincResNetKWS - a learnable-filter waveform CNN

Implemented in `src/model/sincnet.py`, `SincResNetKWS` operates directly on the waveform instead of requiring a Mel spectrogram as input.

Its frontend uses `SincConv1d` with 80 parameterized band-pass filters, a kernel size of 251, a sample rate of 16 kHz and a stride of 10. Rather than learning every filter coefficient independently, the Sinc layer parameterizes filters through their cutoff frequencies.

The resulting filter responses pass through a one-dimensional convolutional stem, residual stages with dilations and downsampling blocks. Attention pooling compresses the temporal feature map into a fixed-size representation, which is then passed to the classification head.

The main difference from `MelResNetKWS` is how the initial representation is obtained: the Mel model starts from a computed Mel spectrogram, while the Sinc model learns frequency-selective filters directly from raw audio.

### 3. Wav2Vec2KWS - a waveform encoder with a Transformer

Implemented in `src/model/wav2vec2.py`, `Wav2Vec2KWS` is a compact task-specific architecture inspired by wav2vec 2.0. It is **not** a pretrained wav2vec 2.0 checkpoint.

The model consists of:

1. A stack of strided 1D convolutions that extracts waveform features and reduces sequence length.
2. Layer normalization and a linear projection to a 384-dimensional representation.
3. A grouped convolutional positional embedding.
4. Six Transformer encoder blocks with multi-head self-attention and feed-forward layers.
5. Global average pooling and a linear classification layer.

This model provides a Transformer-based comparison to the convolutional architectures.

### Shared Components

The `src/model/` package contains reusable components used across the architectures:

- One-dimensional and two-dimensional residual blocks.
- Squeeze-and-excitation blocks.
- Downsampling modules.
- Attention pooling and classification head.
- Convolutional feature encoder and feature projection.
- Positional embedding and Transformer blocks.

The models produce logits for the 12 classification classes: the ten target commands, `unknown` and `silence`.

## Knowledge Distillation

The main experiment is implemented in `notebooks/distillation.ipynb`. It studies whether a waveform-based SincNet model can benefit from a model trained on Mel spectrograms. Both models receive the same audio examples, but their backbones process different representations and have different architectural biases.

### Teacher and Student

- **Teacher:** a pretrained `MelResNetKWS` loaded from `mel.pt`.
- **Student:** a separately initialized `SincResNetKWS`.
- **Shared classification head:** both backbones produce 256-dimensional feature vectors, allowing them to use the same `ClassificationHead`.

The notebook first copies the teacher's classification head into the student and freezes the head while training the student backbone with the standard classification objective. It then runs a joint distillation stage. A frozen copy of the pretrained Mel model provides teacher predictions, while the Mel and Sinc backbones are optimized for classification and encouraged to produce similar representations.

### Loss Functions

The joint stage combines three objectives.

#### 1. Supervised classification loss

Both backbones are evaluated with the shared classification head and the ground-truth labels. Their cross-entropy losses are averaged:

$$L_{CE} = \frac{1}{2}\left(CE(h(z_m), y) + CE(h(z_s), y)\right)$$

Here, $z_m$ and $z_s$ are the Mel and Sinc backbone features, $h$ is the shared classification head, and $y$ is the target label. This term keeps the models anchored to the actual classification task.

#### 2. Intermediate-feature alignment

The two backbones are encouraged to produce similar 256-dimensional feature vectors for the same input using mean squared error:

$$L_{align} = MSE(z_m, z_s)$$

This is representation-level distillation. Instead of transferring knowledge only through final class predictions, it directly encourages the waveform model's representation to align with the Mel model's representation.

The alignment weight increases with the epoch number during joint training:

$$L_{\text{feature}} = \lambda_{\text{align}} \cdot epoch \cdot L_{align}$$

The notebook sets `lambda_align = 0.1`.

#### 3. Prediction-level knowledge distillation

The frozen teacher produces logits for each input. Teacher and student logits are converted to softened probability distributions using temperature \(T=2\). The student is trained to match the teacher distribution using KL divergence:

$$L_{KD} = T^2 D_{KL}\left(softmax(a/T) \,\|\, softmax(s/T)\right)$$

Here, $a$ denotes teacher logits and $s$ denotes student logits. The notebook computes this loss for both backbones against the frozen teacher and averages the two terms. The factor $T^2$ follows the common knowledge-distillation formulation.

### Combined Objective

The joint objective is:

$$L = L_{CE} + \lambda_{\text{align}} \cdot epoch \cdot L_{align} + \lambda_{\text{KD}} L_{KD}$$

The notebook uses `lambda_align = 0.1` and `lambda_kd = 1.0`. The resulting objective combines ground-truth supervision, intermediate-feature alignment and teacher-prediction matching.

Each term plays a different role:

- Cross-entropy trains the models to predict the correct labels.
- Feature alignment transfers information about the teacher's internal representation.
- Prediction-level distillation transfers the teacher's relative preferences across classes, including information about plausible alternative labels.

This combination makes it possible to investigate whether a model operating directly on raw audio can benefit from a model trained on a Mel representation.

### Training Schedule

The distillation notebook uses a staged training procedure:

1. **Student warm-up.** The teacher's classification head is copied to the student and frozen. The SincNet backbone is trained using the standard training loop.
2. **Joint distillation, epochs 1–5.** The Mel and Sinc backbones are optimized while the shared classification head remains frozen. The teacher copy is also frozen and provides the distillation targets.
3. **Joint distillation, epochs 6–10.** The shared classification head is unfrozen, allowing the classifier and both backbones to adapt together.

The joint optimizer uses AdamW with separate learning rates for the backbones (`3e-5`) and the shared head (`1e-5`), and weight decay `1e-4`. Gradients are clipped to a maximum norm of 5. The notebook records the total loss and its cross-entropy, feature-alignment and distillation components.

The saved `distillated.pt` checkpoint contains the two backbone state dictionaries and the shared head. The notebook reloads these components to evaluate the resulting Mel and Sinc models and inspect their feature similarity.

### What the Experiment Tests

Prediction-level distillation communicates the teacher's relative preferences among classes. Feature alignment adds a direct constraint on the representations produced by the two different backbones. Cross-entropy ensures that the models remain supervised by the ground-truth labels rather than merely imitating the teacher.

The experiment is intended as an educational comparison of these transfer mechanisms. It does not assume that representations from different architectures are inherently equivalent, nor that distillation must improve the student's score. Its effect should be evaluated on the same held-out split as the baseline models.

## Training and Evaluation

The training utilities in `src/train.py` provide epoch-based training and validation, metric reporting and best-checkpoint saving. The model notebooks use cross-entropy and AdamW, with a `ReduceLROnPlateau` learning-rate scheduler.

Evaluation includes:

- Accuracy.
- Precision, recall and F1 score.
- Confusion matrices.
- Validation and test-split evaluation.
- Checkpoint selection based on validation performance.

A typical notebook workflow is:

1. Configure the dataset path.
2. Create train, validation and test datasets through `SpeechCommandsDatasetFactory`.
3. Create data loaders.
4. Instantiate and train the selected model.
5. Load the best checkpoint and evaluate the model.
6. Use `submission.ipynb` to generate predictions in the Kaggle submission format.

The notebooks are the primary entry points. The repository does not currently provide one universal command-line configuration for all experiments; model-specific training settings are defined in the corresponding notebooks.

## Running the Notebooks

A suggested exploration order is:

| Order | Notebook | Purpose |
|---:|---|---|
| 1 | `eda.ipynb` | Inspect audio, labels, signal representations and potential anomalies |
| 2 | `mel.ipynb` | Train and evaluate the Mel-spectrogram CNN |
| 3 | `sincnet.ipynb` | Train and evaluate the SincNet-based waveform CNN |
| 4 | `wav2vec.ipynb` | Train and evaluate the Transformer-based waveform model |
| 5 | `distillation.ipynb` | Run feature alignment and prediction-level knowledge distillation |
| 6 | `submission.ipynb` | Prepare Kaggle predictions |

Before running a notebook, check its first cells for expected input files, dataset paths, checkpoint names and hardware requirements. The distillation experiment expects the pretrained Mel checkpoint `mel.pt`; the notebook saves the resulting distillation checkpoint as `distillated.pt`.

## Requirements

The project is implemented in Python and PyTorch. Depending on the notebook, it also uses libraries for audio processing, data handling, visualization and evaluation, including `torchaudio`, `librosa`, `soundfile`, NumPy, pandas, Matplotlib, scikit-learn and tqdm.

Install PyTorch and `torchaudio` versions that are compatible with each other and with the target operating system and accelerator.

## License

This project is distributed under the MIT License. See [`LICENSE`](LICENSE) for details.
