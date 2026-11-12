<div align="center">

# ୨୧ ˚₊‧ Voice Recognition ‧₊˚ ୨୧

**a little voice, a little math, and a lot of pink 🎀**

*Two-Speaker Voice Identification using Linear Regression & Gradient Descent*

![Python](https://img.shields.io/badge/Python-3.11%2B-EFA6C8?style=for-the-badge&logo=python&logoColor=white)
![Numerical Methods](https://img.shields.io/badge/Numerical_Methods-Project-F4B6CF?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-In_Progress-D98FB4?style=for-the-badge)
![Presentation](https://img.shields.io/badge/Presentation-24_Nov_2026-C77FA3?style=for-the-badge)

♡ ─── ⋆⋅☆⋅⋆ ─── ♡

*Made for our Numerical Methods final presentation.*

</div>

## ୨୧ about our little project ♡

We're building a small **two-speaker identification** system from scratch. Instead of using a pretrained voice AI, we'll turn audio into numerical features, train a **linear regression model**, and optimize its parameters using **gradient descent**.

> 🎀 **Our goal:** explain the numerical method clearly — how the error is calculated, how gradients update the weights, and how the algorithm converges.

**Scope:** recognize two *known* speakers (A and B) from short WAV recordings. Identifying speakers in a longer conversation is a stretch goal; overlapping speech and speech-to-text are out of scope.

## ୨୧ meet the team ♡

| Member | Our little mission | Files |
|:--|:--|:--|
| 🎀 **Regina** | Regression, loss function, gradient descent, convergence | `src/model.py`, `src/gradient_descent.py` |
| 🩰 **Aurora** | Audio preprocessing, pitch, ZCR, RMS, dataset coordination | `src/audio_processing.py`, `src/features.py` |
| 🧸 **Putra** | Testing, evaluation, confusion matrix, final demo | `evaluate.py`, `demo.py` |

<div align="center">

**୨୧ one repo · three contributors · one working project ୨୧**

</div>

## ୨୧ how it works ♡

```text
   🎙️ Voice recording (.wav)
               │
               ▼
   ♡ Audio preprocessing
               │
               ▼
   ♡ Feature extraction
     [pitch, zcr, rms]
               │
               ▼
   ♡ Standardization
     (training data only)
               │
               ▼
   ♡ Linear Regression
     + Gradient Descent
               │
               ▼
   🎀 Speaker A / Speaker B
```

### the numerical magic ✧

**Linear regression:**

$$\hat{y}_i = w_1x_{i1} + w_2x_{i2} + w_3x_{i3} + b$$

**Squared-error objective (half MSE):**

$$J(w,b) = \frac{1}{2N}\sum_{i=1}^{N}(\hat{y}_i-y_i)^2$$

**Gradient descent updates:**

$$w_j \leftarrow w_j - \alpha\frac{\partial J}{\partial w_j}$$

$$b \leftarrow b - \alpha\frac{\partial J}{\partial b}$$

Here, `0 = Speaker A`, `1 = Speaker B`, and $\alpha$ is the learning rate. Our linear model produces **scores, not calibrated probabilities**. A threshold of `0.5` turns a score into an A/B prediction.

## ୨୧ our project structure ♡

> The tree below is our **planned layout**. Not every file exists yet — we're building it together, one branch at a time.

```text
Voice-recognition/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── audio_processing.py      # Aurora
│   ├── features.py              # Aurora
│   ├── model.py                 # Regina
│   └── gradient_descent.py      # Regina
├── tests/
├── experiments/
├── figures/
├── data/                        # local/private; not committed
├── evaluate.py                  # Putra
├── demo.py                      # Putra
├── requirements.txt
├── .gitignore
└── README.md
```

## ୨୧ setup guide ♡

**1 · Clone this repository**

```powershell
git clone https://github.com/yunkixr/Voice-recognition.git
cd Voice-recognition
```

**2 · Create a virtual environment (Windows / PowerShell)**

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**3 · Run the project**

Our end-to-end demo is still **in development**. Once `demo.py` is merged, we'll document its exact command here. Please don't expect the demo to run from the initial repo structure alone. ♡

## ୨୧ shared settings ♡

| Setting | Agreement |
|:--|:--|
| Audio format | WAV · mono · 16-bit PCM · **8 kHz** |
| Feature order | `[pitch_hz, zcr, rms]` |
| Labels | `A = 0`, `B = 1` |
| Model | Linear regression with manual batch gradient descent |
| Dataset | Separate training / validation / test clips |
| Privacy | Keep raw audio **out of Git**; only record/share with permission |

### our coding contract ✧

- **Aurora → Regina:** each WAV becomes a numeric feature vector in the agreed feature order.
- **Regina → Putra:** a trained model accepts new features and returns speaker scores/labels.
- **Putra → everyone:** evaluation results and a reproducible demo.
- Agree on function signatures **before merging**; coordinate any interface changes in the group chat.

## ୨୧ our mini timeline ♡

| Date | Milestone |
|:--|:--|
| **13 Nov** | GitHub structure & branches ready |
| **15 Nov** | Dataset ready |
| **16 Nov** | Audio features ready |
| **17 Nov** | Regression + gradient descent ready |
| **18 Nov** | First end-to-end integration |
| **20 Nov** | Experiments & evaluation |
| **21 Nov** | Code freeze |
| **22 Nov** | Slides finished |
| **23 Nov** | Rehearsal & backup demo |
| **24 Nov 2026** | ✨ Presentation day! |

## ୨୧ working together on GitHub ♡

Everyone works in a separate branch:

- `aurora-audio` — preprocessing and feature extraction
- `regina-model` — numerical model and gradient descent
- `putra-demo` — evaluation and demo

Our flow: **branch → code → test → commit → push → pull request → review → merge**.

```powershell
git switch main
git pull origin main
git switch -c your-branch-name
# edit and test your files
git add .
git commit -m "Describe what you implemented"
git push -u origin your-branch-name
```

Then open a pull request to `main` and ask a teammate to review it. Avoid committing directly to `main` after the initial setup.

## ୨୧ what we'll show in the presentation ♡

- 📈 **Loss vs. iterations:** how gradient descent reduces training error.
- 🌸 **Learning-rate comparison:** how the step size affects convergence.
- 🎙️ **Speaker predictions:** test on unseen recordings.
- 🧮 **Confusion matrix:** see where A and B are mixed up.
- 💭 **Limitations:** similar voices, noisy audio, and overfitting.

> ♡ **A gentle reminder:** a smaller training loss doesn't automatically mean better recognition on new speakers or recordings. We want to understand *why* our numerical method works — and where it doesn't.

<div align="center">

---

**₊˚⊹♡ made with teamwork, tiny gradients & pink dreams ♡⊹˚₊**

*Numerical Methods · November 2026*

</div>
