# Regina: Linear Regression with Gradient Descent

Files: `src/model.py`, `src/gradient_descent.py`, `experiments/run_regina.py`, `tests/test_regina_model.py`.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m experiments.run_regina
```

## 1. The problem

Every 30 ms frame of audio becomes three numbers, in the order from `src/config.py`:

```
x = [pitch_hz, zcr, rms]
```

The label is `y = 0` for speaker A and `y = 1` for speaker B. We want a formula that
turns `x` into a score close to 0 for A and close to 1 for B.

## 2. Data preparation (in `experiments/run_regina.py`)

1. Read Aurora's `output/audio_features.csv`.
2. Drop silent frames (`rms < 0.01`). Silence looks the same for both speakers.
3. Time-based split per recording: the first 70% of the frames are training data, the
   rest is test data. The 20 frames (0.3 s) right after the cut are skipped, because
   frames overlap by 50% and neighbouring frames share audio.
4. Standardize each feature: `x_scaled = (x - mean) / std`, with `mean` and `std` taken
   from the training frames only.

Why standardize: pitch is in the hundreds while ZCR is around 0.02. Without scaling,
one learning rate cannot suit both weights. It is too big for one or too small for the other.

## 3. The model

```
y_pred = w1*x1 + w2*x2 + w3*x3 + b        (code: X @ weights + bias)
label  = B if y_pred >= 0.5, otherwise A
```

`y_pred` is a score, not a probability. It can be below 0 or above 1.

## 4. The loss

```
error_i = y_pred_i - y_i
J(w, b) = (1 / 2N) * sum(error_i^2)
```

Squaring makes every error positive and punishes big errors more. The `1/2` is only
there so the 2 from the derivative cancels.

## 5. Deriving the gradients

`J` depends on `w_j` through every `y_pred_i`, and `d(y_pred_i)/d(w_j) = x_ij`.
With the chain rule:

```
dJ/dw_j = (1 / 2N) * sum( 2 * error_i * x_ij ) = (1 / N) * sum( error_i * x_ij )
dJ/db   = (1 / 2N) * sum( 2 * error_i * 1    ) = (1 / N) * sum( error_i )
```

For all weights at once, `dw = X.T @ errors / N` and `db = sum(errors) / N`.
The test `test_gradient_matches_finite_difference` checks these formulas against the
numerical derivative `(J(w + h) - J(w - h)) / 2h`.

## 6. Gradient Descent

The gradient points uphill, so we step the other way:

```
w = w - lr * dw
b = b - lr * db
```

The loop in `gradient_descent()` starts at `w = 0, b = 0` and repeats:
predict, compute loss, compute gradients, update. It stops when the loss changes by
less than `TOLERANCE` between two iterations, or after `MAX_EPOCHS`.

## 7. One iteration by hand

Tiny dataset with 3 samples, 2 features, `lr = 0.1`:

| i | x1 | x2 | y |
|---|----|----|---|
| 1 | 1  | 2  | 0 |
| 2 | 2  | 0  | 1 |
| 3 | 3  | 1  | 1 |

Start: `w = [0, 0]`, `b = 0`.

**Predict.** Every `y_pred = 0*x1 + 0*x2 + 0 = 0`.

**Errors.** `error = y_pred - y = [0 - 0, 0 - 1, 0 - 1] = [0, -1, -1]`

**Loss.** `J = (0^2 + 1^2 + 1^2) / (2 * 3) = 2 / 6 = 0.3333`

**Gradients.**

```
dw1 = (0*1 + (-1)*2 + (-1)*3) / 3 = -5/3 = -1.6667
dw2 = (0*2 + (-1)*0 + (-1)*1) / 3 = -1/3 = -0.3333
db  = (0 + (-1) + (-1)) / 3       = -2/3 = -0.6667
```

**Update.**

```
w1 = 0 - 0.1 * (-1.6667) = 0.1667
w2 = 0 - 0.1 * (-0.3333) = 0.0333
b  = 0 - 0.1 * (-0.6667) = 0.0667
```

**Did it help?** New predictions:

```
y_pred1 = 0.1667*1 + 0.0333*2 + 0.0667 = 0.3
y_pred2 = 0.1667*2 + 0.0333*0 + 0.0667 = 0.4
y_pred3 = 0.1667*3 + 0.0333*1 + 0.0667 = 0.6
J = (0.3^2 + 0.6^2 + 0.4^2) / 6 = 0.61 / 6 = 0.1017
```

The loss dropped from 0.3333 to 0.1017 after one step. `TestOneIterationByHand` in
`tests/test_regina_model.py` runs `gradient_descent(..., max_epochs=1)` on this dataset
and checks every number above.

## 8. Results on the real features

Output of `python -m experiments.run_regina` (learning rate 0.05, tolerance `1e-10`):

- Training frames: 1133 (A 654, B 479). Test frames: 316 (A 189, B 127).
- Gradient Descent stopped after 208 iterations. Loss went from 0.2114 to 0.0451.
- Final parameters (on standardized features):

| Parameter | Gradient Descent | `np.linalg.lstsq` (check only) |
|-----------|------------------|--------------------------------|
| w_pitch   | 0.34641          | 0.34644                        |
| w_zcr     | -0.00653         | -0.00656                       |
| w_rms     | 0.13228          | 0.13224                        |
| b         | 0.42276          | 0.42277                        |

GD reaches the same minimum as the exact least-squares solution (largest difference
4.1e-05). This shows the loop is correct. Training itself only uses Gradient Descent.

Reading the weights: pitch has the biggest weight, so a higher pitch pushes the score
toward 1 (speaker B, the higher voice). RMS helps a little. ZCR is almost 0, so it barely
matters for these two recordings.

The bias equals the share of B frames in training (479 / 1133 = 0.4228). At the minimum
`db = 0`, so the average prediction equals the average label. After standardization every
feature has mean 0, so the average prediction is just `b`.

## 9. Loss graph and learning rates

`figures/regina_loss.png`: the loss drops fast in the first 30 iterations, where the
gradients are large, then flattens. Near the minimum the gradient is close to 0, so the
steps become tiny. The loop stops once the change per iteration is below `1e-10`.
The curve flattens at 0.045, not 0, because a straight line cannot fit every frame.

`figures/regina_learning_rates.png`, 100 iterations each:

| lr   | Loss after 100 iterations | Behaviour |
|------|---------------------------|-----------|
| 0.01 | 0.0656                    | Too slow, still going down |
| 0.1  | 0.0451                    | Within 1% of the minimum after about 30 iterations |
| 1.6  | 1.56e+06                  | Diverges: each step overshoots and lands further away |

With `lr = 1.6`, the loss first drops for a few iterations, then grows every step. The step
is so big that it jumps over the valley to a higher point on the other side.

Optional detail: for this loss, Gradient Descent is stable only when `lr` is below
`2 / (largest eigenvalue of X^T X / N)`, with a column of ones for the bias. For our
training data that limit is about 1.53, which is why 1.6 diverges.

## 10. Diagnostic accuracy and limitations

- Train accuracy 91.26%, test accuracy 84.49% (`evaluate.py`: 267 of 316 test frames correct).
- These numbers do not show generalization. There is one recording per speaker, so the
  test frames come from the same recording, microphone and room as the training frames.
  A new recording of the same people could score very differently.
- The split is by time with a 0.3 s gap, so train and test frames do not share audio.
  They are still from the same speaking session.
- Each frame is classified on its own. Almost all 49 wrong test frames have a measured
  pitch in the other speaker's range (A frames at 265 to 400 Hz, B frames at 79 to 236 Hz).
  We did not check whether those are pitch-tracking errors or real changes in the voice.
- The model separates these two voices mostly by pitch, so it depends on the two
  speakers having clearly different pitch.

## 11. Difference from `speaker_recognition.py`

That script uses logistic regression: `p = sigmoid(Xw + b)` with five features and
cross-entropy loss. Our version has no sigmoid and uses half-MSE. The gradient formula
`(prediction - y) * x / N` looks the same in both, but the prediction inside it is
different: a sigmoid output there, a straight-line score here. We keep the linear model
because the loss is a simple quadratic, which we can derive by hand and check against the
exact least-squares answer.
