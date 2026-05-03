"""
ECG_Classification_Python. py

Final - year research project - University of Reading, 2025/26
Author: Arham Ali (Student ID: 31022107)
Module: BI3RP3 Research Project

This is the main Python pipeline for classifying ECG arrhythmias from the
MIT - BIH Arrhythmia Database. It walks through the full process: loading the
records via the wfdb library, preprocessing the signals with a Butterworth
bandpass filter, extracting features in the frequency domain (FFT) and
time - frequency domain (Discrete Wavelet Transform), reducing dimensionality
with PCA and finally training Support Vector Machine classifiers using both
linear and RBF kernels.

The dissertation evaluates ten different feature - classifier combinations
under 5-fold stratified cross - validation. The four arrhythmia classes are:
 Normal sinus rhythm (N)
 Premature ventricular contractions (V / PVC)
 Left bundle branch block (L / LBBB)
 Right bundle branch block (R / RBBB)

Setup (run once in a terminal before first use):
pip install wfdb numpy scipy scikit - learn PyWavelets matplotlib seaborn pandas

How to run the script:
python ECG_Classification_Python. py

Note: the wfdb library can either download the MIT - BIH records directly from
PhysioNet or read them from a local folder. The USE_LOCAL flag below
controls which mode is used. I tend to keep it set to True so I'm not
hitting PhysioNet servers every time I run the pipeline.
"""

# Standard scientific Python stack - these are the workhorses for almost
# everything below. NumPy handles all the array maths, SciPy gives us the
# Butterworth filter and the FFT, PyWavelets handles the wavelet decomposition,
# and scikit - learn handles all the machine learning bits.
Import numpy as np
import matplotlib. pyplot as plt
import seaborn as sns
import pandas as pd
from scipy. signal import butter, filtfilt
from scipy. fft import fft
import pywt
from sklearn. decomposition import PCA
from sklearn. svm import SVC
from sklearn. model_selection import (
 StratifiedShuffleSplit, GridSearchCV, cross_val_score,
 StratifiedKFold
)
from sklearn. preprocessing import StandardScaler
from sklearn. metrics import (
 confusion_matrix, classification_report, accuracy_score
)
import wfdb
import warnings
import os
import time

# Suppress sklearn convergence warnings - these get noisy during GridSearchCV
# when some parameter combinations don't converge perfectly. They don't affect
# the final results, just clutter the terminal output.
Warnings. filterwarnings('ignore')

# =============================================================================
# USER SETTINGS - edit these before running
# =============================================================================

# Two modes for loading the data:
# USE_LOCAL = True → read records from a folder on this machine (faster)
# USE_LOCAL = False → download records from PhysioNet over the network
# Local mode is much quicker once you've downloaded the database once.
USE_LOCAL = True
LOCAL_PATH = r'C: \Users\arham\OneDrive - University of Reading\ECG Project\ECG_Project\data\mit - bih - arrhythmia - database-1.0.0'

# Sampling parameters - these come from the MIT - BIH database documentation.
# All 48 records were sampled at 360 Hz with 11-bit resolution, so the
# Nyquist frequency is 180 Hz and our usable bandwidth tops out around 40 Hz
# for ECG morphology purposes.
Fs = 360 # MIT - BIH sampling frequency in Hz
WIN_BEFORE = 90 # samples taken before each R - peak
WIN_AFTER = 109 # samples taken after each R - peak
BEAT_LEN = WIN_BEFORE + WIN_AFTER + 1 # total beat length = 200 samples

# Class balancing - taking the same number of beats per class so that the
# classifier doesn't get biased towards the most common class. 500 beats per
# class gives 2,000 beats total, which is enough for stable cross - validation.
MAX_BEATS_PER_CLASS = 500
RANDOM_STATE = 42 # fixed seed for reproducibility

# Records selected per class - chosen to give a good mix of patients while
# keeping the records balanced. PVC and Normal each draw from 5 records,
# LBBB from 3, RBBB from 4. These are the same records used in the MATLAB
# pipeline so cross - platform results can be compared directly.
CLASS_CONFIG = {
 'Normal': {'symbol': 'N', 'records': [100, 103, 112, 115, 122]},
 'PVC': {'symbol': 'V', 'records': [106, 119, 200, 208, 233]},
 'LBBB': {'symbol': 'L', 'records': [109, 111, 214]},
 'RBBB': {'symbol': 'R', 'records': [118, 124, 212, 231]},
}
CLASS_NAMES = list(CLASS_CONFIG. keys())

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_record(rec_num):
 """
 Load one MIT - BIH record using the wfdb library.
 Returns both the signal and the annotation file. The annotation contains
 cardiologist - verified beat labels at each R - peak - these are our ground
 truth. We use them directly than running our own QR detector,
 since I tried Pan - Tompkins early in the project and it wasn't reliable
 enough on the noisier records.
 """
 if USE_LOCAL:
 try:
 record = wfdb. rdrecord(os. path. join(LOCAL_PATH, str(rec_num)))
 annotation = wfdb. rdann(os. path. join(LOCAL_PATH, str(rec_num)), 'atr')
 except FileNotFoundError:
 print(f"Error: Record {rec_num} not found in {LOCAL_PATH}. ")
 return None, None
 else:
 try:
 record = wfdb. rdrecord(str(rec_num), pn_dir='mit - bih - arrhythmia - database-1.0.0')
 annotation = wfdb. rdann(str(rec_num), 'atr', pn_dir='mit - bih - arrhythmia - database-1.0.0')
 except Exception as e:
 print(f"Error loading record {rec_num} from PhysioNet: {e}")
 return None, None

 # Check if we loaded signal successfully
 if record. p_signal is None:
 print(f"Warning: No signal found for record {rec_num}. Skipping. ")
 return None, None

 # MIT - BIH records typically have multiple signals, we only use the first one
 # which corresponds to lead II, according to database documentation.
 # Shape is (num_samples, num_signals), we want (num_samples, ).
 Signal = record. p_signal[: , 0]

 return signal, annotation


def extract_beats(record, annotation, target_symbol, win_before, win_after):
    """
    Pull out the beats matching a particular annotation symbol.
 
    Each beat is a fixed-length window centred on the R-peak — 90 samples
    before and 109 samples after, giving 200 samples total (about 555 ms at
    360 Hz). Beats too close to the start or end of the recording are
    skipped because we can't extract a full window for them.
 
    A simple quality check rejects any beat that's either too flat
    (std < 0.01 mV → probably a flat-line artefact) or too tall
    (peak > 10 mV → probably an electrode pop or saturation artefact).
    """
    ecg = record.p_signal[:, 0]   # take channel 1 (usually MLII lead)
 
    beats = []
    for i, symbol in enumerate(annotation.symbol):
        if symbol == target_symbol:
            rpeak = annotation.sample[i]
            start = rpeak - win_before
            end = rpeak + win_after
 
            # Skip beats too close to recording boundaries
            if start < 0 or end >= len(ecg):
                continue
 
            segment = ecg[start:end + 1]   # +1 makes the slice inclusive
 
            # Quality control — reject obvious artefacts
            if np.std(segment) < 0.01 or np.max(np.abs(segment)) > 10:
                continue
 
            beats.append(segment)
 
    # Return as a numpy array, or an empty array if nothing was extracted
    return np.array(beats) if beats else np.empty((0, win_before + win_after + 1))
 
 
def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=360, order=3):
    """
    Apply a third-order Butterworth bandpass filter using zero-phase filtering.
 
    The 0.5 Hz lower cutoff removes baseline wander caused by patient breathing
    and electrode movement, and the 40 Hz upper cutoff removes high-frequency
    noise from muscle artefacts and powerline interference. Using filtfilt
    rather than lfilter means the filter is applied forwards and then backwards,
    which cancels out the phase delay that would otherwise distort the QRS
    morphology.
    """
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)
 
 
def extract_fft_features(beats, n_coeffs=50):
    """
    Compute the FFT magnitude spectrum and keep the first 50 coefficients.
 
    Why 50? Earlier in the project I tried using just the dominant frequency
    plus one secondary peak — that gave terrible classification because two
    features simply isn't enough to separate four classes. Going up to 50
    coefficients captures both the gross spectral shape and finer harmonic
    detail without exploding the dimensionality.
    """
    features = np.zeros((len(beats), n_coeffs))
    for i, beat in enumerate(beats):
        spectrum = np.abs(fft(beat))    # magnitude spectrum (we discard phase)
        features[i, :] = spectrum[:n_coeffs]
    return features
 
 
def extract_wavelet_features(beats, wavelet='db4', level=4):
    """
    Decompose each beat with a 4-level Daubechies-4 wavelet and pull out four
    statistics per sub-band.
 
    Why db4? It's compactly supported and its waveform shape happens to look
    quite similar to a QRS complex, which is one reason it's so commonly used
    in ECG analysis. At level 4, the decomposition gives us five sub-bands:
        cA4 = approximation (low-frequency content)
        cD4, cD3, cD2, cD1 = detail coefficients (progressively higher frequencies)
 
    For each sub-band I compute mean, standard deviation, energy, and Shannon
    entropy → 4 statistics × 5 sub-bands = 20 features per beat.
 
    Energy captures how much signal power is in each frequency band, while
    entropy captures how 'spread out' or 'concentrated' that energy is, which
    helps distinguish smoothly varying Normal beats from chaotic PVCs.
    """
    n_features = (level + 1) * 4
    features = np.zeros((len(beats), n_features))
 
    for i, beat in enumerate(beats):
        coeffs = pywt.wavedec(beat, wavelet, level=level)
        # coeffs[0] is the approximation (cA4), then coeffs[1..4] are details
 
        feat_idx = 0
        for j, c in enumerate(coeffs):
            c = np.array(c, dtype=float)
            energy = np.sum(c ** 2)
 
            # Shannon entropy on normalised squared coefficients.
            # The 1e-12 prevents log(0) issues when a sub-band happens to
            # contain a zero coefficient.
            c_sq = c ** 2
            c_sq_norm = c_sq / (np.sum(c_sq) + 1e-12)
            entropy = -np.sum(c_sq_norm * np.log2(c_sq_norm + 1e-12))
 
            features[i, feat_idx]     = np.mean(c)
            features[i, feat_idx + 1] = np.std(c)
            features[i, feat_idx + 2] = energy
            features[i, feat_idx + 3] = entropy
            feat_idx += 4
 
    return features
 
 
def compute_metrics(y_true, y_pred, class_names):
    """
    Compute per-class sensitivity, specificity, positive predictivity, and
    overall accuracy from a confusion matrix.
 
    These are the standard biomedical classification metrics:
        Sensitivity (SE) = TP / (TP + FN)  — how good are we at finding the class?
        Specificity (SP) = TN / (TN + FP)  — how good are we at rejecting non-members?
        Positive Predictivity (PP) = TP / (TP + FP) — when we say it's class X, how often is it really class X?
 
    Sensitivity in particular is what cardiologists care about — missing a PVC
    is a much worse error than flagging a Normal beat for review.
    """
    cm = confusion_matrix(y_true, y_pred)
    n_classes = len(class_names)
 
    metrics = {}
    for k in range(n_classes):
        tp = cm[k, k]
        fn = np.sum(cm[k, :]) - tp
        fp = np.sum(cm[:, k]) - tp
        tn = np.sum(cm) - tp - fn - fp
 
        # Adding 1e-12 prevents division-by-zero if any class has zero predictions
        se = tp / (tp + fn + 1e-12) * 100
        sp = tn / (tn + fp + 1e-12) * 100
        pp = tp / (tp + fp + 1e-12) * 100
 
        metrics[class_names[k]] = {'SE': se, 'SP': sp, 'PP': pp}
 
    metrics['Overall'] = {
        'Accuracy': accuracy_score(y_true, y_pred) * 100,
        'Avg SE': np.mean([m['SE'] for m in metrics.values() if 'SE' in m]),
        'Avg SP': np.mean([m['SP'] for m in metrics.values() if 'SP' in m]),
        'Avg PP': np.mean([m['PP'] for m in metrics.values() if 'PP' in m]),
    }
    return metrics, cm
 
 
# =============================================================================
#  STEP 1 — Load all the records and extract beats for each class
# =============================================================================
print("=" * 60)
print("STEP 1: Loading MIT-BIH records and extracting beats")
print("=" * 60)
 
all_beats = []
all_labels = []
 
for class_idx, (class_name, config) in enumerate(CLASS_CONFIG.items()):
    class_beats = []
 
    # Loop through every record assigned to this class and pull out beats
    for rec_num in config['records']:
        try:
            record, annotation = load_record(rec_num)
            beats = extract_beats(
                record, annotation, config['symbol'],
                WIN_BEFORE, WIN_AFTER
            )
            class_beats.append(beats)
            print(f"  Record {rec_num}: {len(beats)} {class_name} beats")
        except Exception as e:
            # If a record fails to load, log it but keep going — better to have
            # partial results than crash the whole pipeline
            print(f"  Record {rec_num}: FAILED — {e}")
 
    if class_beats:
        class_beats = np.vstack(class_beats)
    else:
        print(f"  WARNING: No beats for class {class_name}")
        continue
 
    # Random subsampling to balance the classes — pick MAX_BEATS_PER_CLASS at
    # random so that no single record dominates the dataset
    n_avail = len(class_beats)
    n_use = min(n_avail, MAX_BEATS_PER_CLASS)
    rng = np.random.RandomState(RANDOM_STATE)
    sel_idx = rng.permutation(n_avail)[:n_use]
    class_beats = class_beats[sel_idx]
 
    all_beats.append(class_beats)
    all_labels.extend([class_idx] * n_use)
    print(f"  → {class_name}: using {n_use}/{n_avail} beats\n")
 
# Stack everything into one big array
all_beats = np.vstack(all_beats)
all_labels = np.array(all_labels)
print(f"Total: {len(all_labels)} beats across {len(CLASS_NAMES)} classes\n")
 
 
# =============================================================================
#  STEP 2 — Preprocess the beats: bandpass filter then normalise
# =============================================================================
print("=" * 60)
print("STEP 2: Bandpass filtering and normalisation")
print("=" * 60)
 
# Apply the Butterworth filter to every single beat
beats_filtered = np.zeros_like(all_beats)
for i in range(len(all_beats)):
    beats_filtered[i] = bandpass_filter(all_beats[i], 0.5, 40.0, Fs, 3)
 
# Z-score normalisation per beat — subtract the mean and divide by the standard
# deviation. This puts every beat on the same amplitude scale, which matters
# because the MIT-BIH records were collected from different patients with
# different ECG amplitudes. Without this step, the classifier could end up
# learning patient-specific amplitude characteristics rather than genuine
# arrhythmia morphology.
for i in range(len(beats_filtered)):
    mu = np.mean(beats_filtered[i])
    sigma = np.std(beats_filtered[i])
    if sigma > 0:
        beats_filtered[i] = (beats_filtered[i] - mu) / sigma
 
print(f"  Filtered and normalised {len(beats_filtered)} beats\n")
 
 
# =============================================================================
#  STEP 3 — Feature extraction (FFT, Wavelet, then PCA on each)
# =============================================================================
print("=" * 60)
print("STEP 3: Feature extraction (FFT, Wavelet, PCA)")
print("=" * 60)
 
# Frequency-domain features — first 50 FFT magnitude coefficients per beat
feat_fft = extract_fft_features(beats_filtered, n_coeffs=50)
print(f"  FFT features: {feat_fft.shape[1]} per beat")
 
# Time-frequency features — 20 wavelet statistics per beat (db4, level 4)
feat_wav = extract_wavelet_features(beats_filtered, 'db4', 4)
print(f"  Wavelet features: {feat_wav.shape[1]} per beat")
 
# PCA needs zero-mean unit-variance inputs to work properly, otherwise
# features with large numerical ranges dominate the principal components
scaler_fft = StandardScaler()
feat_fft_scaled = scaler_fft.fit_transform(feat_fft)
 
scaler_wav = StandardScaler()
feat_wav_scaled = scaler_wav.fit_transform(feat_wav)
 
# PCA reduction — 12 components for each feature set. I picked 12 after
# inspecting the variance-explained curve; beyond that, additional components
# add noise more than signal.
n_pcs = 12
pca_fft = PCA(n_components=n_pcs, random_state=RANDOM_STATE)
feat_fft_pca = pca_fft.fit_transform(feat_fft_scaled)
print(f"  FFT PCA: {n_pcs} PCs (variance explained: "
      f"{100*np.sum(pca_fft.explained_variance_ratio_):.1f}%)")
 
pca_wav = PCA(n_components=n_pcs, random_state=RANDOM_STATE)
feat_wav_pca = pca_wav.fit_transform(feat_wav_scaled)
print(f"  Wavelet PCA: {n_pcs} PCs (variance explained: "
      f"{100*np.sum(pca_wav.explained_variance_ratio_):.1f}%)")
 
# Combined feature set — concatenates FFT and wavelet features (70 features)
# then projects down to 12 PCs. This tests whether combining frequency-domain
# and time-frequency information adds anything over each domain alone.
feat_combined = np.hstack([feat_fft_scaled, feat_wav_scaled])
pca_combined = PCA(n_components=n_pcs, random_state=RANDOM_STATE)
feat_combined_pca = pca_combined.fit_transform(feat_combined)
print(f"  Combined PCA: {n_pcs} PCs (variance explained: "
      f"{100*np.sum(pca_combined.explained_variance_ratio_):.1f}%)\n")
 
 
# =============================================================================
#  STEP 4 — Train and evaluate all 10 feature/classifier combinations
# =============================================================================
print("=" * 60)
print("STEP 4: Classification (GridSearchCV + 5-fold CV)")
print("=" * 60)
 
# 50/50 stratified train-test split. Stratified ensures each class is
# represented proportionally in both training and test sets.
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5,
                                  random_state=RANDOM_STATE)
train_idx, test_idx = next(splitter.split(all_beats, all_labels))
 
# Five different feature representations × two kernels = 10 total combinations
feature_sets = {
    'FFT (50 coeffs)':         feat_fft_scaled,
    'FFT + PCA (12 PCs)':      feat_fft_pca,
    'Wavelet (20 feats)':      feat_wav_scaled,
    'Wavelet + PCA (12 PCs)':  feat_wav_pca,
    'Combined + PCA (12 PCs)': feat_combined_pca,
}
 
# Hyperparameter grids for the SVM. C controls the trade-off between margin
# size and training accuracy; gamma controls how 'tight' the RBF kernel
# wraps around each training point. GridSearchCV tries every combination.
param_grid_linear = {'C': [0.01, 0.1, 1, 10, 100]}
param_grid_rbf = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
}
 
# Containers for tracking results across all combinations
all_results = []
best_model = None
best_acc = 0
best_cm = None
baseline_cm = None
baseline_acc = 0
 
# Outer loop iterates through each feature set; inner loop tries both kernels
for feat_name, feat_data in feature_sets.items():
    X_train = feat_data[train_idx]
    X_test = feat_data[test_idx]
    y_train = all_labels[train_idx]
    y_test = all_labels[test_idx]
 
    for kernel_name, param_grid in [('Linear SVM', param_grid_linear),
                                     ('RBF SVM', param_grid_rbf)]:
 
        t0 = time.time()
 
        # Initialise the SVM with the appropriate kernel
        if kernel_name == 'Linear SVM':
            base_svm = SVC(kernel='linear', random_state=RANDOM_STATE)
        else:
            base_svm = SVC(kernel='rbf', random_state=RANDOM_STATE)
 
        # GridSearchCV finds the best hyperparameters using inner 5-fold CV
        # on the training data. n_jobs=-1 parallelises across all CPU cores
        # to speed things up — the RBF grid has 20 combinations × 5 folds = 100
        # fits per feature set, so parallelisation makes a noticeable difference.
        grid = GridSearchCV(
            base_svm, param_grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True,
                               random_state=RANDOM_STATE),
            scoring='accuracy',
            n_jobs=-1,
            refit=True
        )
        grid.fit(X_train, y_train)
 
        # Hold-out test set evaluation for confusion matrix generation
        y_pred = grid.predict(X_test)
        acc = accuracy_score(y_test, y_pred) * 100
 
        # Cross-validation on the full dataset for a more robust performance
        # estimate. The single test-set number can be misleading on a small
        # dataset, so the CV mean ± std is the headline figure I report.
        cv_scores = cross_val_score(
            grid.best_estimator_, feat_data, all_labels,
            cv=StratifiedKFold(n_splits=5, shuffle=True,
                               random_state=RANDOM_STATE),
            scoring='accuracy'
        )
 
        metrics, cm = compute_metrics(y_test, y_pred, CLASS_NAMES)
        elapsed = time.time() - t0
 
        # Save everything for the summary table
        result = {
            'Features': feat_name,
            'Classifier': kernel_name,
            'Test Accuracy (%)': round(acc, 2),
            'CV Mean (%)': round(cv_scores.mean() * 100, 2),
            'CV Std (%)': round(cv_scores.std() * 100, 2),
            'Best Params': str(grid.best_params_),
            'Avg SE (%)': round(metrics['Overall']['Avg SE'], 2),
            'Avg SP (%)': round(metrics['Overall']['Avg SP'], 2),
            'Avg PP (%)': round(metrics['Overall']['Avg PP'], 2),
            'Time (s)': round(elapsed, 1),
        }
        all_results.append(result)
 
        print(f"\n  {feat_name} + {kernel_name}")
        print(f"    Test Accuracy: {acc:.2f}%")
        print(f"    5-Fold CV:     {cv_scores.mean()*100:.2f}% "
              f"(±{cv_scores.std()*100:.2f}%)")
        print(f"    Best params:   {grid.best_params_}")
        print(f"    SE: {metrics['Overall']['Avg SE']:.1f}%  "
              f"SP: {metrics['Overall']['Avg SP']:.1f}%  "
              f"PP: {metrics['Overall']['Avg PP']:.1f}%")
 
        # Track the FFT + Linear baseline for comparison plots
        if feat_name == 'FFT (50 coeffs)' and kernel_name == 'Linear SVM':
            baseline_cm = cm
            baseline_acc = acc
        # Track the overall best model so far
        if acc > best_acc:
            best_acc = acc
            best_cm = cm
            best_model = grid.best_estimator_
            best_feat_name = feat_name
            best_clf_name = kernel_name
            best_metrics = metrics
 
print("\n")
 
 
# =============================================================================
#  Results table — print to console and save to CSV
# =============================================================================
print("=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
df_results = pd.DataFrame(all_results)
print(df_results.to_string(index=False))
 
df_results.to_csv('ecg_classification_results.csv', index=False)
print("\nResults saved to ecg_classification_results.csv")
 
 
# =============================================================================
#  STEP 5 — Generate all the figures used in the dissertation
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5: Generating figures")
print("=" * 60)
 
# Class colour scheme — picked to be colourblind-friendly and consistent
# across all figures in the dissertation
colors = ['#1f77b4', '#ff7f0e', '#d4ac0a', '#9467bd']  # blue, orange, gold, purple
 
# ------- Figure 1: Three example beats per class, post-preprocessing -------
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
t_ms = np.arange(BEAT_LEN) / Fs * 1000
for c in range(4):
    ax = axes[c // 2, c % 2]
    idx = np.where(all_labels == c)[0][:3]   # take first 3 beats of this class
    for j in idx:
        ax.plot(t_ms, beats_filtered[j], linewidth=0.8)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Normalised Amplitude')
    ax.set_title(f'{CLASS_NAMES[c]} beats')
    ax.grid(True, alpha=0.3)
fig.suptitle('Representative ECG Beat Morphologies (MIT-BIH)', fontsize=13)
plt.tight_layout()
plt.savefig('Fig1_Sample_Beats.png', dpi=150, bbox_inches='tight')
print("  Saved Fig1_Sample_Beats.png")
 
# ------- Figure 2: Feature space comparison (FFT PCA / raw wavelet / wavelet PCA) -------
# This figure is important for the Discussion — it shows visually why the RBF
# kernel is needed (the class boundaries in feature space are clearly non-linear)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
 
for c in range(4):
    mask = all_labels == c
    axes[0].scatter(feat_fft_pca[mask, 0], feat_fft_pca[mask, 1],
                    s=8, c=colors[c], alpha=0.5, label=CLASS_NAMES[c])
axes[0].set_xlabel('PC1'); axes[0].set_ylabel('PC2')
axes[0].set_title('FFT Features (PCA)')
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
 
for c in range(4):
    mask = all_labels == c
    axes[1].scatter(feat_wav_scaled[mask, 1], feat_wav_scaled[mask, 5],
                    s=8, c=colors[c], alpha=0.5, label=CLASS_NAMES[c])
axes[1].set_xlabel('cA4 Std Dev'); axes[1].set_ylabel('cD1 Std Dev')
axes[1].set_title('Wavelet Features (Raw)')
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
 
for c in range(4):
    mask = all_labels == c
    axes[2].scatter(feat_wav_pca[mask, 0], feat_wav_pca[mask, 1],
                    s=8, c=colors[c], alpha=0.5, label=CLASS_NAMES[c])
axes[2].set_xlabel('PC1'); axes[2].set_ylabel('PC2')
axes[2].set_title('Wavelet Features (PCA)')
axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)
 
fig.suptitle('Feature Space Comparison: FFT vs Wavelet', fontsize=13)
plt.tight_layout()
plt.savefig('Fig2_Feature_Space.png', dpi=150, bbox_inches='tight')
print("  Saved Fig2_Feature_Space.png")
 
# ------- Figure 3: Confusion matrices side by side (worst linear vs best RBF) -------
# This is the figure that most clearly demonstrates the kernel selection effect
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 
sns.heatmap(baseline_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
axes[0].set_xlabel('Predicted Class'); axes[0].set_ylabel('True Class')
axes[0].set_title(f'FFT + Linear SVM ({baseline_acc:.1f}%)')
 
sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1])
axes[1].set_xlabel('Predicted Class'); axes[1].set_ylabel('True Class')
axes[1].set_title(f'{best_feat_name} + {best_clf_name} ({best_acc:.1f}%)')
 
fig.suptitle('Confusion Matrix Comparison', fontsize=13)
plt.tight_layout()
plt.savefig('Fig3_Confusion_Matrices.png', dpi=150, bbox_inches='tight')
print("  Saved Fig3_Confusion_Matrices.png")
 
# ------- Figure 4: Bar chart of accuracy across all 10 combinations -------
fig, ax = plt.subplots(figsize=(12, 5))
labels = [f"{r['Features']}\n+ {r['Classifier']}" for r in all_results]
accs = [r['Test Accuracy (%)'] for r in all_results]
cv_means = [r['CV Mean (%)'] for r in all_results]
cv_stds = [r['CV Std (%)'] for r in all_results]
 
x = np.arange(len(labels))
width = 0.35
bars1 = ax.bar(x - width/2, accs, width, label='Test Accuracy', color='#1f77b4')
bars2 = ax.bar(x + width/2, cv_means, width, yerr=cv_stds,
               label='5-Fold CV (±std)', color='#ff7f0e', capsize=3)
ax.set_ylabel('Classification Accuracy (%)')
ax.set_title('Classification Performance: Test vs Cross-Validation')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
ax.legend()
ax.set_ylim(max(0, min(accs + cv_means) - 5), 100)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('Fig4_Accuracy_Comparison.png', dpi=150, bbox_inches='tight')
print("  Saved Fig4_Accuracy_Comparison.png")
 
# ------- Figure 5: Per-class metrics for the best model -------
fig, ax = plt.subplots(figsize=(8, 5))
se_vals = [best_metrics[cn]['SE'] for cn in CLASS_NAMES]
sp_vals = [best_metrics[cn]['SP'] for cn in CLASS_NAMES]
pp_vals = [best_metrics[cn]['PP'] for cn in CLASS_NAMES]
 
x = np.arange(len(CLASS_NAMES))
width = 0.25
ax.bar(x - width, se_vals, width, label='Sensitivity', color='#1f77b4')
ax.bar(x, sp_vals, width, label='Specificity', color='#ff7f0e')
ax.bar(x + width, pp_vals, width, label='Pos. Predictivity', color='#d4ac0a')
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES)
ax.set_ylabel('Performance (%)')
ax.set_title(f'Per-Class Metrics: {best_feat_name} + {best_clf_name}')
ax.legend(loc='lower right')
ax.set_ylim(80, 100)   # zoom in to make small differences visible
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('Fig5_PerClass_Metrics.png', dpi=150, bbox_inches='tight')
print("  Saved Fig5_PerClass_Metrics.png")
 
# ------- Figure 6: PCA variance explained per component -------
# Helps justify why I chose 12 components rather than something else
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(range(1, n_pcs+1),
            pca_fft.explained_variance_ratio_ * 100, color='#1f77b4')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Variance Explained (%)')
axes[0].set_title('FFT PCA — Variance per Component')
axes[0].grid(True, axis='y', alpha=0.3)
 
axes[1].bar(range(1, n_pcs+1),
            pca_wav.explained_variance_ratio_ * 100, color='#ff7f0e')
axes[1].set_xlabel('Principal Component')
axes[1].set_ylabel('Variance Explained (%)')
axes[1].set_title('Wavelet PCA — Variance per Component')
axes[1].grid(True, axis='y', alpha=0.3)
 
plt.tight_layout()
plt.savefig('Fig6_PCA_Variance.png', dpi=150, bbox_inches='tight')
print("  Saved Fig6_PCA_Variance.png")
 
 
# =============================================================================
#  Final summary printout
# =============================================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"\nBest model: {best_feat_name} + {best_clf_name}")
print(f"  Test accuracy:  {best_acc:.2f}%")
print(f"  Avg Sensitivity: {best_metrics['Overall']['Avg SE']:.2f}%")
print(f"  Avg Specificity: {best_metrics['Overall']['Avg SP']:.2f}%")
print(f"  Avg Pos. Pred.:  {best_metrics['Overall']['Avg PP']:.2f}%")
print(f"\nBaseline (FFT + Linear SVM): {baseline_acc:.2f}%")
print(f"Improvement: +{best_acc - baseline_acc:.2f} percentage points")
print(f"\nAll figures saved as PNG files in current directory.")
print(f"Results table saved as ecg_classification_results.csv")
print(f"\nTotal beats analysed: {len(all_labels)}")
print(f"Classes: {CLASS_NAMES}")
print(f"Beats per class: {MAX_BEATS_PER_CLASS}")
 
