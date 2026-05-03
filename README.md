# ECG-SVM-Classification-31022107
Final-year research project — Evaluating Feature Extraction and Kernel Selection for SVM-Based ECG Arrhythmia Classification
# ECG-SVM-Classification-31022107

Final-year research project at the University of Reading evaluating feature
extraction and kernel selection for SVM-based ECG arrhythmia classification
on the MIT-BIH Arrhythmia Database.

## Best result
98.6% ± 0.4% cross-validation accuracy using FFT (50 coefficients) with RBF SVM.

## Repository structure
- `/python/` — Primary Python implementation (Python 3.14, scikit-learn)
- `/matlab/` — MATLAB R2025a validation implementation
- `/data/` — Instructions for obtaining the MIT-BIH database

## Setup
1. Install Python dependencies: `pip install -r python/requirements.txt`
2. Download the MIT-BIH Arrhythmia Database from
   https://physionet.org/content/mitdb/1.0.0/
3. Update the `LOCAL_PATH` variable in `ECG_Classification_Python.py` to
   point to your local copy of the database
4. Run: `python python/ECG_Classification_Python.py`

## Citation
Ali, A. (2026) *Evaluating Feature Extraction and Kernel Selection for
SVM-Based ECG Arrhythmia Classification*. Final-year research project,
University of Reading.

## License
MIT License — see LICENSE file for details.
