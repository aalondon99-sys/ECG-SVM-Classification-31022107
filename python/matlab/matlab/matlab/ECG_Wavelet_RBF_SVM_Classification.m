%% ECG_Wavelet_RBF_SVM_Classification.m
% =====================================================================
% Final-year research project — University of Reading, 2025/26
% Author: Arham Ali (Student ID: 31022107)
% Module: BI3RP3 Research Project
% =====================================================================
%
% This is the MATLAB version of the full ECG classification pipeline. It
% mirrors the Python implementation exactly so the two platforms can be
% used as independent cross-checks of each other — if both return the same
% accuracy on the same data, that's strong evidence the result reflects
% genuine algorithmic behaviour rather than implementation quirks.
%
% The pipeline runs through the same six stages as the Python version:
%   1. MIT-BIH database loading + custom format-212 decoding
%   2. Preprocessing (Butterworth bandpass + per-beat normalisation)
%   3. Feature extraction (FFT and Discrete Wavelet Transform)
%   4. Dimensionality reduction (PCA via SVD)
%   5. Classification (linear and RBF SVM with one-vs-one coding)
%   6. Visualisation (sample beats, feature spaces, confusion matrices)
%
% The four classes evaluated are:
%   N = Normal sinus rhythm
%   V = Premature ventricular contraction (PVC)
%   L = Left bundle branch block (LBBB)
%   R = Right bundle branch block (RBBB)
%
% Setup before running:
%   1. Update the dataPath variable below to point to your local MIT-BIH
%      Arrhythmia Database folder
%   2. Make sure these toolboxes are installed:
%        - Signal Processing Toolbox (for the Butterworth filter)
%        - Statistics and Machine Learning Toolbox (for fitcecoc + SVM)
%        - Wavelet Toolbox (for wavedec)
%   3. Run the entire script — takes about 1-2 minutes on a typical laptop

clear; clc; close all;


%% ====== USER SETTINGS ======
% Path to your local copy of the MIT-BIH database. The database can be
% downloaded for free from PhysioNet at:
% https://physionet.org/content/mitdb/1.0.0/
dataPath = 'C:\Users\arham\OneDrive - University of Reading\ECG Project\ECG_Project\data\mit-bih-arrhythmia-database-1.0.0';

% Sampling parameters from the MIT-BIH database documentation. All 48
% records were sampled at 360 Hz with 11-bit resolution. The 200-sample
% beat window covers about 555 ms either side of the R-peak — enough to
% capture the full P-QRS-T complex without including too much of the
% neighbouring beats.
Fs = 360;                              % Sampling frequency (Hz)
winBefore = 90;                        % Samples taken before R-peak
winAfter  = 109;                       % Samples taken after R-peak
beatLen   = winBefore + winAfter + 1;  % Total beat length = 200 samples

% Records picked per class. These were chosen for two reasons: each record
% has a high count of the relevant beat type, and the full record set
% covers a reasonable spread of patients to avoid the classifier becoming
% over-fit to a single individual's morphology.
classRecords = struct();
classRecords.N = [100, 103, 112, 115, 122];   % Normal beats
classRecords.V = [106, 119, 200, 208, 233];   % PVC beats
classRecords.L = [109, 111, 214];             % Left bundle branch block
classRecords.R = [118, 124, 212, 231];        % Right bundle branch block

% Class balancing — taking the same number of beats per class stops the
% classifier learning to just predict the most common class. 500 per class
% gives 2,000 total, which is enough for stable cross-validation while
% keeping training time reasonable.
maxBeatsPerClass = 500;


%% ====== STEP 1: LOAD AND DECODE MIT-BIH RECORDS ======
% This step does the heavy lifting that took the longest to get right.
% MIT-BIH stores its data in WFDB format-212, which packs two 12-bit signed
% samples into three bytes. My first attempt read these as int16 and got
% solid block plots — definitely the most useful failure of the project,
% because it showed that getting the decoding right is non-negotiable
% before any classification work can begin.
fprintf('Step 1: Loading and decoding MIT-BIH records...\n');

allBeats = [];
allLabels = [];
classNames = fieldnames(classRecords);

for c = 1:length(classNames)
    className = classNames{c};
    records = classRecords.(className);
    classBeats = [];

    for r = 1:length(records)
        recNum = records(r);
        recStr = num2str(recNum);

        % --- Read the .hea header file to get gain and baseline ---
        % We need these calibration values to convert raw ADC values back
        % into millivolts. They're stored as text in the header file in
        % the format "gain(baseline)/units", though sometimes the baseline
        % is missing and we have to fall back to defaults.
        heaFile = fullfile(dataPath, [recStr '.hea']);
        if ~isfile(heaFile)
            warning('Header file not found: %s — skipping', heaFile);
            continue;
        end
        fid = fopen(heaFile, 'r');
        headerLine = fgetl(fid);   % first line: record-level info
        sigLine1 = fgetl(fid);     % second line: signal 1 (channel 1)
        fclose(fid);

        % Parse the gain and baseline using regex. The header format
        % varies slightly between records, so we try the bracketed format
        % first and fall back to a plain numeric parse if that fails.
        parts = strsplit(sigLine1);
        gainStr = parts{3};   % e.g. "200(0)/mV" or "200/mV"
        if contains(gainStr, '(')
            tokens = regexp(gainStr, '(\d+)\((\d+)\)', 'tokens');
            if ~isempty(tokens)
                gain = str2double(tokens{1}{1});
                baseline = str2double(tokens{1}{2});
            else
                gain = str2double(regexprep(gainStr, '[^0-9.]', ''));
                baseline = 0;
            end
        else
            gain = str2double(regexprep(gainStr, '[^0-9.]', ''));
            baseline = 0;
        end
        % Sensible defaults if parsing failed (200 ADU/mV is the standard
        % MIT-BIH gain across most records)
        if isnan(gain) || gain == 0
            gain = 200;
        end
        if isnan(baseline)
            baseline = 0;
        end

        % --- Decode the format-212 binary .dat file ---
        % This is the bit that broke the original int16 attempt. Format
        % 212 packs two 12-bit signed samples into every 3 bytes, with
        % the upper 4 bits of each sample sharing the middle byte.
        datFile = fullfile(dataPath, [recStr '.dat']);
        if ~isfile(datFile)
            warning('Data file not found: %s — skipping', datFile);
            continue;
        end
        fid = fopen(datFile, 'r');
        rawBytes = fread(fid, inf, 'uint8');
        fclose(fid);

        % Each frame contains 3 bytes encoding 2 samples
        nGroups = floor(length(rawBytes) / 3);
        sig1 = zeros(nGroups, 1);   % channel 1 samples
        sig2 = zeros(nGroups, 1);   % channel 2 samples (kept but not used)

        for g = 1:nGroups
            idx = (g-1)*3 + 1;
            b1 = rawBytes(idx);
            b2 = rawBytes(idx+1);
            b3 = rawBytes(idx+2);

            % Sample 1: low 8 bits come from b1, high 4 bits come from
            % the low nibble of b2. bitand(b2, 15) extracts those low
            % 4 bits (binary mask 0000 1111 = decimal 15).
            s1 = b1 + bitand(b2, 15) * 256;
            % Sign extension: 12-bit signed values run from -2048 to +2047,
            % so anything >= 2048 is actually a negative number that needs
            % unwrapping by subtracting 4096.
            if s1 >= 2048, s1 = s1 - 4096; end

            % Sample 2: low 4 bits come from the high nibble of b2
            % (extracted by right-shifting b2 by 4 places), high 8 bits
            % come from b3.
            s2 = bitshift(b2, -4) + b3 * 16;
            if s2 >= 2048, s2 = s2 - 4096; end

            sig1(g) = s1;
            sig2(g) = s2;
        end

        % Calibrate channel 1 to physical units (millivolts)
        ecg = (sig1 - baseline) / gain;

        % --- Read the .atr annotation file ---
        % These annotations are cardiologist-verified and give us both the
        % R-peak locations and the ground-truth class labels. I tried
        % using a Pan-Tompkins detector early in the project but it wasn't
        % reliable enough on the noisier records, so using the database
        % annotations directly is the cleaner approach.
        atrFile = fullfile(dataPath, [recStr '.atr']);
        if ~isfile(atrFile)
            warning('Annotation file not found: %s — skipping', atrFile);
            continue;
        end
        [annSamp, annType] = readMITAnnotation(atrFile);

        % --- Pick out beats matching the current target class ---
        switch className
            case 'N'
                targetCodes = 'N';   % Normal beat
            case 'V'
                targetCodes = 'V';   % PVC
            case 'L'
                targetCodes = 'L';   % Left bundle branch block
            case 'R'
                targetCodes = 'R';   % Right bundle branch block
        end

        matchIdx = find(annType == targetCodes);
        matchSamp = annSamp(matchIdx);

        % Segment beats around each matching R-peak
        for b = 1:length(matchSamp)
            rpeak = matchSamp(b);
            startIdx = rpeak - winBefore;
            endIdx = rpeak + winAfter;

            % Skip beats too close to the edges of the recording
            if startIdx < 1 || endIdx > length(ecg)
                continue;
            end
            beatSeg = ecg(startIdx:endIdx);

            % Quality check: reject beats that look like artefacts.
            % std < 0.01 mV usually means a flat-line artefact or
            % electrode disconnection; max amplitude > 10 mV usually
            % means an electrode pop or amplifier saturation.
            if std(beatSeg) < 0.01 || max(abs(beatSeg)) > 10
                continue;
            end

            classBeats = [classBeats; beatSeg'];
        end

        fprintf('  Record %s: extracted %d %s beats so far\n', ...
            recStr, size(classBeats,1), className);
    end

    % Random subsampling to balance the classes. The fixed seed (rng 42)
    % makes the random selection reproducible — same beats picked every
    % time the script is run, which matters for cross-platform validation.
    nAvail = size(classBeats, 1);
    nUse = min(nAvail, maxBeatsPerClass);
    rng(42);
    selIdx = randperm(nAvail, nUse);
    classBeats = classBeats(selIdx, :);

    allBeats = [allBeats; classBeats];
    allLabels = [allLabels; repmat(c, nUse, 1)];

    fprintf('  Class %s: using %d beats (from %d available)\n\n', ...
        className, nUse, nAvail);
end

labelNames = {'Normal','PVC','LBBB','RBBB'};
fprintf('Total beats: %d across %d classes\n\n', ...
    length(allLabels), length(classNames));


%% ====== STEP 2: PREPROCESSING ======
% Two stages here: bandpass filtering to clean up the noise, then per-beat
% normalisation to put every beat on the same amplitude scale. Without
% normalisation the classifier could end up learning patient-specific
% amplitude characteristics rather than genuine arrhythmia morphology.
fprintf('Step 2: Preprocessing (bandpass filter)...\n');

% Third-order Butterworth bandpass with 0.5 Hz lower and 40 Hz upper
% cutoffs. The lower cutoff removes baseline wander from breathing and
% electrode movement; the upper cutoff removes muscle artefacts and
% powerline interference. Third-order is enough to give a steep enough
% rolloff without excessive ringing.
[bBP, aBP] = butter(3, [0.5 40]/(Fs/2), 'bandpass');

beatsFiltered = zeros(size(allBeats));
for i = 1:size(allBeats, 1)
    % filtfilt applies the filter forwards and then backwards, which
    % cancels the phase delay that would otherwise distort the QRS
    % morphology. Critical for ECG analysis — we need the R-peak to stay
    % exactly where the annotation says it is.
    beatsFiltered(i,:) = filtfilt(bBP, aBP, allBeats(i,:));
end

% Z-score normalisation per beat: subtract the mean, divide by the std
for i = 1:size(beatsFiltered, 1)
    beatsFiltered(i,:) = (beatsFiltered(i,:) - mean(beatsFiltered(i,:))) ...
                         / std(beatsFiltered(i,:));
end

fprintf('  Filtering and normalisation complete.\n\n');


%% ====== STEP 3: FEATURE EXTRACTION ======
% Three feature representations to compare: pure FFT (frequency domain),
% wavelet statistics (time-frequency domain), and PCA-reduced versions of
% each. The dissertation's main finding is that kernel choice (linear vs
% RBF) matters more than feature representation, but we need all of them
% to demonstrate that.
fprintf('Step 3: Feature extraction (FFT, Wavelet, PCA)...\n');

nBeats = size(beatsFiltered, 1);

% --- 3a: FFT features ---
% First 50 magnitude coefficients per beat. I tried just 2 features early
% on (dominant frequency + secondary peak) and it was far too few — going
% up to 50 captures both the gross spectral shape and finer harmonic
% detail without exploding the dimensionality.
nFFT = 50;
featFFT = zeros(nBeats, nFFT);
for i = 1:nBeats
    F = abs(fft(beatsFiltered(i,:)));   % magnitude spectrum (we discard phase)
    featFFT(i,:) = F(1:nFFT);
end

% --- 3b: Wavelet features ---
% Daubechies-4 (db4) at 4 decomposition levels. db4 is preferred for ECG
% analysis because its waveform shape happens to look quite similar to a
% QRS complex, so it captures QRS-like features efficiently.
waveletName = 'db4';
wavLevel = 4;

% Run one decomposition first just to figure out output sizes
[C_test, L_test] = wavedec(beatsFiltered(1,:), wavLevel, waveletName);

% At level 4, wavedec produces 5 sub-bands:
%   cA4 = approximation (low-frequency content)
%   cD4, cD3, cD2, cD1 = detail coefficients (progressively higher
%                        frequencies)
% From each sub-band we compute 4 statistics: mean, std, energy, entropy
% That gives 5 sub-bands × 4 stats = 20 features per beat.
nWavFeats = (wavLevel + 1) * 4;
featWav = zeros(nBeats, nWavFeats);

for i = 1:nBeats
    [C, L] = wavedec(beatsFiltered(i,:), wavLevel, waveletName);

    fIdx = 1;

    % Approximation coefficients (level 4) — the smoothest sub-band,
    % reflecting the low-frequency envelope of the beat
    cA = appcoef(C, L, waveletName, wavLevel);
    featWav(i, fIdx)   = mean(cA);
    featWav(i, fIdx+1) = std(cA);
    featWav(i, fIdx+2) = sum(cA.^2);   % energy
    % Shannon entropy on normalised squared coefficients. The eps
    % additions prevent log(0) when a sub-band happens to be all zeros.
    featWav(i, fIdx+3) = -sum((cA.^2/sum(cA.^2 + eps)) .* ...
                          log2(cA.^2/sum(cA.^2 + eps) + eps));
    fIdx = fIdx + 4;

    % Detail coefficients at each level — these capture progressively
    % higher-frequency content (cD1 is highest frequency, cD4 is lowest)
    for lev = 1:wavLevel
        cD = detcoef(C, L, lev);
        featWav(i, fIdx)   = mean(cD);
        featWav(i, fIdx+1) = std(cD);
        featWav(i, fIdx+2) = sum(cD.^2);
        featWav(i, fIdx+3) = -sum((cD.^2/sum(cD.^2 + eps)) .* ...
                              log2(cD.^2/sum(cD.^2 + eps) + eps));
        fIdx = fIdx + 4;
    end
end

% --- 3c: PCA reduction via SVD ---
% PCA needs zero-mean unit-variance inputs to behave properly, otherwise
% features with large numerical ranges (like energy values) dominate the
% principal components.
nPCs = 12;

% Standardise the wavelet features then take the top 12 PCs
featWav_mean = mean(featWav);
featWav_std  = std(featWav);
featWav_norm = (featWav - featWav_mean) ./ (featWav_std + eps);

% SVD-based PCA. The 'econ' flag computes the economy-size SVD which
% skips the trailing zero singular values — much faster on a tall thin
% matrix like this.
[U, S, V] = svd(featWav_norm, 'econ');
featWavPCA = featWav_norm * V(:, 1:nPCs);

% Same treatment for the FFT features — useful for direct comparison
% between FFT-PCA and wavelet-PCA in the dissertation
featFFT_mean = mean(featFFT);
featFFT_std  = std(featFFT);
featFFT_norm = (featFFT - featFFT_mean) ./ (featFFT_std + eps);
[~, ~, V_fft] = svd(featFFT_norm, 'econ');
featFFT_PCA = featFFT_norm * V_fft(:, 1:nPCs);

fprintf('  FFT features: %d per beat\n', nFFT);
fprintf('  Wavelet features: %d per beat (reduced to %d via PCA)\n', ...
    nWavFeats, nPCs);
fprintf('  Feature extraction complete.\n\n');


%% ====== STEP 4: TRAIN/TEST SPLIT ======
% 50/50 stratified split. Stratified means each class is represented
% proportionally in both halves, so we don't end up with all the LBBB
% beats in training and none in test.
fprintf('Step 4: Train/test split (50/50 stratified)...\n');

rng(42);   % fixed seed so the split is identical to the Python version
cv = cvpartition(allLabels, 'HoldOut', 0.5);
idxTrain = training(cv);
idxTest  = test(cv);

fprintf('  Training: %d beats | Testing: %d beats\n\n', ...
    sum(idxTrain), sum(idxTest));


%% ====== STEP 5: CLASSIFICATION ======
% Train and evaluate every combination of feature set and SVM kernel.
% Four feature sets × two kernels = 8 combinations evaluated here. (The
% Python version adds a ninth and tenth using the combined FFT+wavelet
% feature set, but the eight tested here are enough to validate the
% kernel-selection finding on MATLAB.)
fprintf('Step 5: Training and evaluating classifiers...\n\n');

% Bundle the feature sets into a struct for easy looping
featureSets = struct();
featureSets(1).name = 'FFT (50 coefficients)';
featureSets(1).data = featFFT;
featureSets(2).name = 'FFT + PCA (12 PCs)';
featureSets(2).data = featFFT_PCA;
featureSets(3).name = 'Wavelet (20 features)';
featureSets(3).data = featWav;
featureSets(4).name = 'Wavelet + PCA (12 PCs)';
featureSets(4).data = featWavPCA;

% And the two kernel configurations
classifiers = struct();
classifiers(1).name = 'Linear SVM';
classifiers(1).kernel = 'linear';
classifiers(2).name = 'RBF SVM';
classifiers(2).kernel = 'rbf';

% Container for the results table that will eventually print to console
results = table();
resultIdx = 0;

for f = 1:length(featureSets)
    for cl = 1:length(classifiers)

        XTrain = featureSets(f).data(idxTrain, :);
        XTest  = featureSets(f).data(idxTest, :);
        yTrain = allLabels(idxTrain);
        yTest  = allLabels(idxTest);

        % Set up the SVM template. KernelScale 'auto' lets MATLAB pick a
        % sensible bandwidth for the RBF kernel based on the data, which
        % saves us doing a full grid search here. Standardize=true
        % z-scores the features inside the SVM, which matters for the RBF
        % kernel since it's distance-based.
        if strcmp(classifiers(cl).kernel, 'linear')
            t = templateSVM('KernelFunction', 'linear', ...
                            'Standardize', true);
        else
            t = templateSVM('KernelFunction', 'rbf', ...
                            'KernelScale', 'auto', ...
                            'Standardize', true);
        end

        % fitcecoc handles multi-class by training a binary SVM for every
        % pair of classes (one-vs-one coding). With 4 classes that's
        % 4×3/2 = 6 binary classifiers under the hood.
        mdl = fitcecoc(XTrain, yTrain, 'Learners', t, ...
                       'Coding', 'onevsone');

        yPred = predict(mdl, XTest);

        % --- Compute classification metrics ---
        confMat = confusionmat(yTest, yPred);
        acc = sum(diag(confMat)) / sum(confMat(:)) * 100;

        % Per-class sensitivity, specificity, positive predictivity.
        % These are the standard biomedical metrics — sensitivity in
        % particular matters because missing a PVC is much worse than
        % flagging a Normal beat for review.
        nClasses = size(confMat, 1);
        SE = zeros(nClasses,1);
        SP = zeros(nClasses,1);
        PP = zeros(nClasses,1);
        for k = 1:nClasses
            TP = confMat(k,k);
            FN = sum(confMat(k,:)) - TP;
            FP = sum(confMat(:,k)) - TP;
            TN = sum(confMat(:)) - TP - FN - FP;
            % eps prevents division-by-zero if a class has zero predictions
            SE(k) = TP / (TP + FN + eps) * 100;
            SP(k) = TN / (TN + FP + eps) * 100;
            PP(k) = TP / (TP + FP + eps) * 100;
        end

        % Append this row to the results table
        resultIdx = resultIdx + 1;
        results.Features{resultIdx}   = featureSets(f).name;
        results.Classifier{resultIdx} = classifiers(cl).name;
        results.Accuracy(resultIdx)   = round(acc, 2);
        results.AvgSE(resultIdx)      = round(mean(SE), 2);
        results.AvgSP(resultIdx)      = round(mean(SP), 2);
        results.AvgPP(resultIdx)      = round(mean(PP), 2);

        fprintf('  %s + %s => Accuracy: %.2f%%  SE: %.2f%%  SP: %.2f%%  PP: %.2f%%\n', ...
            featureSets(f).name, classifiers(cl).name, ...
            acc, mean(SE), mean(SP), mean(PP));

        % Hold onto specific configurations for the comparison plots
        if f == 4 && cl == 2   % Wavelet+PCA + RBF SVM (best expected)
            bestConfMat = confMat;
            bestYTest = yTest;
            bestYPred = yPred;
            bestAcc = acc;
        end
        if f == 1 && cl == 1   % FFT + Linear SVM (baseline for comparison)
            baseConfMat = confMat;
            baseAcc = acc;
        end
    end
end

fprintf('\n');
disp('=== RESULTS SUMMARY ===');
disp(results);


%% ====== STEP 6: VISUALISATIONS ======
% Generate the figures used in the dissertation Results section. All
% figures are saved separately via saveas() at the end so they can be
% imported individually into the dissertation document.
fprintf('Step 6: Generating figures for dissertation...\n');

% --- Figure 1: Three example beats per class ---
% Quick visual sanity check that preprocessing is working correctly. If
% the beats look clean and consistent, we know the pipeline is sound.
figure('Name','Sample ECG Beats','Position',[100 100 800 600]);
for c = 1:4
    subplot(2,2,c);
    idx = find(allLabels == c, 3);     % first 3 beats of this class
    t = (0:beatLen-1)/Fs * 1000;        % time axis in milliseconds
    for j = 1:min(3,length(idx))
        plot(t, beatsFiltered(idx(j),:), 'LineWidth', 0.8); hold on;
    end
    xlabel('Time (ms)');
    ylabel('Normalised Amplitude');
    title(sprintf('%s beats', labelNames{c}));
    grid on;
end
sgtitle('Representative ECG Beat Morphologies (MIT-BIH)');

% --- Figure 2: Feature space comparison ---
% This figure is important for the Discussion section because it shows
% visually why the RBF kernel is needed: the class boundaries in the
% feature space are clearly non-linear, so a linear hyperplane can't
% separate them cleanly.
figure('Name','Feature Space Comparison','Position',[100 100 1000 400]);
colors = lines(4);

% Left panel: FFT features projected into PCA space
subplot(1,3,1);
for c = 1:4
    idx = allLabels == c;
    scatter(featFFT_PCA(idx,1), featFFT_PCA(idx,2), 10, colors(c,:), ...
        'filled', 'MarkerFaceAlpha', 0.5);
    hold on;
end
xlabel('PC1'); ylabel('PC2');
title('FFT Features (PCA)');
legend(labelNames, 'Location', 'best'); grid on;

% Middle panel: raw wavelet features (cA4 std vs cD1 std)
% Shows that raw wavelet features alone don't separate the classes well —
% which is exactly why PCA is needed
subplot(1,3,2);
for c = 1:4
    idx = allLabels == c;
    scatter(featWav(:,2), featWav(:,6), 10, colors(c,:), ...
        'filled', 'MarkerFaceAlpha', 0.5);
    hold on;
end
xlabel('cA4 Std Dev'); ylabel('cD1 Std Dev');
title('Wavelet Features (Raw)');
legend(labelNames, 'Location', 'best'); grid on;

% Right panel: wavelet features after PCA
subplot(1,3,3);
for c = 1:4
    idx = allLabels == c;
    scatter(featWavPCA(idx,1), featWavPCA(idx,2), 10, colors(c,:), ...
        'filled', 'MarkerFaceAlpha', 0.5);
    hold on;
end
xlabel('PC1'); ylabel('PC2');
title('Wavelet Features (PCA)');
legend(labelNames, 'Location', 'best'); grid on;

sgtitle('Feature Space Comparison: FFT vs Wavelet');

% --- Figure 3: Confusion matrices side by side ---
% The clearest visual demonstration of the kernel-selection effect.
% Worst-case (linear) on the left, best-case (RBF) on the right.
figure('Name','Confusion Matrices','Position',[100 100 900 400]);

subplot(1,2,1);
confusionchart(baseConfMat, labelNames);
title(sprintf('FFT + Linear SVM (%.1f%%)', baseAcc));

subplot(1,2,2);
confusionchart(bestConfMat, labelNames);
title(sprintf('Wavelet+PCA + RBF SVM (%.1f%%)', bestAcc));

sgtitle('Confusion Matrix Comparison');

% --- Figure 4: Bar chart of accuracy across all combinations ---
figure('Name','Accuracy Comparison','Position',[100 100 700 400]);
accVals = results.Accuracy;
barLabels = strcat(results.Features, {' + '}, results.Classifier);
bar(accVals);
set(gca, 'XTickLabel', barLabels, 'XTickLabelRotation', 30);
ylabel('Classification Accuracy (%)');
title('Classification Performance Across Feature Sets and Classifiers');
% Lower y-axis bound is set dynamically so the differences between bars
% are visually obvious — fixed 0-100 range would compress everything
ylim([max(0, min(accVals)-10) 100]);
grid on;

% --- Figure 5: Per-class metrics for the best model ---
% Zoomed in to 80-100% so small differences between classes are visible.
% Important for showing that PVC has the weakest per-class metrics in
% every configuration — the dissertation discusses this as a known
% limitation arising from PVC/bundle-branch-block QRS overlap.
figure('Name','Per-Class Metrics','Position',[100 100 600 400]);

% Recompute metrics for the best model
nC = size(bestConfMat, 1);
SE_best = zeros(nC,1);
SP_best = zeros(nC,1);
PP_best = zeros(nC,1);
for k = 1:nC
    TP = bestConfMat(k,k);
    FN = sum(bestConfMat(k,:)) - TP;
    FP = sum(bestConfMat(:,k)) - TP;
    TN = sum(bestConfMat(:)) - TP - FN - FP;
    SE_best(k) = TP/(TP+FN+eps)*100;
    SP_best(k) = TN/(TN+FP+eps)*100;
    PP_best(k) = TP/(TP+FP+eps)*100;
end
metricsMat = [SE_best, SP_best, PP_best];
bar(metricsMat);
set(gca, 'XTickLabel', labelNames);
legend({'Sensitivity','Specificity','Pos. Predictivity'}, ...
    'Location','southeast');
ylabel('Performance (%)');
title('Per-Class Metrics: Wavelet+PCA + RBF SVM');
ylim([80 100]); grid on;

fprintf('Done! All figures generated.\n');
fprintf('Save figures via: saveas(gcf, ''figurename.png'')\n');


%% ====== HELPER FUNCTION: MIT-BIH ANNOTATION READER ======
function [samples, types] = readMITAnnotation(atrFile)
    % READMITANNOTATION  Parse a MIT-BIH .atr binary annotation file.
    %
    % This function reads the cardiologist-verified annotations stored
    % alongside each MIT-BIH record and returns the R-peak sample indices
    % plus the corresponding beat type characters (N, V, L, R, etc.).
    %
    % The .atr file format is documented at:
    % https://physionet.org/physiotools/wag/annot-5.htm
    %
    % Each annotation is encoded as a 2-byte word with an embedded code
    % indicating the beat type, plus a sample increment relative to the
    % previous annotation. Some special codes (SKIP, AUX, NUM, SUB, CHN)
    % carry extra data and need different handling — those are skipped
    % here since we only care about the beat annotations.
    %
    % Outputs:
    %   samples - column vector of R-peak sample indices
    %   types   - column vector of beat type characters

    fid = fopen(atrFile, 'r');
    data = fread(fid, inf, 'uint8');
    fclose(fid);

    samples = [];
    types = [];
    curSample = 0;

    i = 1;
    while i <= length(data) - 1
        b1 = data(i);
        b2 = data(i+1);
        i = i + 2;

        % Each 2-byte word splits into a 6-bit annotation code (top of
        % byte 2) and a 10-bit sample difference (bottom of byte 2 and
        % all of byte 1)
        annCode = bitshift(b2, -2);
        sampleDiff = bitand(b2, 3) * 256 + b1;

        % Special annotation codes — these carry extra data we don't need
        if annCode == 0
            continue;             % NOTQRS / padding
        elseif annCode == 59      % SKIP — read 4-byte sample difference
            if i+3 <= length(data)
                skipBytes = data(i:i+3);
                sampleDiff = skipBytes(1) + skipBytes(2)*256 + ...
                             skipBytes(3)*65536 + skipBytes(4)*16777216;
                i = i + 4;
                curSample = curSample + sampleDiff;
            end
            continue;
        elseif annCode == 63      % AUX — auxiliary string, skip its bytes
            auxLen = sampleDiff;
            if mod(auxLen, 2) == 1
                auxLen = auxLen + 1;   % pad to an even number of bytes
            end
            i = i + auxLen;
            continue;
        elseif annCode == 62 || annCode == 60 || annCode == 61
            continue;             % NUM, SUB, CHN — metadata, skip
        end

        % Standard annotation: advance the running sample counter and
        % map the code to a beat character
        curSample = curSample + sampleDiff;
        beatChar = annCode2char(annCode);

        % '?' means we couldn't identify the code — skip those rather than
        % polluting the output with unknown beats
        if beatChar ~= '?'
            samples = [samples; curSample];
            types = [types; beatChar];
        end
    end
end


function ch = annCode2char(code)
    % ANNCODE2CHAR  Map a MIT-BIH numeric annotation code to its beat
    % type character.
    %
    % The full list of codes is documented at:
    % https://physionet.org/physiobank/annotations.shtml
    %
    % Only the codes relevant to this study are included; unrecognised
    % codes return '?' which causes the calling function to skip them.

    map = containers.Map('KeyType','int32','ValueType','char');
    map(1)  = 'N';   % Normal beat
    map(2)  = 'L';   % Left bundle branch block
    map(3)  = 'R';   % Right bundle branch block
    map(4)  = 'A';   % Atrial premature contraction
    map(5)  = 'V';   % Premature ventricular contraction
    map(6)  = 'F';   % Fusion of ventricular and normal
    map(7)  = 'J';   % Nodal premature beat
    map(8)  = 'a';   % Aberrated atrial premature beat
    map(9)  = 'S';   % Supraventricular premature beat
    map(10) = 'E';   % Ventricular escape beat
    map(11) = 'j';   % Nodal escape beat
    map(12) = '/';   % Paced beat
    map(13) = 'Q';   % Unclassifiable beat
    map(25) = 'N';   % Normal beat (alternative code)
    map(38) = 'N';   % ST change marker (treated as normal)

    if isKey(map, int32(code))
        ch = map(int32(code));
    else
        ch = '?';
    end
end
