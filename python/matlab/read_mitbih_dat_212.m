function sig = read_mitbih_dat_212(datFile, gain, baseline)
% READ_MITBIH_DAT_212 - Decode an MIT-BIH .dat file in WFDB format-212.
%
% Quick context:
% --------------
% The MIT-BIH Arrhythmia Database stores its ECG data in a binary scheme called
% format-212, where two 12-bit signed samples are packed into three bytes. This
% saves space compared to using a full 16-bit integer per sample, but it means
% you can't just read the file as int16 and expect anything sensible to come
% out. (I tried that early in the project and got plots that looked like solid
% blocks of noise — definitely the most useful early failure of the project.)
%
% This function decodes one of those .dat files properly: it unpacks the two
% 12-bit samples from each three-byte block, sign-extends them to make negative
% voltages negative, then converts the raw ADC values to physical units (mV)
% using the gain and baseline calibration parameters from the matching .hea
% header file.
%
% Inputs:
%   datFile  - full path to the .dat file (e.g. '100.dat')
%   gain     - 1x2 vector of gain values (ADC units per mV) for each channel
%   baseline - 1x2 vector of baseline offsets in ADC units for each channel
%
% Output:
%   sig      - [N x 2] matrix of decoded ECG samples in millivolts
%
% Reference:
%   PhysioNet WFDB signal format documentation:
%   https://physionet.org/physiotools/wag/signal-5.htm
%
% Author:  Arham Ali (Student ID: 31022107)
% Module:  BI3RP3 Final-Year Research Project
% Year:    2025/26

% --- Open the binary file for reading ---
fid = fopen(datFile, 'rb');
if fid == -1
    error('Cannot open data file: %s', datFile);
end

% Read every byte in the file as unsigned 8-bit integers. The 'uint8=>uint8'
% cast keeps the values as bytes rather than auto-promoting to double, which
% saves memory on the longer records.
b = fread(fid, Inf, 'uint8=>uint8');
fclose(fid);

% --- Reshape into 3-byte frames ---
% Each frame contains two 12-bit samples. If the file length isn't an exact
% multiple of 3 (rare, but possible if a recording was truncated), drop any
% trailing partial frame.
nFrames = floor(numel(b) / 3);
b = b(1:3 * nFrames);

% Split into three byte streams: byte 1, byte 2, byte 3 of each frame
b1 = double(b(1:3:end));   % low byte of sample 1
b2 = double(b(2:3:end));   % high nibble of sample 1 + high nibble of sample 2
b3 = double(b(3:3:end));   % low byte of sample 2

% --- Reconstruct the 12-bit samples from the packed bytes ---
% The format packs the two samples like this:
%   sample 1 = (low 4 bits of b2) << 8  |  b1
%   sample 2 = (high 4 bits of b2) << 8 |  b3
% bitand() with 15 (binary 0000 1111) extracts the low 4 bits of b2.
% bitshift() with -4 shifts b2 right by 4 bits to extract the high 4 bits.
s1 = b1 + 256 * bitand(b2, 15);
s2 = b3 + 256 * bitshift(b2, -4);

% --- Sign extension (12-bit signed → MATLAB integer) ---
% A 12-bit signed value can hold numbers from -2048 to +2047. Anything we just
% computed that's >= 2048 is actually a negative number that needs unwrapping
% by subtracting 4096. This is the step that turns positive-looking ADC values
% back into the negative voltages they originally represented.
s1(s1 >= 2048) = s1(s1 >= 2048) - 4096;
s2(s2 >= 2048) = s2(s2 >= 2048) - 4096;

% Stack the two channels side-by-side into an [N x 2] matrix
raw = [s1(:) s2(:)];

% --- Calibration: ADC units → millivolts ---
% Both channels typically have different gain values (usually ~200 ADU/mV for
% MIT-BIH) and slightly different baselines, so they're calibrated separately.
sig = zeros(size(raw));
sig(:, 1) = (raw(:, 1) - baseline(1)) / gain(1);
sig(:, 2) = (raw(:, 2) - baseline(2)) / gain(2);
end
