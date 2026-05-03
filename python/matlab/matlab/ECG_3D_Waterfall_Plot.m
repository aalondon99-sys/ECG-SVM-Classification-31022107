%% ECG_3D_Waterfall_Plot.m
%  Generate a 3D waterfall view of preprocessed ECG beats.
%
%  Quick context:
%  --------------
%  The Jannah and Hadjiloucas (2023) paper that this project follows on from
%  used a really nice visualisation style: it stacks individual ECG beats
%  in 3D so you can see how consistent (or inconsistent) they are across
%  multiple heartbeats. This script reproduces that style for our four
%  arrhythmia classes after the main classification pipeline has been run.
%
%  Why bother with a 3D plot when we have 2D figures already?
%  Because it makes beat-to-beat morphological consistency visually obvious
%  in a way that overlapping 2D traces don't. If the R-peaks line up cleanly
%  across the depth axis, you know the preprocessing pipeline (filtering and
%  segmentation) is doing its job. If they don't, something needs fixing.
%
%  Prerequisites:
%  --------------
%  Run the main classification script first. This script needs two variables
%  to exist in the workspace:
%      beatsFiltered  -  matrix of preprocessed beats (rows = beats)
%      allLabels      -  vector of class labels (1 = Normal, 2 = PVC, etc.)
%
%  Author:  Arham Ali (Student ID: 31022107)
%  Module:  BI3RP3 Final-Year Research Project
%  Year:    2025/26

% --- User settings ---
% Which class to plot in the single-figure view. Change this to focus on a
% different arrhythmia class:
%   1 = Normal, 2 = PVC, 3 = LBBB, 4 = RBBB
classToPlot = 1;
classNames = {'Normal', 'PVC', 'LBBB', 'RBBB'};

% How many beats to stack in the depth direction. Ten is a good balance —
% enough to show beat-to-beat variability, not so many that the figure
% gets cluttered.
nBeatsToShow = 10;

% --- Find the beats belonging to the chosen class ---
% Take the first nBeatsToShow indices where the label matches the chosen class
classIdx = find(allLabels == classToPlot);
classIdx = classIdx(1:min(nBeatsToShow, length(classIdx)));

% Pull those beats out into their own matrix
beatsToPlot = beatsFiltered(classIdx, :);

% Build the time axis (in samples). Could convert to milliseconds by dividing
% by Fs and multiplying by 1000, but sample index is fine for visualisation.
timeAxis = 0:size(beatsToPlot, 2) - 1;


% =====================================================================
%  FIGURE 1: Single-class waterfall view
% =====================================================================
figure('Name', '3D ECG Beats', 'Position', [100 100 800 500]);

% waterfall() draws successive rows of the matrix as connected lines in 3D,
% which gives that characteristic stacked-curtain look
waterfall(timeAxis, 1:size(beatsToPlot, 1), beatsToPlot);

% Label the axes properly so the figure is publication-ready
xlabel('Time index');
ylabel('Beat number');
zlabel('Amplitude (mV)');
title(sprintf('%s ECG Beats — 3D Waterfall View', classNames{classToPlot}));

% Jet colormap gives a nice rainbow gradient across the depth axis.
% Could also use parula (the modern MATLAB default) but jet matches the
% reference paper better.
colormap(jet);

% View angle (-30, 35) was chosen by trial and error to match the example
% figure in the Jannah and Hadjiloucas (2023) paper. The first number is the
% azimuth (rotation around the vertical axis), the second is the elevation
% (tilt of the camera up or down).
view(-30, 35);

grid on;
set(gca, 'FontSize', 11);

fprintf('Showing %d %s beats in 3D waterfall plot\n', ...
    size(beatsToPlot, 1), classNames{classToPlot});


% =====================================================================
%  FIGURE 2: Four-class panel comparison
% =====================================================================
% Same idea as above, but laid out as a 2x2 grid showing all four classes
% side-by-side. This is the version that actually makes it into the
% dissertation because it lets the reader compare morphology across classes
% in a single glance.
figure('Position', [100 100 1000 800]);
for c = 1:4
    subplot(2, 2, c);

    % Take the first 10 beats of class c
    idx = find(allLabels == c, 10);

    waterfall(timeAxis, 1:length(idx), beatsFiltered(idx, :));
    xlabel('Time index');
    ylabel('Beat number');
    zlabel('Amplitude');
    title(sprintf('%s beats', classNames{c}));

    colormap(jet);
    view(-30, 35);
    grid on;
end
