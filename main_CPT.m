%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% main_CPT.m - Linear mixed models on CPT T-scores
%
% Fits polynomial mixed-effect models (via nlmefit) to each CPT T-score
% measure across subjects and timepoints, with diagnosis as the grouping
% variable, age as the continuous predictor, and claustrum volume (LH, RH)
% as covariates.
%
% Model per score column:
%   score ~ 1 + group + age + age*group + clau_LH + clau_RH + (1 + age | subject)
%
% Model selection uses BIC (orders 0-1 by default). Group and interaction
% effects are tested with likelihood-ratio tests and corrected for
% multiple comparisons using FDR.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear all
close all

addpath(genpath('./functions'));
startup;

%% ------------------------------------------------------------------------
% Load data and match CPT to imaging by subject / age
% ------------------------------------------------------------------------
load("/home/aaron/myMixedModelsTrajectories/demographics.mat")
load("/home/aaron/myMixedModelsTrajectories/clau_brain_vols.mat")
load("/home/aaron/myMixedModelsTrajectories/CPT_table_Tscore.mat")

sorted = sortDemographics(X, 'demographics.mat');
[mX, mCPT, info] = matchAndConcatCPT(sorted, 'CPT_table_Tscore.mat');

%% ------------------------------------------------------------------------
% Identify CPT score columns (exclude metadata: ID, AGE, DIAG, GENDER)
% ------------------------------------------------------------------------
metaFields = {'ID','AGE','DIAG','GENDER'};
allFields  = fieldnames(mCPT);
scoreMask  = ~ismember(upper(allFields), upper(metaFields));
% keep only numeric vectors
for iF = 1:length(allFields)
    if scoreMask(iF) && (~isnumeric(mCPT.(allFields{iF})) || ~isvector(mCPT.(allFields{iF})))
        scoreMask(iF) = false;
    end
end
scoreCols = allFields(scoreMask);

fprintf('\nCPT score columns (%d):\n', length(scoreCols));
for iS = 1:length(scoreCols)
    fprintf('  %d. %s\n', iS, scoreCols{iS});
end

%% ------------------------------------------------------------------------
% Build score matrix and extract covariates
% ------------------------------------------------------------------------
nObs    = length(mCPT.ID);
nScores = length(scoreCols);
scoreData = nan(nObs, nScores);
for iS = 1:nScores
    scoreData(:, iS) = double(mCPT.(scoreCols{iS}));
end

% Remove outliers (> 3 SD from mean) per score column
for iS = 1:nScores
    col = scoreData(:, iS);
    mu  = nanmean(col);
    sd  = nanstd(col);
    outliers = abs(col - mu) > 3 * sd;
    nOut = sum(outliers);
    if nOut > 0
        fprintf('  %s: %d outlier(s) removed\n', scoreCols{iS}, nOut);
        scoreData(outliers, iS) = NaN;
    end
end

% Claustrum volumes (columns 1=LH, 2=RH from matched imaging data)
clau = mX(:, 1:2);

fprintf('\n%d matched observations from %d subjects.\n', ...
    nObs, length(unique(mCPT.ID)));

%% ------------------------------------------------------------------------
% Set up input struct for fitOptModel
% ------------------------------------------------------------------------
input.subjID   = mCPT.ID;
input.age      = mCPT.AGE;
input.grouping = mCPT.DIAG;        % 0/1 binary (HC vs 22q)
input.data     = scoreData;         % nObs x nScores
input.cov      = clau;              % claustrum LH + RH as covariates

%% ------------------------------------------------------------------------
% Model estimation options
% ------------------------------------------------------------------------
opts.orders    = [0 1];     % test constant and linear age effect
opts.mType     = 'slope';   % random intercept + slope (recommended)
opts.vertID    = 1:nScores; % one model per CPT score
opts.modelNames = scoreCols;
opts.alpha     = 0.05;
opts.figPosition = [440 488 525 310];

%% ------------------------------------------------------------------------
% Fit models
% ------------------------------------------------------------------------
fprintf('\n=== Fitting mixed models for %d CPT measures ===\n', nScores);
outModelVect = fitOptModel(input, opts);

% FDR correction across all CPT measures
outModelVect_corr = fdr_correct(outModelVect, opts.alpha);

%% ------------------------------------------------------------------------
% Plot and save results
% ------------------------------------------------------------------------
outDir = fullfile('./results_CPT');

plotOpts.legTxt   = {'HC', '22q'};
plotOpts.xLabel   = 'Age';
plotOpts.yLabel   = 'CPT T-score';
plotOpts.plotCI   = 1;
plotOpts.plotType  = 'redInter';
plotOpts.nCov     = size(input.cov, 2);

saveResults = 2;  % 0=no, 1=table only, 2=table + plots

plotModelsAndSaveResults(outModelVect_corr, plotOpts, saveResults, outDir);

%% ------------------------------------------------------------------------
% Scatter plots: summed claustrum volume vs CPT score
% ------------------------------------------------------------------------
clauSum = clau(:, 1) + clau(:, 2);   % LH + RH
diag    = mCPT.DIAG;
grpIdx0 = diag == 0;
grpIdx1 = diag == 1;
colHC   = [0 0 1];
col22q  = [1 0 0];

scatterDir = fullfile(outDir, 'scatter_clau_vs_cpt');
if ~exist(scatterDir, 'dir'), mkdir(scatterDir); end

for iS = 1:nScores
    y = scoreData(:, iS);
    valid = ~isnan(y) & ~isnan(clauSum);

    fig = figure('Position', [440 488 525 420], 'Visible', 'off');
    hold on;

    scatter(clauSum(valid & grpIdx0), y(valid & grpIdx0), 40, colHC,  'filled', 'MarkerFaceAlpha', 0.5);
    scatter(clauSum(valid & grpIdx1), y(valid & grpIdx1), 40, col22q, 'filled', 'MarkerFaceAlpha', 0.5);

    xlabel('Summed Claustrum Volume (LH + RH)');
    ylabel(scoreCols{iS}, 'Interpreter', 'none');
    title(sprintf('Claustrum Volume vs %s', scoreCols{iS}), 'Interpreter', 'none');
    legend({'HC', '22q'}, 'Location', 'best');
    hold off;

    safeName = regexprep(scoreCols{iS}, '[^a-zA-Z0-9_]', '_');
    saveas(fig, fullfile(scatterDir, ['scatter_clau_vs_' safeName '.eps']), 'epsc');
    close(fig);
end
fprintf('\nScatter plots saved to %s\n', scatterDir);

%% ------------------------------------------------------------------------
% Effect sizes
% ------------------------------------------------------------------------
effectSizeGroup = GroupCalculationEffect(outModelVect);
effectSizeInter = InterCalculationEffect(outModelVect);

fprintf('\n=== Done. Results saved to %s ===\n', outDir);
