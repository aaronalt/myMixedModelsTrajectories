%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% main_CPT_controls.m - Linear mixed models on CPT T-scores (Controls)
%
% Same analysis as main_CPT.m but for healthy controls (DIAG == 0).
%
% Pass 1: score ~ age + clau_LH + clau_RH + age*clau interactions
% Pass 2: score ~ age only (no claustrum covariate)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear all
close all

scriptDir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(scriptDir, 'functions')));
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

% Filter: keep only observations with age < 35
ageKeep = mCPT.AGE < 35;
fprintf('\nAge filter (< 35): keeping %d of %d observations.\n', sum(ageKeep), nObs);
mCPT.ID     = mCPT.ID(ageKeep);
mCPT.AGE    = mCPT.AGE(ageKeep);
mCPT.DIAG   = mCPT.DIAG(ageKeep);
scoreData   = scoreData(ageKeep, :);
clau        = clau(ageKeep, :);
nObs        = sum(ageKeep);

fprintf('%d observations from %d subjects after filtering.\n', ...
    nObs, length(unique(mCPT.ID)));

% Filter: keep only Controls (DIAG == 0)
ctrlKeep = mCPT.DIAG == 0;
fprintf('Control filter: keeping %d of %d observations.\n', sum(ctrlKeep), nObs);
mCPT.ID     = mCPT.ID(ctrlKeep);
mCPT.AGE    = mCPT.AGE(ctrlKeep);
mCPT.DIAG   = mCPT.DIAG(ctrlKeep);
scoreData   = scoreData(ctrlKeep, :);
clau        = clau(ctrlKeep, :);
nObs        = sum(ctrlKeep);

fprintf('%d Control observations from %d subjects.\n', ...
    nObs, length(unique(mCPT.ID)));

%% ------------------------------------------------------------------------
% Set up input struct for fitOptModel
% ------------------------------------------------------------------------
input.subjID   = mCPT.ID;
input.age      = mCPT.AGE;
input.grouping = [];                % no group — controls only
input.data     = scoreData;         % nObs x nScores
input.cov      = clau;              % claustrum LH + RH as covariates

%% ------------------------------------------------------------------------
% Model estimation options
% ------------------------------------------------------------------------
opts.orders    = 1;         % linear age effect only
opts.mType     = 'slope';   % random intercept + slope (recommended)
opts.vertID    = 1:nScores; % one model per CPT score
opts.modelNames = scoreCols;
opts.alpha     = 0.05;
opts.figPosition = [440 488 525 310];

%% ------------------------------------------------------------------------
% Fit models (with claustrum covariate)
% ------------------------------------------------------------------------
fprintf('\n=== Fitting mixed models for %d CPT measures (Controls) ===\n', nScores);
outModelVect = fitOptModel(input, opts);

% FDR correction across all CPT measures
outModelVect_corr = fdr_correct(outModelVect, opts.alpha);

%% ------------------------------------------------------------------------
% Plot and save results
% ------------------------------------------------------------------------
outDir = fullfile('./results_CPT_controls');

plotOpts.legTxt   = {'Control'};
plotOpts.xLabel   = 'Age';
plotOpts.yLabel   = 'CPT T-score';
plotOpts.plotCI   = 1;
plotOpts.plotType  = 'redInter';
plotOpts.nCov     = size(input.cov, 2) * (1 + max(opts.orders));

saveResults = 2;  % 0=no, 1=table only, 2=table + plots

plotModelsAndSaveResults(outModelVect_corr, plotOpts, saveResults, outDir);

fprintf('\n=== Done. Results saved to %s ===\n', outDir);

%% ========================================================================
% PASS 2: No claustrum covariate — pure age effect on CPT (Controls)
% ========================================================================
fprintf('\n=== Fitting models WITHOUT claustrum covariate (Controls) ===\n');
input.cov = [];

outModelVect_noCov = fitOptModel(input, opts);
outModelVect_noCov = fdr_correct(outModelVect_noCov, opts.alpha);

outDir_noCov = fullfile('./results_CPT_controls', 'noCov');

plotOpts.nCov = 0;
plotModelsAndSaveResults(outModelVect_noCov, plotOpts, saveResults, outDir_noCov);

fprintf('\n=== No-covariate results saved to %s ===\n', outDir_noCov);
