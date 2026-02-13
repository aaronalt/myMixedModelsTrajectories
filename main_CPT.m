%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% main_CPT.m - Linear mixed models on CPT T-scores
%
% Fits polynomial mixed-effect models (via nlmefit) to each CPT T-score
% measure across subjects and timepoints, with diagnosis as the grouping
% variable, age as the continuous predictor, and claustrum volume (LH, RH)
% as covariates.
%
% VCFS-only model per score column:
%   score ~ 1 + age + clau_LH + clau_RH + age*clau_LH + age*clau_RH + (1 + age | subject)
%
% The age*claustrum interaction tests whether the claustrum's contribution
% to attention changes over development in VCFS subjects.
% Model selection uses BIC (orders 0-1 by default), corrected for
% multiple comparisons using FDR.
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

% Filter: keep only VCFS (22q) subjects
vcfsKeep = mCPT.DIAG == 1;
fprintf('VCFS filter: keeping %d of %d observations.\n', sum(vcfsKeep), nObs);
mCPT.ID     = mCPT.ID(vcfsKeep);
mCPT.AGE    = mCPT.AGE(vcfsKeep);
mCPT.DIAG   = mCPT.DIAG(vcfsKeep);
scoreData   = scoreData(vcfsKeep, :);
clau        = clau(vcfsKeep, :);
nObs        = sum(vcfsKeep);

fprintf('%d VCFS observations from %d subjects.\n', ...
    nObs, length(unique(mCPT.ID)));

%% ------------------------------------------------------------------------
% Common input and options
% ------------------------------------------------------------------------
input.subjID   = mCPT.ID;
input.age      = mCPT.AGE;
input.grouping = [];                % no group — VCFS only
input.data     = scoreData;         % nObs x nScores

opts.orders    = 1;         % linear age effect only
opts.mType     = 'slope';   % random intercept + slope (recommended)
opts.vertID    = 1:nScores; % one model per CPT score
opts.modelNames = scoreCols;
opts.alpha     = 0.05;
opts.figPosition = [440 488 525 310];

plotOpts.legTxt   = {'22q'};
plotOpts.xLabel   = 'Age';
plotOpts.yLabel   = 'CPT T-score';
plotOpts.plotCI   = 1;
plotOpts.plotType  = 'redInter';
saveResults = 2;  % 0=no, 1=table only, 2=table + plots

%% ------------------------------------------------------------------------
% Left Hemisphere — claustrum LH as covariate
% ------------------------------------------------------------------------
fprintf('\n=== LH: Fitting mixed models for %d CPT measures ===\n', nScores);
input.cov = clau(:, 1);
plotOpts.nCov = size(input.cov, 2) * (1 + max(opts.orders));

outModelVect_LH = fitOptModel(input, opts);
outModelVect_LH = fdr_correct(outModelVect_LH, opts.alpha);

outDir_LH = fullfile('./results_CPT', 'LH');
plotModelsAndSaveResults(outModelVect_LH, plotOpts, saveResults, outDir_LH);
fprintf('\n=== LH done. Results saved to %s ===\n', outDir_LH);

%% ------------------------------------------------------------------------
% Right Hemisphere — claustrum RH as covariate
% ------------------------------------------------------------------------
fprintf('\n=== RH: Fitting mixed models for %d CPT measures ===\n', nScores);
input.cov = clau(:, 2);
plotOpts.nCov = size(input.cov, 2) * (1 + max(opts.orders));

outModelVect_RH = fitOptModel(input, opts);
outModelVect_RH = fdr_correct(outModelVect_RH, opts.alpha);

outDir_RH = fullfile('./results_CPT', 'RH');
plotModelsAndSaveResults(outModelVect_RH, plotOpts, saveResults, outDir_RH);
fprintf('\n=== RH done. Results saved to %s ===\n', outDir_RH);
