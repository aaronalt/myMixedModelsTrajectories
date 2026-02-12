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

% Mean-center age to reduce multicollinearity with claustrum volume
ageMean = nanmean(mCPT.AGE);
mCPT.AGE = mCPT.AGE - ageMean;
fprintf('Age mean-centered (mean = %.2f subtracted).\n', ageMean);

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
opts.orders    = 1;         % linear age effect only
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
% Effect sizes
% ------------------------------------------------------------------------
effectSizeGroup = GroupCalculationEffect(outModelVect);
effectSizeInter = InterCalculationEffect(outModelVect);

fprintf('\n=== Done. Results saved to %s ===\n', outDir);
