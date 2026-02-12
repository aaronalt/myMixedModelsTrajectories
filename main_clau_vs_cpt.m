%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% main_clau_vs_cpt.m - Linear mixed models of claustrum volumes vs CPT scores
%
% Fits polynomial mixed-effect models (via nlmefit) to claustrum volume
% (LH, RH) across subjects and timepoints, with diagnosis as the grouping
% variable, age as the continuous predictor, and CPT T-scores as covariates.
%
% Claustrum volumes are first residualized for TIV and gender.
%
% Model per claustrum hemisphere:
%   clau_vol ~ 1 + group + age + age*group + CPT_scores + (1 + age | subject)
%
% Model selection uses BIC (orders 0-1). Group and interaction effects are
% tested with likelihood-ratio tests and corrected for multiple comparisons
% using FDR.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear all
close all

addpath(genpath('./functions'));
startup;

%% ------------------------------------------------------------------------
% Load data and match CPT to imaging by subject / age
% ------------------------------------------------------------------------
load("demographics.mat")
load("clau_brain_vols.mat")
load("CPT_table_Tscore.mat")

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

fprintf('\nCPT score columns used as covariates (%d):\n', length(scoreCols));
for iS = 1:length(scoreCols)
    fprintf('  %d. %s\n', iS, scoreCols{iS});
end

%% ------------------------------------------------------------------------
% Build CPT covariate matrix
% ------------------------------------------------------------------------
nObs    = length(mCPT.ID);
nScores = length(scoreCols);
cptCov  = nan(nObs, nScores);
for iS = 1:nScores
    cptCov(:, iS) = double(mCPT.(scoreCols{iS}));
end

%% ------------------------------------------------------------------------
% Prepare claustrum volumes (residualized for TIV and gender)
% ------------------------------------------------------------------------
clau   = mX(:, 1:2);              % columns 1=LH, 2=RH
tivcov = mX(:, 16);               % total intracranial volume
clau_res = compute_residuals_dimitri_2(clau, [tivcov, mCPT.GENDER]);

fprintf('\n%d matched observations from %d subjects.\n', ...
    nObs, length(unique(mCPT.ID)));

%% ------------------------------------------------------------------------
% Set up input struct for fitOptModel
% ------------------------------------------------------------------------
input.subjID   = mCPT.ID;
input.age      = mCPT.AGE;
input.grouping = mCPT.DIAG;         % 0/1 binary (HC vs 22q)
input.data     = clau_res;           % claustrum LH + RH (residualized)
input.cov      = cptCov;             % CPT T-scores as covariates

%% ------------------------------------------------------------------------
% Model estimation options
% ------------------------------------------------------------------------
opts.orders    = [0 1];              % test constant and linear age effect
opts.mType     = 'slope';            % random intercept + slope
opts.vertID    = [1 2];              % LH and RH
opts.modelNames = {'Claustrum_LH', 'Claustrum_RH'};
opts.alpha     = 0.05;
opts.figPosition = [440 488 525 310];

%% ------------------------------------------------------------------------
% Fit models
% ------------------------------------------------------------------------
fprintf('\n=== Fitting mixed models for claustrum volumes ===\n');
outModelVect = fitOptModel(input, opts);

% FDR correction
outModelVect_corr = fdr_correct(outModelVect, opts.alpha);

%% ------------------------------------------------------------------------
% Plot and save results
% ------------------------------------------------------------------------
outDir = fullfile('./results_clau_vs_cpt');

plotOpts.legTxt   = {'HC', '22q'};
plotOpts.xLabel   = 'Age';
plotOpts.yLabel   = 'Claustrum volume (residualized)';
plotOpts.plotCI   = 1;
plotOpts.plotType = 'redInter';
plotOpts.nCov     = size(input.cov, 2);

saveResults = 2;  % 0=no, 1=table only, 2=table + plots

plotModelsAndSaveResults(outModelVect_corr, plotOpts, saveResults, outDir);

%% ------------------------------------------------------------------------
% Effect sizes
% ------------------------------------------------------------------------
effectSizeGroup = GroupCalculationEffect(outModelVect);
effectSizeInter = InterCalculationEffect(outModelVect);

fprintf('\n=== Done. Results saved to %s ===\n', outDir);
