%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% main_CPT.m - Linear mixed models on CPT T-scores
%
% Fits polynomial mixed-effect models (via nlmefit) to each CPT T-score
% measure across subjects and timepoints, with diagnosis as the grouping
% variable and age as the continuous predictor.
%
% Model per score column:
%   score ~ 1 + group + age + age*group + (1 + age | subject)
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
% Load data
% ------------------------------------------------------------------------
load('demographics.mat');       % subject, age, diagnosis_bin, gender_bin
load('CPT_table_Tscore.mat');   % CPT_table_Tscore (table)

% Unwrap the table if the load placed it inside a struct field
if ~exist('CPT_table_Tscore','var')
    ws = whos;
    for iW = 1:length(ws)
        if istable(eval(ws(iW).name))
            CPT_table_Tscore = eval(ws(iW).name);
            break;
        end
    end
end
if ~istable(CPT_table_Tscore)
    error('Expected CPT_table_Tscore to be a MATLAB table.');
end

cptTable = CPT_table_Tscore;
fprintf('CPT table: %d rows x %d columns\n', height(cptTable), width(cptTable));
disp(cptTable.Properties.VariableNames');

%% ------------------------------------------------------------------------
% Identify ID, age, and score columns
% ------------------------------------------------------------------------
varNames = cptTable.Properties.VariableNames;

% Subject ID column: first column matching 'subj' or exactly 'ID'
idIdx = find(cellfun(@(v) ~isempty(regexpi(v,'subj')), varNames), 1);
if isempty(idIdx)
    idIdx = find(cellfun(@(v) ~isempty(regexp(v,'^ID$','ignorecase')), varNames), 1);
end
if isempty(idIdx)
    error('No subject ID column found in CPT table. Available: %s', strjoin(varNames,', '));
end
idCol = varNames{idIdx};

% Age column: first column matching 'age'
ageIdx = find(cellfun(@(v) ~isempty(regexpi(v,'age')), varNames), 1);
if isempty(ageIdx)
    error('No age column found in CPT table. Available: %s', strjoin(varNames,', '));
end
ageCol = varNames{ageIdx};

% Score columns = all remaining numeric columns
scoreColMask = true(1, length(varNames));
scoreColMask(idIdx)  = false;
scoreColMask(ageIdx) = false;
% keep only numeric columns
for iV = 1:length(varNames)
    if scoreColMask(iV) && ~isnumeric(cptTable.(varNames{iV}))
        scoreColMask(iV) = false;
    end
end
scoreCols = varNames(scoreColMask);

fprintf('\nSubject ID column : %s\n', idCol);
fprintf('Age column        : %s\n', ageCol);
fprintf('Score columns (%d):\n', length(scoreCols));
for iS = 1:length(scoreCols)
    fprintf('  %d. %s\n', iS, scoreCols{iS});
end

%% ------------------------------------------------------------------------
% Extract CPT vectors and build score matrix
% ------------------------------------------------------------------------
cptID  = cptTable.(idCol);
cptAge = cptTable.(ageCol);

% Convert cell / categorical ID to numeric if needed
if iscell(cptID)
    cptID = str2double(cptID);
end
if iscategorical(cptID)
    cptID = double(cptID);
end
cptAge = double(cptAge);

nObs    = height(cptTable);
nScores = length(scoreCols);
scoreData = nan(nObs, nScores);
for iS = 1:nScores
    scoreData(:, iS) = double(cptTable.(scoreCols{iS}));
end

%% ------------------------------------------------------------------------
% Merge diagnosis from demographics by subject ID
% ------------------------------------------------------------------------
cptDx = nan(nObs, 1);
for iR = 1:nObs
    demoIdx = find(subject == cptID(iR), 1);
    if ~isempty(demoIdx)
        cptDx(iR) = diagnosis_bin(demoIdx);
    end
end

%% ------------------------------------------------------------------------
% Filter observations
% ------------------------------------------------------------------------
validMask = ~isnan(cptDx) & ~isnan(cptAge) & cptAge < 34;

cptID_f     = cptID(validMask);
cptAge_f    = cptAge(validMask);
cptDx_f     = cptDx(validMask);
scoreData_f = scoreData(validMask, :);

% Sort by subject then age (required by nlmefit)
[~, sortIdx] = sortrows([cptID_f, cptAge_f], [1 2]);
cptID_f     = cptID_f(sortIdx);
cptAge_f    = cptAge_f(sortIdx);
cptDx_f     = cptDx_f(sortIdx);
scoreData_f = scoreData_f(sortIdx, :);

fprintf('\n%d observations from %d subjects after filtering.\n', ...
    length(cptID_f), length(unique(cptID_f)));

%% ------------------------------------------------------------------------
% Set up input struct for fitOptModel
% ------------------------------------------------------------------------
input.subjID   = cptID_f;
input.age      = cptAge_f;
input.grouping = cptDx_f;       % 0/1 binary (HC vs 22q)
input.data     = scoreData_f;   % nObs x nScores
input.cov      = [];            % no additional covariates

%% ------------------------------------------------------------------------
% Model estimation options
% ------------------------------------------------------------------------
opts.orders    = [0 1];     % test constant and linear age effect
opts.mType     = 'slope';   % random intercept + slope (recommended)
opts.vertID    = 1:nScores; % one model per CPT score
opts.modelNames = scoreCols';
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
