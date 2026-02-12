function [matchedX, matchedCPT, matchInfo] = matchAndConcatCPT(sorted, cptMatFile, subjField, ageField, cptSubjField, cptAgeField)
% matchAndConcatCPT - match sorted imaging data with CPT scores by subject
%                     and closest age within 1 year, then concatenate
%
% Syntax:
%    [matchedX, matchedCPT, matchInfo] = matchAndConcatCPT(sorted, cptMatFile)
%    [matchedX, matchedCPT, matchInfo] = matchAndConcatCPT(sorted, cptMatFile, ...
%                                           subjField, ageField, cptSubjField, cptAgeField)
%
% Inputs:
%    sorted       - struct from sortDemographics, must contain .X and
%                   fields for subject ID and age
%    cptMatFile   - path to a .mat file containing CPT T-score data. Must
%                   have variables for subject ID, age, and score data.
%    subjField    - (optional) name of subject ID field in sorted
%                   (default: auto-detect field containing 'subj')
%    ageField     - (optional) name of age field in sorted
%                   (default: auto-detect field containing 'age')
%    cptSubjField - (optional) name of subject ID variable in CPT .mat
%                   (default: auto-detect variable containing 'subj')
%    cptAgeField  - (optional) name of age variable in CPT .mat
%                   (default: auto-detect variable containing 'age')
%
% Outputs:
%    matchedX    - subset of sorted.X rows that had a CPT match within 1 year
%    matchedCPT  - struct with every variable from the CPT .mat file,
%                  subset to the matched rows (same row count as matchedX)
%    matchInfo   - struct with fields: subject, sortedAge, cptAge, ageDiff
%                  documenting every match that was made
%
% Example:
%    sorted = sortDemographics(lh_thickness, 'demographics.mat');
%    [mX, mCPT, info] = matchAndConcatCPT(sorted, 'CPT_table_Tscore.mat');

maxAgeDiff = 1; % maximum allowed age difference in years

%% load CPT .mat file
CPT = load(cptMatFile);
cptFields = fieldnames(CPT);

%% auto-detect field names in sorted struct
sFields = fieldnames(sorted);

if nargin < 3 || isempty(subjField)
    idx = find(cellfun(@(f) ~isempty(regexpi(f,'subj')), sFields));
    if isempty(idx)
        idx = find(cellfun(@(f) ~isempty(regexp(f,'^ID$','ignorecase')), sFields));
    end
    if isempty(idx)
        error('matchAndConcatCPT:noSubjField', ...
            'No subject ID field found in sorted. Available: %s', ...
            strjoin(sFields, ', '));
    end
    subjField = sFields{idx(1)};
end

if nargin < 4 || isempty(ageField)
    idx = find(cellfun(@(f) ~isempty(regexpi(f,'age')), sFields));
    if isempty(idx)
        error('matchAndConcatCPT:noAgeField', ...
            'No field containing ''age'' found in sorted. Available: %s', ...
            strjoin(sFields, ', '));
    end
    ageField = sFields{idx(1)};
end

%% auto-detect field names in CPT struct
if nargin < 5 || isempty(cptSubjField)
    idx = find(cellfun(@(f) ~isempty(regexpi(f,'subj')), cptFields));
    if isempty(idx)
        % fall back to a field named 'ID' (case-insensitive exact match)
        idx = find(cellfun(@(f) ~isempty(regexp(f,'^ID$','ignorecase')), cptFields));
    end
    if isempty(idx)
        error('matchAndConcatCPT:noCptSubj', ...
            'No subject ID variable found in %s. Available: %s', ...
            cptMatFile, strjoin(cptFields, ', '));
    end
    cptSubjField = cptFields{idx(1)};
end

if nargin < 6 || isempty(cptAgeField)
    idx = find(cellfun(@(f) ~isempty(regexpi(f,'age')), cptFields));
    if isempty(idx)
        error('matchAndConcatCPT:noCptAge', ...
            'No variable containing ''age'' found in %s. Available: %s', ...
            cptMatFile, strjoin(cptFields, ', '));
    end
    cptAgeField = cptFields{idx(1)};
end

%% extract vectors
sortedSubj = sorted.(subjField);
sortedAge  = sorted.(ageField);
cptSubj    = CPT.(cptSubjField);
cptAge     = CPT.(cptAgeField);

nSorted = length(sortedSubj);
nCpt    = length(cptSubj);

%% match each row in sorted to the closest CPT row by age (within 1 year)
keepIdx     = false(nSorted, 1);  % which sorted rows have a match
cptMatchIdx = zeros(nSorted, 1);  % index into CPT data for each match
ageDiffs    = nan(nSorted, 1);

for iR = 1:nSorted
    % find all CPT rows for this subject
    sameSubj = (cptSubj == sortedSubj(iR));
    if ~any(sameSubj)
        continue;
    end

    candidateIdx = find(sameSubj);
    candidateAges = cptAge(candidateIdx);

    % find closest age
    [minDiff, bestPos] = min(abs(candidateAges - sortedAge(iR)));

    if minDiff <= maxAgeDiff
        keepIdx(iR)     = true;
        cptMatchIdx(iR) = candidateIdx(bestPos);
        ageDiffs(iR)    = candidateAges(bestPos) - sortedAge(iR);
    end
end

%% build outputs
matchedX = sorted.X(keepIdx, :);

% subset every CPT variable to the matched rows
keptCptIdx = cptMatchIdx(keepIdx);
matchedCPT = struct();
for iF = 1:length(cptFields)
    val = CPT.(cptFields{iF});
    if isvector(val) && length(val) == nCpt
        matchedCPT.(cptFields{iF}) = val(keptCptIdx);
    elseif ismatrix(val) && size(val,1) == nCpt
        matchedCPT.(cptFields{iF}) = val(keptCptIdx, :);
    else
        matchedCPT.(cptFields{iF}) = val;
    end
end

matchInfo.ID   = sortedSubj(keepIdx);
matchInfo.sortedAge = sortedAge(keepIdx);
matchInfo.cptAge    = cptAge(keptCptIdx);
matchInfo.ageDiff   = ageDiffs(keepIdx);

nMatched = sum(keepIdx);
nDropped = nSorted - nMatched;
fprintf('Matched %d of %d rows (%d dropped, no CPT within %.0f year).\n', ...
    nMatched, nSorted, nDropped, maxAgeDiff);
end
