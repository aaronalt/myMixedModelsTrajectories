function [matchedX, matchedCPT, matchInfo] = matchAndConcatCPT(sorted, CPT_table_Tscore, subjField, ageField, cptSubjField, cptAgeField)
% matchAndConcatCPT - match sorted imaging data with CPT scores by subject
%                     and closest age within 1 year, then concatenate
%
% Syntax:
%    [matchedX, matchedCPT, matchInfo] = matchAndConcatCPT(sorted, CPT_table_Tscore)
%    [matchedX, matchedCPT, matchInfo] = matchAndConcatCPT(sorted, CPT_table_Tscore, ...
%                                           subjField, ageField, cptSubjField, cptAgeField)
%
% Inputs:
%    sorted           - struct from sortDemographics, must contain .X and
%                       fields for subject ID and age
%    CPT_table_Tscore - table with CPT T-score data, must contain columns
%                       for subject ID and age plus score columns
%    subjField        - (optional) name of subject ID field in sorted
%                       (default: auto-detect field containing 'subj')
%    ageField         - (optional) name of age field in sorted
%                       (default: auto-detect field containing 'age')
%    cptSubjField     - (optional) name of subject column in CPT table
%                       (default: auto-detect column containing 'subj')
%    cptAgeField      - (optional) name of age column in CPT table
%                       (default: auto-detect column containing 'age')
%
% Outputs:
%    matchedX    - subset of sorted.X rows that had a CPT match within 1 year
%    matchedCPT  - corresponding CPT table rows (same row count as matchedX)
%    matchInfo   - table with columns: subject, sortedAge, cptAge, ageDiff
%                  documenting every match that was made
%
% Example:
%    sorted = sortDemographics(lh_thickness, 'demographics.mat');
%    CPT = readtable('CPT_Tscores.xlsx');
%    [mX, mCPT, info] = matchAndConcatCPT(sorted, CPT);
%    % mX and mCPT now have the same number of rows, matched by subject
%    % and closest age (within 1 year tolerance)

maxAgeDiff = 1; % maximum allowed age difference in years

%% auto-detect field names in sorted struct
sFields = fieldnames(sorted);

if nargin < 3 || isempty(subjField)
    idx = find(cellfun(@(f) ~isempty(regexpi(f,'subj')), sFields));
    if isempty(idx)
        error('matchAndConcatCPT:noSubjField', ...
            'No field containing ''subj'' found in sorted. Available: %s', ...
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

%% auto-detect column names in CPT table
cptCols = CPT_table_Tscore.Properties.VariableNames;

if nargin < 5 || isempty(cptSubjField)
    idx = find(cellfun(@(c) ~isempty(regexpi(c,'subj')), cptCols));
    if isempty(idx)
        error('matchAndConcatCPT:noCptSubj', ...
            'No column containing ''subj'' found in CPT table. Available: %s', ...
            strjoin(cptCols, ', '));
    end
    cptSubjField = cptCols{idx(1)};
end

if nargin < 6 || isempty(cptAgeField)
    idx = find(cellfun(@(c) ~isempty(regexpi(c,'age')), cptCols));
    if isempty(idx)
        error('matchAndConcatCPT:noCptAge', ...
            'No column containing ''age'' found in CPT table. Available: %s', ...
            strjoin(cptCols, ', '));
    end
    cptAgeField = cptCols{idx(1)};
end

%% extract vectors
sortedSubj = sorted.(subjField);
sortedAge  = sorted.(ageField);
cptSubj    = CPT_table_Tscore.(cptSubjField);
cptAge     = CPT_table_Tscore.(cptAgeField);

nSorted = length(sortedSubj);

%% match each row in sorted to the closest CPT row by age (within 1 year)
keepIdx     = false(nSorted, 1);  % which sorted rows have a match
cptMatchIdx = zeros(nSorted, 1);  % index into CPT table for each match
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
matchedX   = sorted.X(keepIdx, :);
matchedCPT = CPT_table_Tscore(cptMatchIdx(keepIdx), :);

matchInfo = table( ...
    sortedSubj(keepIdx), ...
    sortedAge(keepIdx), ...
    cptAge(cptMatchIdx(keepIdx)), ...
    ageDiffs(keepIdx), ...
    'VariableNames', {'subject', 'sortedAge', 'cptAge', 'ageDiff'});

nMatched  = sum(keepIdx);
nDropped  = nSorted - nMatched;
fprintf('Matched %d of %d rows (%d dropped, no CPT within %.0f year).\n', ...
    nMatched, nSorted, nDropped, maxAgeDiff);
end
