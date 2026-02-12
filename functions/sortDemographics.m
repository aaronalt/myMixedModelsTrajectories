function sortedData = sortDemographics(X, matFile)
% sortDemographics - concatenate X with demographics and sort by subject ID
%                    then by age
%
% Syntax:  sortedData = sortDemographics(X, matFile)
%
% Inputs:
%    X       - data matrix (#observations x #features) to concatenate with
%              the demographics (e.g. imaging saliency data)
%    matFile - path to a .mat file containing demographics data. The file
%              must contain at least two variables interpretable as subject
%              ID and age. The function looks for variables whose names
%              contain 'subj' (for subject ID) and 'age' (for age).
%
% Outputs:
%    sortedData - struct with fields:
%       .X          - the X matrix, sorted to match the demographics order
%       .<demog>    - each variable from the demographics file, sorted
%                     first by subject ID (ascending) then by age (ascending)
%
% Example:
%    sorted = sortDemographics(lh_thickness, 'demographics.mat');
%    % sorted.X          -> thickness matrix sorted by subj then age
%    % sorted.age        -> age vector in the same sorted order
%    % sorted.subject_id -> subject IDs in the same sorted order

S = load(matFile);
fields = fieldnames(S);

% identify the subject ID variable
subjIdx = find(cellfun(@(f) ~isempty(regexpi(f,'subj')), fields));
if isempty(subjIdx)
    error('sortDemographics:noSubjID', ...
        'No variable containing ''subj'' found in %s. Available: %s', ...
        matFile, strjoin(fields,', '));
end
subjField = fields{subjIdx(1)};

% identify the age variable
ageIdx = find(cellfun(@(f) ~isempty(regexpi(f,'age')), fields));
if isempty(ageIdx)
    error('sortDemographics:noAge', ...
        'No variable containing ''age'' found in %s. Available: %s', ...
        matFile, strjoin(fields,', '));
end
ageField = fields{ageIdx(1)};

subjID = S.(subjField);
age = S.(ageField);
nRows = length(subjID);

% validate that X has the same number of rows as the demographics
if size(X, 1) ~= nRows
    error('sortDemographics:sizeMismatch', ...
        'X has %d rows but demographics has %d rows.', size(X,1), nRows);
end

% sortrows: primary key = subject ID, secondary key = age
[~, sortOrder] = sortrows([subjID(:), age(:)], [1, 2]);

% apply the sort order to X
sortedData.X = X(sortOrder, :);

% apply the sort order to every variable in the demographics file
for iF = 1:length(fields)
    val = S.(fields{iF});
    if isvector(val) && length(val) == nRows
        sortedData.(fields{iF}) = val(sortOrder);
    elseif ismatrix(val) && size(val,1) == nRows
        sortedData.(fields{iF}) = val(sortOrder, :);
    else
        % keep variables that don't match the expected row count unchanged
        sortedData.(fields{iF}) = val;
    end
end

fprintf('Concatenated X (%d x %d) with demographics and sorted %d rows by %s then %s.\n', ...
    size(X,1), size(X,2), nRows, subjField, ageField);
end
