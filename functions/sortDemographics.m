function sortedData = sortDemographics(matFile)
% sortDemographics - sort demographics data by subject ID then by age
%
% Syntax:  sortedData = sortDemographics(matFile)
%
% Inputs:
%    matFile - path to a .mat file containing demographics data. The file
%              must contain at least two variables interpretable as subject
%              ID and age. The function looks for variables whose names
%              contain 'subj' (for subject ID) and 'age' (for age).
%
% Outputs:
%    sortedData - struct with the same variables as the input file, sorted
%                 first by subject ID (ascending) then by age (ascending)
%
% Example:
%    sorted = sortDemographics('demographics.mat');

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

% sortrows: primary key = subject ID, secondary key = age
[~, sortOrder] = sortrows([subjID(:), age(:)], [1, 2]);

% apply the sort order to every variable in the file
sortedData = struct();
for iF = 1:length(fields)
    val = S.(fields{iF});
    if isvector(val) && length(val) == length(sortOrder)
        sortedData.(fields{iF}) = val(sortOrder);
    elseif ismatrix(val) && size(val,1) == length(sortOrder)
        sortedData.(fields{iF}) = val(sortOrder, :);
    else
        % keep variables that don't match the expected row count unchanged
        sortedData.(fields{iF}) = val;
    end
end

fprintf('Sorted %d rows by %s then %s.\n', length(sortOrder), subjField, ageField);
end
