def compute_residuals(data,covariates):
    num_of_subjects = data.shape()
    num_of_regions = data.shape[1]
    num_of_covariates = covariates.shape[1]

    return residuals, beta

# function [residuals,beta]=compute_residuals_2(data,covariates)

# num_of_subjects=size(data,1);
# num_of_regions=size(data,2);
# num_of_covariates=size(covariates,2);

data=reshape(data,[numel(data) 1]);

% centering of covariates (demeaning)
covariates=covariates-repmat(mean(covariates,1),[size(covariates,1) 1]);

% add mean
covariates=[covariates ones(size(covariates,1),1)];

X=kron(eye(num_of_regions),covariates);

beta=pinv(X)*data;

% only remove covariates, not means
%beta(size(covariates,2):size(covariates,1):end)=0; % old (and wrong)
beta(num_of_covariates+1:num_of_covariates+1:end)=0;
residuals=data-X*beta;

residuals=reshape(residuals,[num_of_subjects num_of_regions]);
