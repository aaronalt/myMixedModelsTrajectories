%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% example main script to use the mixed effect models toolbox on data stored
% in an examplary mat file
%
% This script gives an example for using the code to fit mixed models
% trajectories with input data from an mat file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% close and clear everything
clc
clear all
close all

% set up path to all necessary functions
addpath(genpath('./functions'));
startup; % set up some options for nicer plots

%% ------------------------------------------------------------------------
% set up all necessary otions here (input file, output directory, etc.)
% ------------------------------------------------------------------------
% tbl = table(input.subjID, input.age, input.grouping, input.data(:,1), input.data(:,2), 'VariableNames', {'subject', 'age', 'group', 'LH', 'RH'});
% 
% tbl.subject = categorical(tbl.subject);
% tbl.group = categorical(tbl.group);
% 
% % No covariates since already regressed out
% lme_lh = fitlme(tbl, 'LH ~ age*group + (1|subject)');
% lme_rh = fitlme(tbl, 'RH ~ age*group + (1|subject)');
% 
% fixedEffects(lme_rh)
% fixedEffects(lme_lh)
% lme_rh.Coefficients 
% lme_lh.Coefficients 
% 
% % Full model
% lme_full = fitlme(tbl, 'RH ~ age*group + (1|subject)');
% 
% % Reduced model (no group, no interaction)
% lme_nogroup = fitlme(tbl, 'RH ~ age + (1|subject)');
% 
% % LRT
% compare(lme_nogroup, lme_full)

% 
% unique(tbl.group)  % what values?
% tbl.group(1:10)  
% unique(dxcov)
% dxcov(1:10)
% -----------------------------
% input data options
% -----------------------------
load("/home/aaron/myMixedModelsTrajectories/demographics.mat")
load("/home/aaron/myMixedModelsTrajectories/clau_brain_vols.mat")
CPT = readtable('CPT_Tscores.mat');
sorted = sortDemographics(X, 'demographics.mat');


% --- 1. ROW FILTERING (Subjects) ---
nan_mask = isnan(diagnosis_bin);
vol_mask = X(:, 1) < 1000;
age_mask = age < 34;
keep_mask = ~nan_mask & ~vol_mask & age_mask;

% Apply row filter to all vectors immediately
idcov  = subject(keep_mask, :);   
agecov = age(keep_mask, :);              
dxcov  = diagnosis_bin(keep_mask, :);    
gender = gender_bin(keep_mask, :);  
tiv_orig = X(:, 16);
tivcov = tiv_orig(keep_mask, :);
X0 = X(keep_mask,:);

bad_subjects = [];
subjects = unique(idcov);
for s = 1:length(subjects)
    idx = find(idcov == subjects(s));
    if length(idx) > 1
        vols = X0(idx, 1);  % LH
        if (max(vols) - min(vols)) > 300
            bad_subjects = [bad_subjects; subjects(s)];
        end
    end
end
good_mask = ~ismember(idcov, bad_subjects);

idcov = idcov(good_mask);
agecov = agecov(good_mask);
dxcov = dxcov(good_mask);
gender = gender(good_mask);
tivcov = tivcov(good_mask);
X0 = X0(good_mask, :);

bad_subjects = [];
subjects = unique(idcov);
for s = 1:length(subjects)
    idx = find(idcov == subjects(s));
    if length(idx) > 1
        vols = X0(idx, 2);  % LH
        if (max(vols) - min(vols)) > 300
            bad_subjects = [bad_subjects; subjects(s)];
        end
    end
end
good_mask = ~ismember(idcov, bad_subjects);

idcov = idcov(good_mask);
agecov = agecov(good_mask);
dxcov = dxcov(good_mask);
gender = gender(good_mask);
gender = double(gender);
tivcov = tivcov(good_mask);
X0 = X0(good_mask, :);

input.subjID=idcov; % subject IDs
input.age=agecov; % age
input.grouping=dxcov;% column (with 0 and 1 values) you want to use for grouping: $
                          % input.grouping=[]; if you have only 1 group
                          % 1 column if you have 2 groups
                          % 2 colums if you have 3 groups (in the first column  you have 1 for everyone in the group 1 and 0 everywhere else,
                          % and in the second column you have 1 for everyone is in the group 2 and 0 everywhere else)  
input.data=compute_residuals_dimitri_2(X0(:,1:2),[tivcov, gender])                          
%input.data=X0(:,1:2);    % data to fit (thickness, volume, behavior, ...)
input.cov=[]; % model covariates (here, only sex is included as covariate)

[~, sort_idx] = sortrows([idcov, agecov]);
idcov = idcov(sort_idx);
agecov = agecov(sort_idx);
dxcov = dxcov(sort_idx);
gender = gender(sort_idx);
tivcov = tivcov(sort_idx);
X0 = X0(sort_idx, :);

% ---------------------------
% model estimation options
% -----------------------------
opts.orders=1; % model orders to check for (0=constant, 1=linear, 2=quadratic, 3=cubic, etc. ...)
opts.mType = 'slope'; % 'intercept' for random intercept, 'slope' for random slope (recommended)

% vertex IDs (colums of lh_thickness to analyze)
opts.vertID=[1, 2];

% -----------------------------
% model plotting options
% -----------------------------
outDir = fullfile('./results'); % directory where to store the result table and plots
saveResults = 2; % do you want to save the results? 
                 % 0=No; 1=Yes, but only the table; 2=Yes, both the table and the plots
plotOpts.legTxt = {'HC','22q'}; % legend: names of your groups (here 0='HC' and 1='Pat')
plotOpts.xLabel = 'age'; % label for x-axis
plotOpts.yLabel = 'claustrum volume'; % label for y-axis
plotOpts.plotCI = 1; % do you want to plot confidence intervals? 0=No; 1=Yes
plotOpts.plotType = 'redInter'; % which models do you want to plot?
        % 'full' - always plot full model, even if interation or intercept are not significantly different
        % 'redInter' - plot reduced model without interaction if interaction is not significant
        % 'redGrp' - plot reduced model without group effect if group effect is not significant 

% you can also try some different colors by uncommenting the next two lines :-)
% colors = cbrewer('qual', 'Set2', 3);
% plotOpts.plotCol = {colors(1,:),colors(2,:),colors(3,:)}; 


%% ------------------------------------------------------------------------
% execute the model estimation and plot/save results
% ------------------------------------------------------------------------

% -----------------------------
% run model fitting
% -----------------------------
% a few more options
opts.modelNames = textscan(num2str(opts.vertID),'%s'); 
opts.modelNames=opts.modelNames{1};
opts.alpha=0.05; % significance level for group and interaction effects
opts.figPosition=[440   488   525   310]; % change the size of your figure (take a figure, change the size to whatever you want, type get(gcf,'Position') and copy the resulting output here


% fit models
outModelVect = fitOptModel(input,opts);

% correct for multiple comparisons using FDR
outModelVect_corr = fdr_correct(outModelVect,opts.alpha);

% -----------------------------
% plot and save models
% -----------------------------
plotOpts.nCov=size(input.cov,2);
plotModelsAndSaveResults(outModelVect_corr,plotOpts,saveResults,outDir);

% -----------------------------
% calculation of the size effect
% -----------------------------
% Reporting the effect size for the groups and for the interaction between groups and age

effectSizeGroup=GroupCalculationEffect(outModelVect);
%table reporting the group (if variable group is included) size effect for each model (each response variable)
effectSizeInter=InterCalculationEffect(outModelVect);
%table reporting the interaction (if variable interaction is included) size effect for each model (each response variable)

figure; hold on;
subjects = unique(idcov);
colors = lines(2);
for s = 1:length(subjects)
    idx = idcov == subjects(s);
    grp = dxcov(idx); grp = grp(1) + 1;  % 1 or 2
    plot(agecov(idx), X0(idx,1), '-o', 'Color', colors(grp,:), 'MarkerSize', 3);
end
xlabel('Age'); ylabel('RH Claustrum Volume');
legend({'Control','VCFS'});