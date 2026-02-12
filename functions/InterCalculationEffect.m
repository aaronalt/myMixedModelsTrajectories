function results = InterCalculationEffect(outModelVect)
i=0;
for iM=1:size(outModelVect,1)
if isfield(outModelVect{iM,1}, 'interEffect')
    newRow=table(iM, outModelVect{iM,1}.interEffect.Chi2, outModelVect{iM,1}.interEffect.dof_diff, outModelVect{iM,1}.interEffect.p);
    newRow.Properties.VariableNames={'Number of model','Chi square statistics','Degree of fredoom','p-value'};
    if i==0
        effectSizeInter=newRow;
    else
        effectSizeInter=[effectSizeInter;newRow];
    end
    i=i+1;
    results=effectSizeInter;
end
end

if i==0
    results=[];
end
