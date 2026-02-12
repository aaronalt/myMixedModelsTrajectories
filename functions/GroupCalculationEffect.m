function results = GroupCalculationEffect(outModelVect)
i=0;
for iM=1:size(outModelVect,1)
if isfield(outModelVect{iM,1}, 'groupEffect')
    newRow=table(iM, outModelVect{iM,1}.groupEffect.Chi2, outModelVect{iM,1}.groupEffect.dof_diff, outModelVect{iM,1}.groupEffect.p);
    newRow.Properties.VariableNames={'Number of model','Chi square statistics','Degree of fredoom','p-value'};
    if i==0
        effectSizeGroup=newRow;
    else
        effectSizeGroup=[effectSizeGroup;newRow];
    end
    i=i+1;
    results=effectSizeGroup;
end
end

if i==0
    results=[];
end


