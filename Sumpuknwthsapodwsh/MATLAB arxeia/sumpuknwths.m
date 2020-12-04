[~,nSheets32]  = xlsfinfo('DATA.xlsx');
Sumpuknwths=[];

    N1=nSheets32{1};
    
    Sumpuknwths=[Sumpuknwths, xlsread('DATA.xlsx',N1)];
    
sizeSump=numel(Sumpuknwths);
rSump=numel(Sumpuknwths(:,1));
cSump=numel(Sumpuknwths(1,:));

max=Sumpuknwths(1,5)-Sumpuknwths(1,4);

for i=1:rSump
    if (Sumpuknwths(i,5)-Sumpuknwths(i,4))>max
        max=Sumpuknwths(i,5)-Sumpuknwths(i,4);
    end
end

apodwshsumpuknwth=[1;0];
for i=1:rSump
    apodwshsumpuknwth(i,1)=(Sumpuknwths(i,5)-Sumpuknwths(i,4))/max;
end

newData = [Sumpuknwths, apodwshsumpuknwth];  % to append the new column with existing data.
xlswrite('DATA.xlsx', newData, 'Fullo1', 'A5');  % to write new data into excel sheet.
winopen('DATA.xlsx');   % to open excel file, just to check.