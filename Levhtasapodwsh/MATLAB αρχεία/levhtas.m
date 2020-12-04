[~,nSheets32]  = xlsfinfo('Dataslevhtas.xlsx');
Levhtas=[];

    N1=nSheets32{1};
    
    Levhtas=[Levhtas, xlsread('Dataslevhtas.xlsx',N1)];
    
sizeLevh=numel(Levhtas);
rLevh=numel(Levhtas(:,1));
cLevh=numel(Levhtas(1,:));

apodwshlevhta=[];
cp=[];
for i=1:rLevh
    cp(i,1)=(((Levhtas(i,39)*Levhtas(i,40)/3600)-Levhtas(i,41)-Levhtas(i,42))/(Levhtas(i,43)*(Levhtas(i,44))))*1000;
    apodwshlevhta(i,1)=(((Levhtas(i,39)*Levhtas(i,40))/3.6)-(Levhtas(i,41)*1000)-(cp(i,1)*Levhtas(i,43)*Levhtas(i,45)))/(((Levhtas(i,39)*Levhtas(i,40))/3.6)-(Levhtas(i,41)*1000));
    
end


newData = [Levhtas, apodwshlevhta];  % to append the new column with existing data.
xlswrite('Dataslevhtas.xlsx', newData, 'Fullo1', 'A5');  % to write new data into excel sheet.
winopen('Dataslevhtas.xlsx');   % to open excel file, just to check.