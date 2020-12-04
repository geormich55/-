[~,nSheets32]  = xlsfinfo('Dataslevhtas-total-HR.xlsx');
Fullo1=[];
N1=nSheets32{1};
Fullo1=[Fullo1, xlsread('Dataslevhtas-total-HR.xlsx',N1)];
sizeSump=numel(Fullo1);
rSump=numel(Fullo1(:,1));
cSump=numel(Fullo1(1,:));

VC=[];
for i=1:rSump
    VC(i,1)=(Fullo1(i,48)*Fullo1(i,41))/(Fullo1(i,39));
    
end

newData = [Fullo1, VC];  % to append the new column with existing data.
xlswrite('Dataslevhtas-total-HR.xlsx', newData, 'Fullo1', 'A5');  % to write new data into excel sheet.
winopen('Dataslevhtas-total-HR.xlsx');   % to open excel file, just to check.

figure(1)
plot(Fullo1(:,48))
hline1 = refline([0 max(Fullo1(:,48))]);
hline1.Color='r';
hline2 = refline([0 min(Fullo1(:,48))]);
hline2.Color= 'y';

figure(2)
plot(Fullo1(:,47))
hline1 = refline([0 max(Fullo1(:,47))]);
hline1.Color='r';
hline2 = refline([0 min(Fullo1(:,47))]);
hline2.Color= 'y';

figure(3)
plot(Fullo1(:,46))
hline1 = refline([0 max(Fullo1(:,46))]);
hline1.Color='r';
hline2 = refline([0 min(Fullo1(:,46))]);
hline2.Color= 'y';

figure(4)
plot(Fullo1(:,49))
hline1 = refline([0 max(Fullo1(:,49))]);
hline1.Color='r';
hline2 = refline([0 min(Fullo1(:,49))]);
hline2.Color= 'y';

figure(1)
title('HeatRate')
xlabel('Observations') 
ylabel('HeatRate(MJ/MWHR)')
legend({'main graph','max(Fullo1(:,48))','min(Fullo1(:,48))'},'Location','southwest')
figure(2)
title('CONDENSER')
xlabel('Observations') 
ylabel('Condenserefficiency')
legend({'main graph','max(Fullo1(:,47))','min(Fullo1(:,47))'},'Location','southwest')
figure(3)
title('HSRG')
xlabel('Observations') 
ylabel('HSRGefficiency')
legend({'main graph','max(Fullo1(:,46))','min(Fullo1(:,46))'},'Location','southwest')
figure(4)
title('VariableCost')
xlabel('Observations') 
ylabel('VC')
legend({'main graph','max(Fullo1(:,49))','min(Fullo1(:,49))'},'Location','southwest')




