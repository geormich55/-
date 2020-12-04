[num,txt,raw] = xlsread('Datas.xlsx');
%for i=20:20:2900
%x=cat(1,x,num(i-19:i,1));
%end

x=[];
for i=20:20:300
x=cat(1,x,num(i-19:i,1));
end

figure(1)
plot(x)
hline1 = refline([0 max(x)]);
hline1.Color='r';
hline2 = refline([0 min(x)]);
hline2.Color= 'y';

figure(1)
title('HeatRate')
xlabel('Observations') 
ylabel('HeatRate(MJ/MWHR)')
legend({'main graph','max(Fullo1(:,48))','min(Fullo1(:,48))'},'Location','southwest')

x=[];
for i=20:20:300
x=cat(1,x,num(i-19:i,2));
end

figure(2)
plot(x)
hline1 = refline([0 max(x)]);
hline1.Color='r';
hline2 = refline([0 min(x)]);
hline2.Color= 'y';

figure(2)
title('VariableCost')
xlabel('Observations') 
ylabel('VC')
legend({'main graph','max(Fullo1(:,49))','min(Fullo1(:,49))'},'Location','southwest')


x1=num(1:20,1);
x2=num(21:40,1);
x3=num(41:60,1);
x4=num(61:80,1);
x5=num(81:100,1);
x6=num(101:120,1);
x7=num(121:140,1);
x8=num(141:160,1);
x9=num(161:180,1);
x10=num(181:200,1);
x11=num(201:220,1);
x12=num(221:240,1);
x13=num(241:260,1);
x14=num(261:280,1);
x15=num(281:300,1);
x16=num(301:320,1);
x17=num(321:340,1);
x18=num(341:360,1);
x19=num(361:380,1);
x20=num(381:400,1);
x21=num(401:420,1);
x22=num(421:440,1);
x23=num(441:460,1);
x24=num(461:480,1);
x25=num(481:500,1);
x26=num(501:520,1);
x27=num(521:540,1);
x28=num(541:560,1);
x29=num(561:580,1);
x30=num(581:600,1);
x31=num(601:620,1);
x32=num(621:640,1);
x33=num(641:660,1);
x34=num(661:680,1);
x35=num(681:700,1);
x36=num(701:720,1);
x37=num(721:740,1);
x38=num(741:760,1);
x39=num(761:780,1);
x40=num(781:800,1);
x41=num(801:820,1);
x42=num(821:840,1);
x43=num(841:860,1);
x44=num(861:880,1);
x45=num(881:900,1);
x46=num(901:920,1);
x47=num(921:940,1);
x48=num(941:960,1);
x49=num(961:980,1);
x50=num(981:1000,1);
x=[];
x=cat(1,x,x1);
x=cat(1,x,x2);
x=cat(1,x,x3);
x=cat(1,x,x4);
x=cat(1,x,x5);
x=cat(1,x,x6);
x=cat(1,x,x7);
x=cat(1,x,x8);
x=cat(1,x,x9);
x=cat(1,x,x10);
x=cat(1,x,x11);
x=cat(1,x,x12);
x=cat(1,x,x13);
x=cat(1,x,x14);
x=cat(1,x,x15);
%x=cat(1,x,x16);



