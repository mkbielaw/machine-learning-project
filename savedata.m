%RICIAN
clearvars; close all; clc;

%simulation parameters
%shared parameters
simtime = 6; %simulation time
sampleRate = 2e4; %fading channel sample rate
MDs = 200; %max Doppler shift [Hz]
EbNo = 4; %signal-noise ratio
sSeed = 1; %source seed

%exclusive params
Kfac = 3; %K-factor
rayDelay = [0]; %discrete path delays
rayGain = [0]; %average path gains
%simulation launch
out=sim('Fading_Rician.slx');
pause(1);
%histogram(out.noNoise);
%histogram(out.withNoise);
noNoise = out.withNoise(:);
rician00 = noNoise;

%PACKET LEVEL NOISE
%{
M=183;
V = ones([length(noNoise),M]);
for i = 0:length(noNoise)-1
    V(i+1,1:M)=V(i+1,1:M)*noNoise(i+1);
end

cutWithNoise = awgn(V,EbNo,'measured');
rician00 = mean(cutWithNoise,2);
%}
%{
figure(4);histogram(noNoise);title('without noise');
figure(5);histogram(rician00);title('averaged with noise');
figure(6);hold on; plot(noNoise,'b'); plot(rician00,'r'); hold off;
%}
%PACKET LEVEL NOISE END

%figure(1);histogram(rician00);title('20 samples'); xlabel('Amplitude'); ylabel('number of samples');

%__________________________________________________________________________

%RAYLEIGH
clearvars -except rician00 simtime sampleRate MDs EbNo sSeed noiseRate; clc;

%simulation parameters
rayDelay = [0]; %[0] %[0 5] %[5 2 8 9]
rayGain = [-9]; %[10]
%simulation launch
out=sim('Fading_Rayleigh.slx');
pause(1);
%histogram(out.noNoise);
%histogram(out.withNoise);
noNoise = out.withNoise(:);

rayleigh09 = noNoise;

%PACKET LEVEL NOISE
%{
M=183;
V = ones([length(noNoise),M]);
for i = 0:length(noNoise)-1
    V(i+1,1:M)=V(i+1,1:M)*noNoise(i+1);
end

cutWithNoise = awgn(V,EbNo,'measured');
rayleigh09 = mean(cutWithNoise,2);
%}
%{
figure(7);histogram(noNoise);title('without noise');
figure(8);histogram(rayleigh09);title('averaged with noise');
figure(9);hold on; plot(noNoise,'b'); plot(rayleigh09,'r'); hold off;
%}
%PACKET LEVEL NOISE END


%figure(2);histogram(rayleigh09);title('20 samples'); xlabel('Amplitude'); ylabel('number of samples');

%{
rayDist = fitdist(rayleigh09,'Rayleigh');
ricDist = fitdist(rician00,'Rician');
x = 0:0.01:2.5;
ricPdf = pdf(ricDist,x);
rayPdf = pdf(rayDist,x);
%}
%distributions comparison

[N,edges] = histcounts(rician00);
edges = edges(2:end) - (edges(2)-edges(1))/2;
[N2,edges2] = histcounts(rayleigh09);
edges2 = edges2(2:end) - (edges2(2)-edges2(1))/2;
%ricRatio = max(N)/max(ricPdf);
%rayRatio = max(N2)/max(rayPdf);
figure(3);hold on;
plot(edges,N,'r','LineWidth',1);
plot(edges2, N2,'b','LineWidth',1);
legend('Rician','Rayleigh');
%plot(x,ricPdf*ricRatio,'r--','LineWidth',1);
%plot(x,rayPdf*rayRatio,'b--','LineWidth',1);
hold off;
















%{
M=183;
V = ones([length(noNoise),M]);
for i = 0:length(noNoise)-1
    V(i+1,1:M)=V(i+1,1:M)*noNoise(i+1);
end

cutWithNoise = awgn(V,12,'measured');
averagedWithNoise = mean(cutWithNoise,2);
figure(1);histogram(noNoise);title('without noise');
figure(2);histogram(averagedWithNoise);title('averaged with noise');
figure(3);hold on; plot(noNoise,'b'); plot(averagedWithNoise,'r'); hold off;
%}

%_________________________________________________________________________________
%sampledNoNoise = noNoise(293:586:end);

%cutWithNoise = reshape(withNoise(1:numel(withNoise)-rem(numel(withNoise),586)), [586, (numel(withNoise)-rem(numel(withNoise),586))/586]);
%%cut data into samples of 568 symbols (without the last sample that is not
%%a full packet of 568 symbols
%averagedWithNoise = mean(cutWithNoise,1);
%averagedWithNoise = averagedWithNoise';

%figure(1);histogram(sampledNoNoise);title('without noise');
%figure(2);histogram(averagedWithNoise);title('averaged with noise');
%___________________________________________________


%{
S = [];
O = [];


for i=1:length(out.tout)
    O = [O;out.SISOout(:,:,i)];
    S = [S;out.binS(:,:,i)];
end

%S = out.binS.Data;
%O = out.SISOout.Data;
%A = out.SISOgain.Data;
t = out.tout;
Or = real(O);
Oi = imag(O);

filename = 'testdata.xls';
Results_Params ={'delay [0 1/2.4e9 2/2.4e9 3/2.4e9 4/2.4e9]','gain [0 -3 -6 -9 -12]'};
Results_Names={'real Output','imag Output'};
Results_Values=[Or,Oi];
sheet=1;
xlRange='K3'; %next letters for next set of samples
xlswrite(filename,Results_Values,sheet,xlRange);

xlRange='K2'; %next letters for next set of samples
xlswrite(filename,Results_Names,sheet,xlRange);

xlRange='K1'; %next letters for next set of samples
xlswrite(filename,Results_Params,sheet,xlRange);
%}