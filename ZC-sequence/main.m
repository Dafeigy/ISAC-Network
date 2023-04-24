clear all;
close all;
%  Setting up
% ZC Sequence length
N  = 63;    % But 61 fuck up
% 2 root
M1 = 61;
M2 = 7;

% m denotes cyclic shift
for m = 0 : N-1
    % 1. Generating Zadoff-Chu
    ak1 = zeros(1,N);
    bk1 = zeros(1,N);
    for k = 0:N-1
        ak1(k+1) = exp(1j * (M1 * pi * k * (k+1) / N));
        bk1(k+1) = exp(1j * (M2 * pi * k * (k+1) / N));
    end
     
    % 2. cyclic shift
    ak2 = [ak1(1, m+1:end), ak1(1, 1:m)];
     
    % 3. autocorrelation
    autoCorr(m+1) = abs(sum(ak1 .* conj(ak2)) / N);
    % 4. cross correlation
    crossCorr(m+1) = abs(sum(ak1 .* conj(bk1)) / N);
end

plot(autoCorr);
hold on;
plot(crossCorr);
title("Cyclic shift v.s. cross/autoCorr");
xlabel("Cyclic Shift");
ylabel("Correlation value");
legend('autoCorrelation', 'crossCorrelation')

% m = 5;
% index = 5;
% for N = 7:61
%     ak1 = zeros(1,N);
%     bk1 = zeros(1,N);
%     for k = 0:N-1
%         ak1(k+1) = exp(1j * (M1 * pi * k * (k+1) / N));
%         bk1(k+1) = exp(1j * (M2 * pi * k * (k+1) / N));
%     end
%      
%     % 2. cyclic shift
%     ak2 = [ak1(1, m+1:end), ak1(1, 1:m)];
%      
%     % 3. autocorrelation
%     autoCorr(index+1) = abs(sum(ak1 .* conj(ak2)) / N);
%     % 4. cross correlation
%     crossCorr(index+1) = abs(sum(ak1 .* conj(bk1)) / N);
%     index = index +1 ;
% end
% 
% plot(autoCorr);
% hold on;
% plot(crossCorr);
% title("Sequence Length v.s. cross/autoCorr");
% xlabel("Sequence Length");
% ylabel("Correlation value");
% legend('autoCorrelation', 'crossCorrelation')









