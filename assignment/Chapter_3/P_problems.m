D = 0:0.001:sqrt(2);
R = 0.5 * log2(sqrt(2)./D);
SNR = 10 * log10(sqrt(2)./D);

figure(1);
plot(R, SNR);

figure(2);
plot(R, D);