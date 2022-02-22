figure(1);
plot(8, 15.46, '*', 'Color', 'b');
text(8, 15.46, '  Lena starting algorithm');
% hold on;
% plot(8, 17.0215, '*', 'Color', 'b');
% text(8, 17.0215, '  monarch starting algorithm');
hold on;
plot(8, 16.6239, '*', 'Color', 'b');
text(8, 16.6239, '  smandril starting algorithm');
hold on;
plot(8, 21.2, '*', 'Color', 'b');
text(8, 21.2, '  sail starting algorithm');
hold on;
plot(6, 33.677, '*', 'Color', 'r');
text(6, 33.677, '  Lena RGB-subsampling (CIF)');
% hold on;
% plot(6, 32.527, '*', 'Color', 'r');
% text(6, 32.527, '  monarch RGB-subsampling (CIF)');
hold on;
plot(6, 23.360, '*', 'Color', 'r');
text(6, 23.360, '  smandril RGB-subsampling (CIF)');
hold on;
plot(6, 28.959, '*', 'Color', 'r');
text(6, 28.959, '  sail RGB-subsampling (CIF)');
hold on;
plot(12, 38.36, '*', 'Color', 'g');
text(12, 38.36, '  Lena Chrominance subsampling');
% hold on;
% plot(12, 49.52, '*', 'Color', 'g');
% text(12, 49.52, '  monarch Chrominance subsampling');
hold on;
plot(12, 29.84, '*', 'Color', 'g');
text(12, 29.84, '  smandril Chrominance subsampling');
hold on;
plot(12, 48.50, '*', 'Color', 'g');
text(12, 48.50, '  sail Chrominance subsampling');
hold on;
plot(6.73, 38.27, '*', 'Color', 'm');
text(6.73, 38.27, '  Lena Prediction Huffman Coding');
hold on;
plot(5.5331, 34.3671, '*', 'Color', 'c');
text(5.5331, 34.3671, '  Lena VQ Huffman Coding');
hold on;
ylim([10 50])
xlim([0 20])
hold on;
grid on;
xlabel('bitrate [bit/pixel]');
ylabel('PSNR [dB]');
title('R-D Plot');