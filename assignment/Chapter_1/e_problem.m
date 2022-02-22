%% E-1-1-de
Image = imread('./IVC_labs_starting_point/data/images/sail.tif');
recImage = imread('./IVC_labs_starting_point/data/reconstructed/sail_rec.tif');

MSE = calcMSE(Image, recImage);
fprintf('MSE is %.3f\n', MSE)
PSNR = calcPSNR(Image, recImage);
fprintf('PSNR is %.3f dB\n', PSNR);
subplot(121), imshow(Image), title('Original Image')
subplot(122), imshow(recImage), title('Reconstructed Image')
bit_rate = 8;

figure(2);
rd_plot = plot(bit_rate, PSNR, '*', 'Color', 'b');
ylim([10 20])
hold on;
grid on;
xlabel('bitrate [bit/pixel]');
ylabel('PSNR [dB]');
title('R-D Plot');
legend([rd_plot, rd_plot_2, rd_plot_3],'smandril', 'lena', 'monarch');
%% E-1-2&3
% Read Image
I = double(imread('./IVC_labs_starting_point/data/images/satpic1.bmp'));
kernel_gauss = [1,2,1; 2,4,2; 1,2,1];
kernel_fir = fir1(40, 0.5);
kernel_fir = kernel_fir' * kernel_fir;
kernel_norm = kernel_gauss ./ sum(kernel_gauss(:));
kernel_norm_fir = kernel_fir ./ sum(kernel_fir(:));

% plot frequency response of the filter
f = fft2(kernel_norm_fir);
imagesc(abs(fftshift(f)));
figure(1); freqz2(kernel);

% filter the test image and display the difference
subplot(121); imshow(I); title('Original Image');
I_pre = prefilterlowpass2d(I, kernel_norm);
subplot(122); imshow(I_pre); title('Filtered Image');

% without prefiltering
% YOUR CODE HERE
tic;
factor = 2;
sz = size(I);
for i=1:sz(3)
    I_down = downsample(I(:,:,i),factor,0);
    I_down = (downsample(I_down',factor,0))';
    I_up = upsample(I_down, 2, 0);
    I_up = (upsample(I_up',2,0))';
    I_up_notpre(:,:,i) = I_up;
end
I_post_notpre = prefilterlowpass2d(I_up_notpre, kernel_norm);
I_rec_notpre = 2 * factor * I_post_notpre;

% Evaluation without prefiltering
% I_rec_notpre is the reconstructed image WITHOUT prefiltering
PSNR_notpre = calcPSNR(I, I_rec_notpre);
fprintf('Reconstructed image, not prefiltered, PSNR = %.2f dB\n', PSNR_notpre)

% with prefiltering
% YOUR CODE HERE
factor = 2;
sz = size(I);
I_pre = prefilterlowpass2d(I, kernel_norm);
for i=1:sz(3)
    I_down = downsample(I_pre(:,:,i),factor,0);
    I_down = (downsample(I_down',factor,0))';
    I_up = upsample(I_down, 2, 0);
    I_up = (upsample(I_up',2,0))';
    I_up_pre(:,:,i) = I_up;
end
I_post_pre = prefilterlowpass2d(I_up_pre, kernel_norm);
I_rec_pre = 2 * factor * I_post_pre;

% Evaluation with prefiltering
% I_rec_pre is the reconstructed image WITH prefiltering
PSNR_pre = calcPSNR(I, I_rec_pre);
fprintf('Reconstructed image, prefiltered, PSNR = %.2f dB\n', PSNR_pre)
toc;

with prefiltering
YOUR CODE HERE
factor = 2;
sz = size(I);
I_pre = prefilterlowpass2d(I, kernel_norm_fir);
for i=1:sz(3)
    I_down = downsample(I_pre(:,:,i),factor,0);
    I_down = (downsample(I_down',factor,0))';
    I_up = upsample(I_down, 2, 0);
    I_up = (upsample(I_up',2,0))';
    I_up_pre(:,:,i) = I_up;
end
I_post_pre = prefilterlowpass2d(I_up_pre, kernel_norm_fir);
I_rec_pre = 2 * factor * I_post_pre;

Evaluation with prefiltering
I_rec_pre is the reconstructed image WITH prefiltering
PSNR_pre_FIR = calcPSNR(I, I_rec_pre);
fprintf('Reconstructed image, prefiltered, PSNR = %.2f dB\n', PSNR_pre_FIR)


figure(2);
rd_plot = plot(bit_rate, PSNR, '*', 'Color', 'b');
ylim([10 30])
hold on;
grid on;
rd_plot_2 = plot(bit_rate, PSNR_2, '*', 'Color', 'g');
rd_plot_3 = plot(bit_rate, PSNR_3, '*', 'Color', 'r');
rd_plot_4 = plot(bit_rate, PSNR_notpre, '*', 'Color', 'k');
rd_plot_5 = plot(bit_rate, PSNR_pre, '*', 'Color', 'm');
rd_plot_6 = plot(bit_rate, PSNR_pre_FIR, '*', 'Color', '#EDB120');
xlabel('bitrate [bit/pixel]');
ylabel('PSNR [dB]');
title('R-D Plot');
legend([rd_plot, rd_plot_2, rd_plot_3, rd_plot_4, rd_plot_5, rd_plot_6],'smandril', 'lena', 'monarch', 'non-prefiltered satpic', 'Gaussian-prefiltered satpic', 'FIR40-prefiltered satpic');
% E-1-4
% image read
I_lena = double(imread('./IVC_labs_starting_point/data/images/monarch.tif'));
I_sail = double(imread('./IVC_labs_starting_point/data/images/smandril.tif'));

% Wrap Round
% YOUR CODE HERE
pad_pixel = 4;
I_lena_wrapped = padarray(I_lena,[pad_pixel,pad_pixel],'symmetric');
I_sail_wrapped = padarray(I_sail,[pad_pixel,pad_pixel],'symmetric');

% Resample(subsample)
% YOUR CODE HERE
sz = size(I_lena);
for i=1:sz(3)
    lena = resample(I_lena_wrapped(:,:,i),1,2,3);
    I_lena_down(:,:,i) = (resample(lena',1,2,3))';
    sail = resample(I_sail_wrapped(:,:,i),1,2,3);
    I_sail_down(:,:,i) = (resample(sail',1,2,3))';
end

% Crop Back
% YOUR CODE HERE
I_lena_down = I_lena_down(pad_pixel/2+1:end-pad_pixel/2,pad_pixel/2+1:end-pad_pixel/2,:);
I_sail_down = I_sail_down(pad_pixel/2+1:end-pad_pixel/2,pad_pixel/2+1:end-pad_pixel/2,:);

% Wrap Round
% YOUR CODE HERE
I_lena_wrapped_2 = padarray(I_lena_down,[pad_pixel/2,pad_pixel/2],'symmetric');
I_sail_wrapped_2 = padarray(I_sail_down,[pad_pixel/2,pad_pixel/2],'symmetric');

% Resample (upsample)
% YOUR CODE HERE
for i=1:sz(3)
    lena = resample(I_lena_wrapped_2(:,:,i),2,1,3);
    I_lena_up(:,:,i) = (resample(lena',2,1,3))';
    sail = resample(I_sail_wrapped_2(:,:,i),2,1,3);
    I_sail_up(:,:,i) = (resample(sail',2,1,3))';
end

% Crop back
% YOUR CODE HERE
I_rec_lena = I_lena_up(pad_pixel+1:end-pad_pixel,pad_pixel+1:end-pad_pixel,:);
I_rec_sail = I_sail_up(pad_pixel+1:end-pad_pixel,pad_pixel+1:end-pad_pixel,:);

% Distortion Analysis
PSNR_lena = calcPSNR(I_lena, I_rec_lena);
PSNR_sail = calcPSNR(I_sail, I_rec_sail);
fprintf('PSNR lena subsampling = %.3f dB\n', PSNR_lena)
fprintf('PSNR sail subsampling = %.3f dB\n', PSNR_sail)
subplot(121), imshow(I_sail./255), title('Original Image')
subplot(122), imshow(I_rec_sail./255), title('Reconstructed')
%% E-1-5
imageLena = double(imread('./IVC_labs_starting_point/data/images/lena.tif'));
I_YCbCr = ictRGB2YCbCr(imageLena);

subplot(221),imshow(I_YCbCr); title('YCbCr Color Space')
subplot(222),imshow(I_YCbCr(:,:,1),[]);title('Luminance Component')
subplot(223),imshow(I_YCbCr(:,:,2),[]);title('Chrominance Component Cb')
subplot(224),imshow(I_YCbCr(:,:,3),[]);title('Chrominance Component Cr')

I_RGB = ictYCbCr2RGB(I_YCbCr);

subplot(221),imshow(uint8(I_RGB)); title('RGB Color Space')
subplot(222),imshow(I_RGB(:,:,1),[]);title('R Component')
subplot(223),imshow(I_RGB(:,:,2),[]);title('G Component')
subplot(224),imshow(I_RGB(:,:,3),[]);title('B Component')
%% E-1-6
% read original RGB image
I_ori = imread('./IVC_labs_starting_point/data/images/sail.tif');

% YOUR CODE HERE for chroma subsampling
I_ori = double(I_ori);
I_YCbCr = ictRGB2YCbCr(I_ori);
pad_pixel = 4;
I_YCbCr_wrapped(:,:,1) = padarray(I_YCbCr(:,:,2),[pad_pixel,pad_pixel],'symmetric');
I_YCbCr_wrapped(:,:,2) = padarray(I_YCbCr(:,:,3),[pad_pixel,pad_pixel],'symmetric');
for i=1:2
    wrapped = resample(I_YCbCr_wrapped(:,:,i),1,2,3);
    I_YCbCr_down(:,:,i) = (resample(wrapped',1,2,3))';
end
I_YCbCr_down = I_YCbCr_down(pad_pixel/2+1:end-pad_pixel/2,pad_pixel/2+1:end-pad_pixel/2,:);
I_YCbCr_wrapped_2 = padarray(I_YCbCr_down,[pad_pixel/2,pad_pixel/2],'symmetric');
for i=1:2
    wrapped = resample(I_YCbCr_wrapped_2(:,:,i),2,1,3);
    wrapped = (resample(wrapped',2,1,3))';
    I_YCbCr_up(:,:,i) = wrapped(pad_pixel+1:end-pad_pixel,pad_pixel+1:end-pad_pixel);
end
I_YCbCr_rec = cat(3, I_YCbCr(:,:,1), I_YCbCr_up(:,:,1), I_YCbCr_up(:,:,2));
I_rec = ictYCbCr2RGB(I_YCbCr_rec);
I_rec = int64(I_rec);

% Evaluation
% I_rec is the reconstructed image in RGB color space
PSNR = calcPSNR(I_ori, I_rec)
fprintf('PSNR is %.2f dB\n', PSNR);
%% E-1-1-de
function PSNR = calcPSNR(Image, recImage)
% Input         : Image    (Original Image)
%                 recImage (Reconstructed Image)
%
% Output        : PSNR     (Peak Signal to Noise Ratio)
% call calcMSE to calculate MSE
MSE = calcMSE(Image, recImage);
max_intensity = 2^8 - 1;
PSNR = 10*log10(max_intensity.^2/MSE);
end
function MSE = calcMSE(Image, recImage)
% Input         : Image    (Original Image)
%                 recImage (Reconstructed Image)
% Output        : MSE      (Mean Squared Error)
img_double = double(Image);
recImg_double = double(recImage);
img_diff = img_double - recImg_double;
mse = img_diff.^2;
MSE = mean(mse(:));
end
%% E-1-2
function pic_pre = prefilterlowpass2d(picture, kernel)
sz = size(picture);
% kernel_pad = padarray(kernel, ([size(picture,1) size(picture,2)]-size(kernel)), 'post');
% kernel_pad = circshift(kernel_pad, -floor(size(kernel)/2));
% kernel_freq = fft2(kernel_pad);
for i=1:sz(3)
    %     pic_freq = fft2(picture(:,:,i));
    %     pic_pre(:,:,i) = ifft2(pic_freq .* kernel_freq);
    pic_pre(:,:,i) = conv2(picture(:,:,i), kernel, 'same');
end
end
%% E-1-5
function yuv = ictRGB2YCbCr(rgb)
% Input         : rgb (Original RGB Image)
% Output        : yuv (YCbCr image after transformation)
% YOUR CODE HERE
yuv(:,:,1) = 0.299*rgb(:,:,1) + 0.587*rgb(:,:,2) + 0.114*rgb(:,:,3);
yuv(:,:,2) = -0.169*rgb(:,:,1) - 0.331*rgb(:,:,2) + 0.5*rgb(:,:,3);
yuv(:,:,3) = 0.5*rgb(:,:,1) - 0.419*rgb(:,:,2) - 0.081*rgb(:,:,3);
end
function rgb = ictYCbCr2RGB(yuv)
% Input         : yuv (Original YCbCr Image)
% Output        : rgb (RGB image after transformation)
% YOUR CODE HERE
rgb(:,:,1) = yuv(:,:,1) + 1.402*yuv(:,:,3);
rgb(:,:,2) = yuv(:,:,1) - 0.344*yuv(:,:,2) - 0.714*yuv(:,:,3);
rgb(:,:,3) = yuv(:,:,1) + 1.772*yuv(:,:,2);
end