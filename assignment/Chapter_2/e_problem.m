%% E2-1 b
imageLena = double(imread('./IVC_labs_starting_point/data/images/lena.tif'));
imageSail = double(imread('./IVC_labs_starting_point/data/images/sail.tif'));
imageSmandril = double(imread('./IVC_labs_starting_point/data/images/smandril.tif'));

pmfLena = stats_marg(imageLena, 0:255);
HLena = calc_entropy(pmfLena);

pmfSail = stats_marg(imageSail, 0:255);
HSail = calc_entropy(pmfSail);

pmfSmandril = stats_marg(imageSmandril, 0:255);
HSmandril = calc_entropy(pmfSmandril);

fprintf('--------------Using individual code table--------------\n');
fprintf('lena.tif      H = %.2f bit/pixel\n', HLena);
fprintf('sail.tif      H = %.2f bit/pixel\n', HSail);
fprintf('smandril.tif  H = %.2f bit/pixel\n', HSmandril);

%% E2-1 c
imageLena = double(imread('./IVC_labs_starting_point/data/images/lena.tif'));
imageSail = double(imread('./IVC_labs_starting_point/data/images/sail.tif'));
imageSmandril = double(imread('./IVC_labs_starting_point/data/images/smandril.tif'));

pmfLena = stats_marg(imageLena, 0:255);
pmfSail = stats_marg(imageSail, 0:255);
pmfSmandril = stats_marg(imageSmandril, 0:255);

mergedPMF = (pmfLena + pmfSail + pmfSmandril)./3; % common code table

minCodeLengthLena = min_code_length(mergedPMF, pmfLena);
minCodeLengthSail = min_code_length(mergedPMF, pmfSail);
minCodeLengthSmandril = min_code_length(mergedPMF, pmfSmandril);

fprintf('--------------Using merged code table--------------\n');
fprintf('lena.tif      H = %.2f bit/pixel\n', minCodeLengthLena);
fprintf('sail.tif      H = %.2f bit/pixel\n', minCodeLengthSail);
fprintf('smandril.tif  H = %.2f bit/pixel\n', minCodeLengthSmandril);

%% E2-2 b
% Read Image
imageLena = double(imread('./IVC_labs_starting_point/data/images/lena.tif'));
% Calculate Joint PMF
jpmfLena = stats_joint(imageLena);
% Plot the joint histogram
[X, Y] = meshgrid(0:255);
figure(1);
mesh(X, Y, jpmfLena);
title('Joint Histogram-Lena (horizontal pixel pairs)');
% Calculate Joint Entropy
Hjoint = calc_entropy(jpmfLena);
fprintf('H_joint = %.2f bit/pixel pair\n', Hjoint);

%% E2-3 a
imageLena = double(imread('./IVC_labs_starting_point/data/images/lena.tif'));
H = stats_cond(imageLena);
% Plot the cond histogram
cpmfLena = condPMF(imageLena);
[X, Y] = meshgrid(0:255);
figure(1);
mesh(X, Y, cpmfLena);
title('Cond. Histogram-Lena (horizontal pixel pairs)');

fprintf('H_cond = %.2f bit/pixel\n',H);

%% E2-4 a
% Read Image
imageLena = double(imread('./IVC_labs_starting_point/data/images/lena.tif'));
% create the predictor and obtain the residual image
resImage = zeros(size(imageLena));
resImage(:,1,:) = imageLena(:,1,:); % copy the 1st col
a_1 = 1; % predictor coeff
for col=2:1:size(resImage,2)
    resImage(:,col,:) = imageLena(:,col-1,:).* a_1; % prediction
end
resImage = imageLena - resImage; % prediction error
% get the PMF of the residual image
% here in prediction error: not the normal range, it's [-146,145]
range = min(resImage(:)): max(resImage(:));
pmfRes = stats_marg(resImage, range);
% calculate the entropy of the residual image
H_res = calc_entropy(pmfRes);

fprintf('H_err_OnePixel   = %.2f bit/pixel\n',H_res);

%% E2-4 b
% Read Image
imageLena = double(imread('./IVC_labs_starting_point/data/images/lena.tif'));
imageLenaYCbCr = ictRGB2YCbCr(imageLena);

% create the predictor and obtain the residual image
resImage = zeros(size(imageLenaYCbCr));
recImage = zeros(size(imageLenaYCbCr));
resImage(:,1,:) = imageLenaYCbCr(:,1,:); % copy the 1st col
resImage(1,:,:) = imageLenaYCbCr(1,:,:); % copy the 1st row
recImage(:,1,:) = imageLenaYCbCr(:,1,:); % copy the 1st col
recImage(1,:,:) = imageLenaYCbCr(1,:,:); % copy the 1st row
sz = size(imageLenaYCbCr);
% Luminance Channel Prediction
coeff_luminance = [-1/2, 5/8; 7/8, 0];
for row=2:1:sz(1)
    for col=2:1:sz(2)
        recPatch = recImage(row-1:row, col-1:col, 1);
        prediction = sum(coeff_luminance .* recPatch, 'all');
        resImage(row,col,1) = round(imageLenaYCbCr(row,col,1) - prediction);
        recImage(row,col,1) = prediction + resImage(row,col,1);
    end
end
% Chrominance Channel Prediction
coeff_chrominance = [-1/2, 7/8; 5/8, 0];
for row=2:1:sz(1)
    for col=2:1:sz(2)
        for channel=2:3
        recPatch = recImage(row-1:row, col-1:col, channel);
        prediction = sum(coeff_chrominance .* recPatch, 'all');
        resImage(row,col,channel) = round(imageLenaYCbCr(row,col,channel) - prediction);
        recImage(row,col,channel) = prediction + resImage(row,col,channel);
        end
    end
end

% get the PMF of the residual image
% here in prediction error: not the normal range, it's [-146,145]
range = min(resImage(:)): max(resImage(:));
pmfRes = stats_marg(resImage, range);

% calculate the entropy of the residual image
H_res = calc_entropy(pmfRes);

fprintf('H_err_minEntropy   = %.2f bit/pixel\n',H_res);

%% E2-5
% Read Image
imageLena = double(imread('./IVC_labs_starting_point/data/images/lena_small.tif'));
imageLenaYCbCr = ictRGB2YCbCr(imageLena);

% create the predictor and obtain the residual image
resImage = zeros(size(imageLenaYCbCr));
recImage = zeros(size(imageLenaYCbCr));
resImage(:,1,:) = imageLenaYCbCr(:,1,:); % copy the 1st col
resImage(1,:,:) = imageLenaYCbCr(1,:,:); % copy the 1st row
recImage(:,1,:) = imageLenaYCbCr(:,1,:); % copy the 1st col
recImage(1,:,:) = imageLenaYCbCr(1,:,:); % copy the 1st row
sz = size(imageLenaYCbCr);
% Luminance Channel Prediction
coeff_luminance = [-1/2, 5/8; 7/8, 0];
for row=2:1:sz(1)
    for col=2:1:sz(2)
        recPatch = recImage(row-1:row, col-1:col, 1);
        prediction = sum(coeff_luminance .* recPatch, 'all');
        resImage(row,col,1) = round(imageLenaYCbCr(row,col,1) - prediction);
        recImage(row,col,1) = prediction + resImage(row,col,1);
    end
end
% Chrominance Channel Prediction
coeff_chrominance = [-1/2, 7/8; 5/8, 0];
for row=2:1:sz(1)
    for col=2:1:sz(2)
        for channel=2:3
        recPatch = recImage(row-1:row, col-1:col, channel);
        prediction = sum(coeff_chrominance .* recPatch, 'all');
        resImage(row,col,channel) = round(imageLenaYCbCr(row,col,channel) - prediction);
        recImage(row,col,channel) = prediction + resImage(row,col,channel);
        end
    end
end

% get the PMF of the residual image
range = -128:255;
pmfRes = stats_marg(resImage, range);

% Huffman Table
[BinaryTree, HuffCode, BinCode, Codelengths] = buildHuffman(pmfRes);
figure(1);
plot(1:length(Codelengths), Codelengths);
title('Codelengths in Huffman Table');

myTreeArray = getTreeArray(BinaryTree);
figure(2);
treeplot(myTreeArray);

%% E2-6
%% read image
imageLena = double(imread('./IVC_labs_starting_point/data/images/lena.tif'));
imageLena_small = double(imread('./IVC_labs_starting_point/data/images/lena_small.tif'));
imageLenaYCbCr = ictRGB2YCbCr(imageLena);
imageLenaSmallYCbCr = ictRGB2YCbCr(imageLena_small);

%% downsample
pad_pixel = 4;
I_CbCr_wrapped(:,:,1) = padarray(imageLenaYCbCr(:,:,2),[pad_pixel,pad_pixel],'symmetric');
I_CbCr_wrapped(:,:,2) = padarray(imageLenaYCbCr(:,:,3),[pad_pixel,pad_pixel],'symmetric');
for i=1:2
    wrapped = resample(I_CbCr_wrapped(:,:,i),1,2,3);
    I_CbCr_down(:,:,i) = (resample(wrapped',1,2,3))';
end
I_CbCr_down = I_CbCr_down(pad_pixel/2+1:end-pad_pixel/2,pad_pixel/2+1:end-pad_pixel/2,:);

%% residual calculation (original lena)
sz_img = size(imageLenaYCbCr);
sz_CbCr = size(I_CbCr_down);
resImage_Y = zeros(sz_img(1), sz_img(2));
recImage_Y = zeros(sz_img(1), sz_img(2));
resImage_Y(:,1) = imageLenaYCbCr(:,1,1); % copy the 1st col
resImage_Y(1,:) = imageLenaYCbCr(1,:,1); % copy the 1st row
recImage_Y(:,1) = imageLenaYCbCr(:,1,1); % copy the 1st col
recImage_Y(1,:) = imageLenaYCbCr(1,:,1); % copy the 1st row

resImage_CbCr = zeros(sz_CbCr(1), sz_CbCr(2), 2);
recImage_CbCr = zeros(sz_CbCr(1), sz_CbCr(2), 2);
resImage_CbCr(:,1,:) = I_CbCr_down(:,1,:); % copy the 1st col
resImage_CbCr(1,:,:) = I_CbCr_down(1,:,:); % copy the 1st row
recImage_CbCr(:,1,:) = I_CbCr_down(:,1,:); % copy the 1st col
recImage_CbCr(1,:,:) = I_CbCr_down(1,:,:); % copy the 1st row

% Luminance Channel Prediction
coeff_luminance = [-1/2, 5/8; 7/8, 0];
for row=2:1:sz_img(1)
    for col=2:1:sz_img(2)
        recPatch = recImage_Y(row-1:row, col-1:col);
        prediction = sum(coeff_luminance .* recPatch, 'all');
        resImage_Y(row,col) = round(imageLenaYCbCr(row,col) - prediction);
        recImage_Y(row,col) = prediction + resImage_Y(row,col);
    end
end

% Chrominance Channel Prediction
coeff_chrominance = [-1/2, 7/8; 5/8, 0];
for row=2:1:sz_CbCr(1)
    for col=2:1:sz_CbCr(2)
        for channel=1:2
            recPatch = recImage_CbCr(row-1:row, col-1:col, channel);
            prediction = sum(coeff_chrominance .* recPatch, 'all');
            resImage_CbCr(row,col,channel) = round(I_CbCr_down(row,col,channel) - prediction);
            recImage_CbCr(row,col,channel) = prediction + resImage_CbCr(row,col,channel);
        end
    end
end

%% Codebook Construction (using small Lena)
resImage_small = zeros(size(imageLenaSmallYCbCr));
recImage_small = zeros(size(imageLenaSmallYCbCr));
resImage_small(:,1,:) = imageLenaSmallYCbCr(:,1,:); % copy the 1st col
resImage_small(1,:,:) = imageLenaSmallYCbCr(1,:,:); % copy the 1st row
recImage_small(:,1,:) = imageLenaSmallYCbCr(:,1,:); % copy the 1st col
recImage_small(1,:,:) = imageLenaSmallYCbCr(1,:,:); % copy the 1st row
sz_small = size(imageLenaSmallYCbCr);
% Luminance Channel Prediction
coeff_luminance = [-1/2, 5/8; 7/8, 0];
for row=2:1:sz_small(1)
    for col=2:1:sz_small(2)
        recPatch = recImage_small(row-1:row, col-1:col, 1);
        prediction = sum(coeff_luminance .* recPatch, 'all');
        resImage_small(row,col,1) = round(imageLenaSmallYCbCr(row,col,1) - prediction);
        recImage_small(row,col,1) = prediction + resImage_small(row,col,1);
    end
end
% Chrominance Channel Prediction
coeff_chrominance = [-1/2, 7/8; 5/8, 0];
for row=2:1:sz_small(1)
    for col=2:1:sz_small(2)
        for channel=2:3
            recPatch = recImage_small(row-1:row, col-1:col, channel);
            prediction = sum(coeff_chrominance .* recPatch, 'all');
            resImage_small(row,col,channel) = round(imageLenaSmallYCbCr(row,col,channel) - prediction);
            recImage_small(row,col,channel) = prediction + resImage_small(row,col,channel);
        end
    end
end

% get the PMF of the residual image
% here in YCbCr: range is [-128,255] to let every possible sourve symbol is
% covered
range_small = -128:255;
pmfRes_small = stats_marg(resImage_small, range_small);

% Huffman Table
[BinaryTree, HuffCode, BinCode, Codelengths] = buildHuffman(pmfRes_small);

%% Encoding
resImage_Y = round(resImage_Y);
resImage_Cb = round(resImage_CbCr(:,:,1));
resImage_Cr = round(resImage_CbCr(:,:,2));
lowerBound = -128;

bytestream_Y = enc_huffman_new(resImage_Y(:)-lowerBound+1, BinCode, Codelengths);
bytestream_Cb = enc_huffman_new(resImage_Cb(:)-lowerBound+1, BinCode, Codelengths);
bytestream_Cr = enc_huffman_new(resImage_Cr(:)-lowerBound+1, BinCode, Codelengths);
bytestream = [bytestream_Y; bytestream_Cb; bytestream_Cr];

%% Decoding
decRes_Y = double(reshape(dec_huffman_new(bytestream_Y, BinaryTree,...
    max(size(resImage_Y(:)))), size(resImage_Y))) - 1 + lowerBound;
decRes_Cb = double(reshape(dec_huffman_new(bytestream_Cb, BinaryTree, ...
    max(size(resImage_Cb(:)))), size(resImage_Cb))) - 1 + lowerBound;
decRes_Cr = double(reshape(dec_huffman_new(bytestream_Cr, BinaryTree, ...
    max(size(resImage_Cr(:)))), size(resImage_Cr))) - 1 + lowerBound;
decRes_CbCr = cat(3, decRes_Cb, decRes_Cr);

%% Reconstruction
sz_img = size(imageLenaYCbCr);
sz_CbCr = size(I_CbCr_down);
decRecImage_Y = zeros(sz_img(1), sz_img(2));
decRecImage_Y(:,1) = imageLenaYCbCr(:,1,1); % copy the 1st col
decRecImage_Y(1,:) = imageLenaYCbCr(1,:,1); % copy the 1st row

decRecImage_CbCr = zeros(sz_CbCr(1), sz_CbCr(2), 2);
decRecImage_CbCr(:,1,:) = I_CbCr_down(:,1,:); % copy the 1st col
decRecImage_CbCr(1,:,:) = I_CbCr_down(1,:,:); % copy the 1st row

% Luminance Channel Prediction
coeff_luminance = [-1/2, 5/8; 7/8, 0];
for row=2:1:sz_img(1)
    for col=2:1:sz_img(2)
        recPatch = decRecImage_Y(row-1:row, col-1:col);
        prediction = sum(coeff_luminance .* recPatch, 'all');
        decRecImage_Y(row,col) = prediction + decRes_Y(row,col);
    end
end

% Chrominance Channel Prediction
coeff_chrominance = [-1/2, 7/8; 5/8, 0];
for row=2:1:sz_CbCr(1)
    for col=2:1:sz_CbCr(2)
        for channel=1:2
            recPatch = decRecImage_CbCr(row-1:row, col-1:col, channel);
            prediction = sum(coeff_chrominance .* recPatch, 'all');
            decRecImage_CbCr(row,col,channel) = prediction + decRes_CbCr(row,col,channel);
        end
    end
end

%% Upsample
I_CbCr_wrapped_2 = padarray(decRecImage_CbCr,[pad_pixel/2,pad_pixel/2],'symmetric');
for i=1:2
    wrapped = resample(I_CbCr_wrapped_2(:,:,i),2,1,3);
    wrapped = (resample(wrapped',2,1,3))';
    I_CbCr_up(:,:,i) = wrapped(pad_pixel+1:end-pad_pixel,pad_pixel+1:end-pad_pixel);
end
rec_image_YCbCr = cat(3, decRecImage_Y, I_CbCr_up(:,:,1), I_CbCr_up(:,:,2));
rec_image = ictYCbCr2RGB(rec_image_YCbCr);
rec_image = int64(rec_image);

%% evaluation and show results
figure
subplot(121)
imshow(uint8(imageLena)), title('Original Image')
subplot(122)

PSNR = calcPSNR(imageLena, rec_image);
imshow(uint8(rec_image)), title(sprintf('Reconstructed Image, PSNR = %.2f dB', PSNR))
BPP = numel(bytestream) * 8 / (numel(imageLena)/3);
CompressionRatio = 24/BPP;

fprintf('Bit Rate         = %.2f bit/pixel\n', BPP);
fprintf('CompressionRatio = %.2f\n', CompressionRatio);
fprintf('PSNR             = %.2f dB\n', PSNR);

%% E2-1 a
function pmf = stats_marg(image, range)
% Input:   image (original image)
%          range (range and step value of histogram calculaton, e.g. 0:255)
% Output:  pmf (probability mass function)
sz = size(image);
num = length(image(:)); % denominator: height*width*channel
image_new = reshape(image, sz(1).*sz(2), sz(3)); % [height*width, channel]
pmf = hist(image_new, range); % count PMF in individual channel
pmf = sum(pmf, 2)./num; % sum up the 3 values for one pixel, normalization
% pmf(pmf==0) = []; % easier for calculating entropy (avoid NaN)
end
%% E2-1 b
function H = calc_entropy(pmf)
% Input:   pmf (probability mass function)
% Output:  H (entropy in bits/pixel)
pmf_log = pmf .* log2(pmf);
H = nansum(pmf_log(:)); % avoid NaN (log2(0))
H = -H;
end
%% E2-1 c
function H = min_code_length(pmf_table, pmf_image)
% Input:   pmf_table (the pmf of a common code table, for our case here,
%                     it means the overall pmf of three images)
%          pmf_image (the pmf of a single image)
% Output:  H (minimum average code word length)
pmf_log = pmf_image .* log2(pmf_table);
H = nansum(pmf_log(:));
H = -H;
end
%% E2-2 a
function pmf = stats_joint(image)
% Input:  image (Original Image)
% Output: pmf   (Probability Mass Function)
sz = size(image);
N = 256;
pmf = zeros(N, N);
for channel=1:sz(3)
    for row=1:sz(1)
        for col=1:2:sz(2) % every two horizontally adjacent pixels, without overlapping
            intensity_left = image(row, col, channel);
            intensity_right = image(row, col+1, channel);
            pmf(intensity_left+1, intensity_right+1)= ...
                pmf(intensity_left+1, intensity_right+1) + 1;
        end
    end
end
num = sz(1)*sz(2)*sz(3)*0.5; % denominator: height*width*channel, joint pixles 1/2
pmf = pmf./num;
end
%% Self-Function: Cond. PMF
function pmf = condPMF(image)
% Input:  image (Original Image)
% Output: pmf   (Probability Mass Function)
sz = size(image);
N = 256;
pmf = zeros(N, N);
for channel=1:sz(3)
    for row=1:sz(1)
        for col=1:1:sz(2)-1 % every two horizontally adjacent pixels, with overlapping
            intensity_left = image(row, col, channel);
            intensity_right = image(row, col+1, channel);
            pmf(intensity_left+1, intensity_right+1)= ...
                pmf(intensity_left+1, intensity_right+1) + 1;
        end
    end
end
num = sz(1)*sz(2)*sz(3); % denominator: height*width*channel
pmf_joint = pmf./num;
pmf_cond = pmf_joint ./ sum(pmf_joint, 1);
end
%% E2-3 a
function H = stats_cond(image)
% Input:  image (Original Image)
% Output: H   (Conditional Entropy)
sz = size(image);
N = 256;
pmf = zeros(N, N);
for channel=1:sz(3)
    for row=1:sz(1)
        for col=1:1:sz(2)-1 % every two horizontally adjacent pixels, with overlapping
            intensity_left = image(row, col, channel);
            intensity_right = image(row, col+1, channel);
            pmf(intensity_left+1, intensity_right+1)= ...
                pmf(intensity_left+1, intensity_right+1) + 1;
        end
    end
end
num = sz(1)*sz(2)*sz(3); % denominator: height*width*channel
pmf_joint = pmf./num;
pmf_cond = pmf_joint ./ sum(pmf_joint, 1);

pmf_log = pmf_joint .* log2(pmf_cond);
H = nansum(pmf_log(:)); % avoid NaN (log2(0))
H = -H;
end
%% calc PSNR
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
%% ICT
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
%% Plot Cell Structure as a Tree
function treearray = getTreeArray(cellarray)
    % initialise the array construction from node 0
    treearray = [0, treebuilder(cellarray, 1)]; 
    % recursive tree building function, pass it a cell array and root node
    function [out, node] = treebuilder(cellarray, rnode)
        % Set up variables to be populated whilst looping
        out = []; 
        % Start node off at root node
        node = rnode;
        % Loop over cell array elements, either recurse or add node
        for ii = 1:numel(cellarray)
            tb = []; node = node + 1;
            if iscell(cellarray{ii})
                [tb, node] = treebuilder(cellarray{ii}, node);
            end
            out = [out, rnode, tb];   
        end
    end
end