%% E3.1 a
imageLena_small = double(imread('./IVC_labs_starting_point/data/images/lena_small.tif'));
imageLena = double(imread('./IVC_labs_starting_point/data/images/lena.tif'));

qImage = {};
qImage_small = {};
bits = 1 : 1 : 7;
for bit = bits
    qImage{end+1} = UniQuant(imageLena, bit);
    qImage_small{end+1} = UniQuant(imageLena_small, bit);
end
fprintf("The syntax of the code seems to be correct,next run the assessment to verify the correctness");

%% E3.1 b
% load('qImage.mat')
% load('qImage_small.mat')

recImage = {};
recImage_small = {};
bits = 1 : 1 : 7;
for bit = bits
    recImage{end+1} = InvUniQuant(qImage{bit}, bit);
    recImage_small{end+1} = InvUniQuant(qImage_small{bit}, bit);
end
fprintf("The syntax of the code seems to be correct, next run the assessment to verify the correctness");

%% E3.1 c
imageLena_small = double(imread('./IVC_labs_starting_point/data/images/lena_small.tif'));
imageLena = double(imread('./IVC_labs_starting_point/data/images/lena.tif'));
bits_small = [1 2 3 5 7];
bits = [3 5];
PSNR_small = [];
for bit = bits_small
    qImageLena_small = UniQuant(imageLena_small, bit);
    recImage_small = InvUniQuant(qImageLena_small, bit);
    PSNR_small = [PSNR_small calcPSNR(imageLena_small, recImage_small)];
end

PSNR = [];
for bit = bits
    qImageLena = UniQuant(imageLena, bit);
    recImage = InvUniQuant(qImageLena, bit);
    PSNR = [PSNR calcPSNR(imageLena, recImage)];
end

%% E3.2
imageLena_small = double(imread('./IVC_labs_starting_point/data/images/lena_small.tif'));
imageLena = double(imread('./IVC_labs_starting_point/data/images/lena.tif'));

bits = 3;
epsilon = 0.001; %% stopping criterion
for bit = bits
    [qImage, clusters] = LloydMax(imageLena, bit, epsilon);
    [qImage_small, clusters_small] = LloydMax(imageLena_small, bit, epsilon);
end
% fprintf("The syntax of the code seems to be correct, next run the assessment to verify the correctness");

%% E3.3 a-1
% load('qImage.mat');
% load('qImage_small.mat');
% load('clusters.mat');
% load('clusters_small.mat');

recImage = InvLloydMax(qImage, clusters);
recImage_small = InvLloydMax(qImage_small, clusters_small);

fprintf("The syntax of the code seems to be correct, next run the assessment to verify the correctness");

%% E3.3 a-2
epsilon = 0.001;

imageLena_small = double(imread('./IVC_labs_starting_point/data/images/lena_small.tif'));
[qImageLena_small clusters_small] = LloydMax(imageLena_small, 3, epsilon);
recImage_small = InvLloydMax(qImageLena_small, clusters_small);
PSNR_small = calcPSNR(imageLena_small, recImage_small);

imageLena = double(imread('./IVC_labs_starting_point/data/images/lena.tif'));
[qImageLena clusters] = LloydMax(imageLena, 3, epsilon);
recImageLena = InvLloydMax(qImageLena, clusters);
PSNR = calcPSNR(imageLena, recImageLena);

%% E3.4 a
imageLena_small = double(imread('./IVC_labs_starting_point/data/images/lena_small.tif'));
imageLena = double(imread('./IVC_labs_starting_point/data/images/lena.tif'));
bits = 8;
epsilon = 0.1;
block_size = 2;
[clusters, Temp_clusters] = VectorQuantizer(imageLena_small, bits, epsilon, block_size);
% Test your function
qImage_small = ApplyVectorQuantizer(imageLena_small, clusters, block_size);
qImage = ApplyVectorQuantizer(imageLena, clusters, block_size);
fprintf("The syntax of the code seems to be correct, next run the assessment to verify the correctness");

%% E3.4 b
% load('clusters')
% load('qImage')
block_size = 2;
reconst_image = InvVectorQuantizer(qImage, clusters, block_size);
fprintf("The syntax of the code seems to be correct, next run the assessment to verify the correctness");

%% E3.1 a
function qImage = UniQuant(image, bits)
%  Input         : image (Original Image)
%                : bits (bits available for representatives)
%
%  Output        : qImage (Quantized Image)
image = image./255;  % normalization
% Method 1:
partition_size = 1./(2.^bits);
qImage = floor(image ./ partition_size);
% Method 2:
% partition = linspace(0, 1, 2.^bits+1);
% qImage = imquantize(image, partition) - 2;  % index range [0, 2.^M-1]
end
%% E3.1 b
function image = InvUniQuant(qImage, bits)
%  Input         : qImage (Quantized Image)
%                : bits (bits available for representatives)
%
%  Output        : image (Mid-rise de-quantized Image)
partition_size = 256./(2.^bits);
qImage = qImage + 0.5;  % mid-rise quantization reconstruction
image = qImage .* partition_size;
end
%% E3.2
function [qImage, clusters] = LloydMax(image, bits, epsilon)
%  Input         : image (Original RGB Image)
%                  bits (bits for quantization)
%                  epsilon (Stop Condition)
%  Output        : qImage (Quantized Image)
%                  clusters (Quantization Table)

% initialization of Rep., code book size [nx3]
M = 2.^bits;  % number of Rep.
partition_size = 256 ./ M;
rep = partition_size/2 : partition_size : 255;
codeBook = cat(2, rep', zeros(M, 1), zeros(M, 1));
d = [9999];  % stored distortion
sz = size(image);
% iteration
while(1)
    d_quad = 0;
    for channel = 1:3
        pixel_value = image(:, :, channel);
        pixel_value = pixel_value(:);
        % everytime: take one channel of image into iteration
        [D, I] = pdist2(codeBook(:,1), pixel_value, ...
            'euclidean', 'Smallest', 1);
        for kk = 1:length(pixel_value)
            index = I(kk);
            codeBook(index, 2) = codeBook(index, 2) + pixel_value(kk);
            codeBook(index, 3) = codeBook(index, 3) + 1;
        end
        d_quad = d_quad + sum(D.^2);
    end
    % update and reset codebook
    index_zeroUpdate = find(codeBook(:, 3)==0);
    if ~isempty(index_zeroUpdate)
        % if there is empty cell, split the most populated
        for k = index_zeroUpdate
            [count_max, index_max] = max(codeBook(:, 3));
            codeBook(index_max, 3) = ceil(count_max/2);
            codeBook(k, 3) = count_max - codeBook(index_max, 3);
            codeBook(k, 2) = codeBook(index_max, 2)+1;
        end
    end
    codeBook(:, 1) = codeBook(:, 2)./codeBook(:, 3);
    codeBook(:, 2) = zeros(M, 1);
    codeBook(:, 3) = zeros(M, 1);
    % check stopping criterion
    d(end+1) = d_quad;
    % Lagrangian cost function with lamba=0: only the average distortion
    J = abs(d(end) - d(end-1))/d(end);
    if J < epsilon
        break
    end
end
% quantize the image using above codebook
qImage = zeros(sz(1), sz(2), 3);
for channel = 1:3
    pixel_value = image(:, :, channel);
    % everytime: take one channel of image into iteration
    [D, I] = pdist2(codeBook(:,1), pixel_value(:), 'euclidean', 'Smallest', 1);
    qImage(:, :, channel) = reshape(I, [sz(1), sz(2)]);
end
clusters = codeBook(:, 1);
end
%% E3.3 a
function image = InvLloydMax(qImage, clusters)
%  Input         : qImage   (Quantized Image)
%                  clusters (Quantization Table)
%  Output        : image    (Recovered Image)
numRep = length(clusters);
sz = size(qImage);
image = zeros(sz(1), sz(2), 3);

for channel = 1:3
    quantized = zeros(sz(1)*sz(2), 1);
    for i = 1:numRep
        pixel_value = qImage(:,:,channel);
        index = find(pixel_value(:) == i);
        quantized(index) = clusters(i);
    end
    image(:, :, channel) = reshape(quantized, [sz(1), sz(2)]);
end
end
%% E3.4 a
function [clusters, Temp_clusters] = VectorQuantizer(image, bits, epsilon, bsize)
% this function trains the codebook of a vector quantizer, returns the
% representative values.

% initialize using uniform vector quantizer
vector_length = bsize.^2;
M = 2.^bits;  % number of Rep.
partition_size = 255 ./ M;
rep = partition_size/2 : partition_size : 255;
rep = repmat(rep', [1, vector_length]);
codeBook = cat(2, rep, zeros(M, vector_length), zeros(M, 1));
d = [9999];  % stored distortion
sz = size(image);
% iteration
while(1)
    d_quad = 0;
    for channel = 1:3
        imageDivided = mat2cell(image(:,:,channel), ...
            ones(sz(1)/bsize,1)*bsize, ones(sz(2)/bsize,1)*bsize);
        for numCell = 1:numel(imageDivided)
            vectorized_imgBlock = imageDivided{numCell};
            vectorized_imgBlock = vectorized_imgBlock(:)';  % col-wise block
            [I, D] = knnsearch(codeBook(:, 1:vector_length),...
                vectorized_imgBlock, 'Distance', 'euclidean');
            codeBook(I, vector_length+1:end-1) = ...
                codeBook(I, vector_length+1:end-1) + vectorized_imgBlock;
            codeBook(I, end) = codeBook(I, end) + 1;
            d_quad = d_quad + D.^2;
        end
    end
    % find the zero and non-zero cells
    index_nonzeroUpdate = find(codeBook(:, end)~=0);
    index_zeroUpdate = find(codeBook(:, end)==0);
    % update the non-zero cells first
    codeBook(index_nonzeroUpdate, 1:vector_length) = ...
        codeBook(index_nonzeroUpdate, vector_length+1:end-1)./ ...
        codeBook(index_nonzeroUpdate, end);
    % handle the zero cells
    if ~isempty(index_zeroUpdate)
        % if there is an empty cell, split the most frequent used cell
        for k = index_zeroUpdate'
            [count_max, index_max] = max(codeBook(:, end));
            codeBook(index_max, end) = ceil(count_max/2);
            codeBook(k, end) = count_max - codeBook(index_max, end);
            codeBook(k, vector_length+1:end-1) = ...
                codeBook(index_max, vector_length+1:end-1);
            codeBook(k, end-1) = codeBook(k, end-1) + 1;
        end
    end
    % reset the codebook
    codeBook(:, vector_length+1:end-1) = zeros(M, vector_length);
    codeBook(:, end) = zeros(M, 1);
    % check stopping criterion
    d(end+1) = d_quad;
    % Lagrangian cost function with lamba=0: only the average distortion
    J = abs(d(end) - d(end-1))/d(end);
    if J < epsilon
        break
    end
end
clusters = codeBook(:, 1:vector_length);
Temp_clusters = codeBook(:, 1:vector_length);
end
%% E3.4 a
function qImage = ApplyVectorQuantizer(image, clusters, bsize)
%  Function Name : ApplyVectorQuantizer.m
%  Input         : image    (Original Image)
%                  clusters (Quantization Representatives)
%                  bsize    (Block Size)
%  Output        : qImage   (Quantized Image)

% quantize the image using existing codebook
sz = size(image);
numRow = sz(1)/bsize;  % number of rows after quantization
numCol = sz(2)/bsize;
qImage = zeros(numRow, numCol, 3);
for channel = 1:3
    % divide into blocks of size [bsize, bsize]
    imageDivided = mat2cell(image(:,:,channel), ...
        ones(sz(1)/bsize,1)*2, ones(sz(2)/bsize,1)*2);
    % col-wise fashion
    imageDivided = cellfun(@(a) a(:)', imageDivided, 'Uniform', false);
    numCell = numel(imageDivided);
    vectorized_imgBlock = reshape(imageDivided, numCell, 1);
    vectorized_imgBlock = cell2mat(vectorized_imgBlock);
    [I, ~] = knnsearch(clusters, ...
        vectorized_imgBlock, 'Distance', 'euclidean');
    qImage(:,:,channel) = reshape(I, numRow, numCol);
end
end

%% E 3.4 b
function image = InvVectorQuantizer(qImage, clusters, block_size)
%  Function Name : VectorQuantizer.m
%  Input         : qImage     (Quantized Image)
%                  clusters   (Quantization clusters)
%                  block_size (Block Size)
%  Output        : image      (Dequantized Images)
numRep = length(clusters);
sz = size(qImage);
originalRow = sz(1)*block_size;
originalCol = sz(2)*block_size;
image = zeros(originalRow, originalCol, 3);

for channel = 1:3
    quantized = cell(sz(1), sz(2));
    for i = 1:numRep
        pixel_value = qImage(:,:,channel);
        index = find(pixel_value(:) == i);
        quantized(index) = {clusters(i, :)};
    end
    % enlarge to the original size, each pixel has the same Rep. value
    % within the block
    quantized = cellfun(@(a) reshape(a, [block_size, block_size]), ...
        quantized, 'Uniform', false);
    image(:, :, channel) = cell2mat(quantized);
end
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
