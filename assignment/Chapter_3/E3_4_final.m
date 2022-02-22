%% E3.4 Final
% A still image codec based on the vector quantization (VQ) and Huffman
% coding.

%% Main
bits = 8;
epsilon = 0.1;
block_size = 2;
%% lena small for VQ training
image_small = double(imread('./IVC_labs_starting_point/data/images/lena_small.tif'));
[clusters, Temp_clusters] = VectorQuantizer(image_small, bits, epsilon, block_size);
qImage_small = ApplyVectorQuantizer(image_small, clusters, block_size);
%% Huffman table training
range_small = 1:2.^bits;
pmfQImg_small = stats_marg(qImage_small, range_small);
% Huffman Table
[BinaryTree, HuffCode, BinCode, Codelengths] = buildHuffman(pmfQImg_small);
%%
image = double(imread('./IVC_labs_starting_point/data/images/lena.tif'));
qImage = ApplyVectorQuantizer(image, clusters, block_size);
%% Huffman encoding
quanImage_R = qImage(:,:,1);
quanImage_G = qImage(:,:,2);
quanImage_B = qImage(:,:,3);
lowerBound = 1;

bytestream_R = enc_huffman_new(quanImage_R(:)-lowerBound+1, BinCode, Codelengths);
bytestream_G = enc_huffman_new(quanImage_G(:)-lowerBound+1, BinCode, Codelengths);
bytestream_B = enc_huffman_new(quanImage_B(:)-lowerBound+1, BinCode, Codelengths);
bytestream = [bytestream_R; bytestream_G; bytestream_B];
%%
bpp = (numel(bytestream) * 8) / (numel(image)/3);
%% Huffman decoding
decQuan_R = double(reshape(dec_huffman_new(bytestream_R, BinaryTree,...
    max(size(quanImage_R(:)))), size(quanImage_R))) - 1 + lowerBound;
decQuan_G = double(reshape(dec_huffman_new(bytestream_G, BinaryTree, ...
    max(size(quanImage_G(:)))), size(quanImage_G))) - 1 + lowerBound;
decQuan_B = double(reshape(dec_huffman_new(bytestream_B, BinaryTree, ...
    max(size(quanImage_B(:)))), size(quanImage_B))) - 1 + lowerBound;
qReconst_image = cat(3, decQuan_R, decQuan_G, decQuan_B);
%%
reconst_image = InvVectorQuantizer(qReconst_image, clusters, block_size);
PSNR = calcPSNR(image, reconst_image);

%% sub-functions
function [clusters, Temp_clusters] = VectorQuantizer(image, bits, epsilon, bsize)
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

function qImage = ApplyVectorQuantizer(image, clusters, bsize)
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

function image = InvVectorQuantizer(qImage, clusters, block_size)
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

function PSNR = calcPSNR(Image, recImage)
% call calcMSE to calculate MSE
MSE = calcMSE(Image, recImage);
max_intensity = 2^8 - 1;
PSNR = 10*log10(max_intensity.^2/MSE);
end

function MSE = calcMSE(Image, recImage)
img_double = double(Image);
recImg_double = double(recImage);
img_diff = img_double - recImg_double;
mse = img_diff.^2;
MSE = mean(mse(:));
end

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