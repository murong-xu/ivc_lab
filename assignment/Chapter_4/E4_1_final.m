lena_small = double(imread('./IVC_labs_starting_point/data/images/monarch.tif'));
Lena       = double(imread('./IVC_labs_starting_point/data/images/monarch.tif'));

scales = 0.5 : 0.5 : 20; % quantization scale factor, for E(4-1), we just evaluate scale factor of 1
for scaleIdx = 1 : numel(scales)
    qScale   = scales(scaleIdx);
    % k and k_small: run-level encoded results in 1D vector [Y, Cb, Cr]
    k_small  = IntraEncode(lena_small, qScale);
    k        = IntraEncode(Lena, qScale);
    %% use pmf of k_small to build and train huffman table
    range_small = min(k):max(k);  % based on the actual range of encoded image
    pmfLena_small = stats_marg_1D(k_small, range_small);  % use marg. PMF for 1D
    % Huffman Table
    [BinaryTree, HuffCode, BinCode, Codelengths] = buildHuffman(pmfLena_small');
    %% use trained table to encode k to get the bytestream
    % encoding
    lowerBound = min(k);  % based on the actual range of encoded image
    bytestream = enc_huffman_new(k-lowerBound+1, BinCode, Codelengths);
    
    % calculate bit rate
    bitPerPixel(scaleIdx) = (numel(bytestream)*8) / (numel(Lena)/3);
    
    % decoding
    k_rec = double(reshape(dec_huffman_new(bytestream, BinaryTree,...
        max(size(k(:)))), size(k))) - 1 + lowerBound;
    %% image reconstruction
    I_rec = IntraDecode(k_rec, size(Lena), qScale);
    PSNR(scaleIdx) = calcPSNR(Lena, I_rec);
    fprintf('QP: %.1f bit-rate: %.2f bits/pixel PSNR: %.2fdB\n', ...
        qScale, bitPerPixel(scaleIdx), PSNR(scaleIdx))
end

%% E4.1
function dst = IntraEncode(image, qScale)
%  Function Name : IntraEncode.m
%  Input         : image (Original RGB Image)
%                  qScale(quantization scale)
%  Output        : dst   (sequences after zero-run encoding, 1xN)

[sizeY, sizeX] = size(image);
blockLength = 8;
numBlockY = sizeY ./ blockLength;
numBlockX = sizeX ./ blockLength;

% RGB to YCbCr
img = ictRGB2YCbCr(image);

% DCT
img_DCT = blockproc(img, [blockLength,blockLength], ...
    @(block_struct) DCT8x8(block_struct.data));

% Quantization
img_Quan = blockproc(img_DCT, [blockLength,blockLength],...
    @(block_struct) Quant8x8(block_struct.data, qScale));

% ZigZag
matrix_ZigZag = blockproc(img_Quan, [blockLength,blockLength],...
    @(block_struct) ZigZag8x8(block_struct.data));
matrixZigZag_Y = matrix_ZigZag(:, 1:3:end);
matrixZigZag_Cb = matrix_ZigZag(:, 2:3:end);
matrixZigZag_Cr = matrix_ZigZag(:, 3:3:end);
sequenceZigZag_Y = matrixZigZag_Y(:)';  % read blocks col-wise
sequenceZigZag_Cb = matrixZigZag_Cb(:)';
sequenceZigZag_Cr = matrixZigZag_Cr(:)';

% Run-level Encoding
EOB = 1000;
dst_Y = ZeroRunEnc_EoB(sequenceZigZag_Y, EOB);
dst_Cb = ZeroRunEnc_EoB(sequenceZigZag_Cb, EOB);
dst_Cr = ZeroRunEnc_EoB(sequenceZigZag_Cr, EOB);
dst = [dst_Y, dst_Cb, dst_Cr];
end

%% E4.1
function dst = IntraDecode(image, img_size, qScale)
%  Function Name : IntraDecode.m
%  Input         : image (zero-run encoded image, 1xN)
%                  img_size (original image size)
%                  qScale(quantization scale)
%  Output        : dst   (decoded image)

sizeY = img_size(1);
sizeX = img_size(2);
blockLength = 8;
symbolInBlock = blockLength.^2;
numBlockY = sizeY ./ blockLength;
numBlockX = sizeX ./ blockLength;

% Run-level Decoding
EOB = 1000;
sequence_IRun = ZeroRunDec_EoB(image, EOB);

% ZigZag Reverse
sequenceLength = length(sequence_IRun)./3;
sequenceIRun_Y = sequence_IRun(1 : sequenceLength);
sequenceIRun_Cb = sequence_IRun(sequenceLength+1 : 2*sequenceLength);
sequenceIRun_Cr = sequence_IRun(2*sequenceLength+1 : end);
matrixIRun_Y = reshape(sequenceIRun_Y, [numBlockY*symbolInBlock, numBlockX]);
matrixIRun_Cb = reshape(sequenceIRun_Cb, [numBlockY*symbolInBlock, numBlockX]);
matrixIRun_Cr = reshape(sequenceIRun_Cr, [numBlockY*symbolInBlock, numBlockX]);
matrixIRun = zeros(numBlockY*symbolInBlock, 3*numBlockX);
matrixIRun(:, 1:3:end) = matrixIRun_Y;
matrixIRun(:, 2:3:end) = matrixIRun_Cb;
matrixIRun(:, 3:3:end) = matrixIRun_Cr;

img_IZigZag = blockproc(matrixIRun, [symbolInBlock, 3], @(block_struct) DeZigZag8x8(block_struct.data));

% Quantization Reverse
img_IQuan = blockproc(img_IZigZag, [blockLength,blockLength], @(block_struct) DeQuant8x8(block_struct.data, qScale));

% IDCT
img_IDCT = blockproc(img_IQuan, [8,8], @(block_struct) IDCT8x8(block_struct.data));

% YCbCr to RGB
dst = ictYCbCr2RGB(img_IDCT);

end

%% DCT Encoder Decoder
function coeff = DCT8x8(block)
%  Input         : block    (Original Image block, 8x8x3)
%
%  Output        : coeff    (DCT coefficients after transformation, 8x8x3)
coeff = zeros(8, 8, 3);
for channel=1:3
    % dct function: apply on each row of the input
    dct_col = dct(block(:, :, channel)');  % apply DCT on columns of block
    coeff(:, :, channel) = dct(dct_col');
end
end

function block = IDCT8x8(coeff)
%  Function Name : IDCT8x8.m
%  Input         : coeff (DCT Coefficients) 8*8*3
%  Output        : block (original image block) 8*8*3
block = zeros(8, 8, 3);
for channel=1:3
    % idct function: apply on each row of the input
    block_col = idct(coeff(:, :, channel)');  % apply IDCT on columns of coeff
    block(:, :, channel) = idct(block_col');
end
end

%% Quantization
function quant = Quant8x8(dct_block, qScale)
%  Input         : dct_block (Original Coefficients, 8x8x3), YCbCr
%                  qScale (Quantization Parameter, scalar)
%
%  Output        : quant (Quantized Coefficients, 8x8x3)
quant = zeros(8, 8, 3);
quanTable_L = [16,11,10,16,24,40,51,61; 12,12,14,19,26,58,60,55;...
    14,13,16,24,40,57,69,56; 14,17,22,29,51,87,80,62;...
    18,55,37,56,68,109,103,77; 24,35,55,64,81,104,113,92;...
    49,64,78,87,103,121,120,101; 72,92,95,98,112,100,103,99];
quanTable_C = [17,18,24,47,99,99,99,99; 18,21,26,66,99,99,99,99;...
    24,13,56,99,99,99,99,99; 47,66,99,99,99,99,99,99;...
    99,99,99,99,99,99,99,99; 99,99,99,99,99,99,99,99;...
    99,99,99,99,99,99,99,99; 99,99,99,99,99,99,99,99];
quant(:, :, 1) = round(dct_block(:, :, 1) ./ (quanTable_L .* qScale));
quant(:, :, 2) = round(dct_block(:, :, 2) ./ (quanTable_C .* qScale));
quant(:, :, 3) = round(dct_block(:, :, 3) ./ (quanTable_C .* qScale));
end

function dct_block = DeQuant8x8(quant_block, qScale)
%  Function Name : DeQuant8x8.m
%  Input         : quant_block  (Quantized Block, 8x8x3)
%                  qScale       (Quantization Parameter, scalar)
%
%  Output        : dct_block    (Dequantized DCT coefficients, 8x8x3)
dct_block = zeros(8, 8, 3);
quanTable_L = [16,11,10,16,24,40,51,61; 12,12,14,19,26,58,60,55;...
    14,13,16,24,40,57,69,56; 14,17,22,29,51,87,80,62;...
    18,55,37,56,68,109,103,77; 24,35,55,64,81,104,113,92;...
    49,64,78,87,103,121,120,101; 72,92,95,98,112,100,103,99];
quanTable_C = [17,18,24,47,99,99,99,99; 18,21,26,66,99,99,99,99;...
    24,13,56,99,99,99,99,99; 47,66,99,99,99,99,99,99;...
    99,99,99,99,99,99,99,99; 99,99,99,99,99,99,99,99;...
    99,99,99,99,99,99,99,99; 99,99,99,99,99,99,99,99];
dct_block(:, :, 1) = quant_block(:, :, 1) .* qScale .* quanTable_L;
dct_block(:, :, 2) = quant_block(:, :, 2) .* qScale .* quanTable_C;
dct_block(:, :, 3) = quant_block(:, :, 3) .* qScale .* quanTable_C;
end

%% ZigZag
function zz = ZigZag8x8(quant)
%  Input         : quant (Quantized Coefficients, 8x8x3)
%
%  Output        : zz (zig-zag scaned Coefficients, 64x3)
zz = zeros(64, 3);
ZigZag = [1,2,6,7,15,16,28,29; 3,5,8,14,17,27,30,43;...
    4,9,13,18,26,31,42,44; 10,12,19,25,32,41,45,54;...
    11,20,24,33,40,46,53,55; 21,23,34,39,47,52,56,61;...
    22,35,38,48,51,57,60,62; 36,37,49,50,58,59,63,64];
for channel=1:3
    quantBlock = quant(:, :, channel);
    zz(ZigZag(:),  channel) = quantBlock(:);
end
end

function coeffs = DeZigZag8x8(zz)
%  Function Name : DeZigZag8x8.m
%  Input         : zz    (Coefficients in zig-zag order)
%
%  Output        : coeffs(DCT coefficients in original order)
coeffs = zeros(8, 8, 3);
ZigZag = [1,2,6,7,15,16,28,29; 3,5,8,14,17,27,30,43;...
    4,9,13,18,26,31,42,44; 10,12,19,25,32,41,45,54;...
    11,20,24,33,40,46,53,55; 21,23,34,39,47,52,56,61;...
    22,35,38,48,51,57,60,62; 36,37,49,50,58,59,63,64];
for channel=1:3
    coeff = zz(ZigZag(:), channel);
    coeffs(:, :, channel) = reshape(coeff, 8, 8);
end
end

%% Zero-run
function zze = ZeroRunEnc_EoB(zz, EOB)
%  Input         : zz (Zig-zag scanned sequence, 1xN)
%                  EOB (End Of Block symbol, scalar)
%
%  Output        : zze (zero-run-level encoded sequence, 1xM)
numElement = size(zz, 2);
numSymbolInBlock = 64;
numBlock = numElement ./ numSymbolInBlock;
zze = zeros(1, numElement) * NaN;  % initialize with larger number of elements (efficiency)

% loop for all blocks
for k=1:numBlock
    blockIncrement = (k - 1) * numSymbolInBlock;
    zz_curent = zz(blockIncrement+1 : blockIncrement+numSymbolInBlock);
    indexNonZero = find(zz_curent~=0);  % non-zeros' indices within a block
    lengthNonZero = length(indexNonZero);
    % case 1: if all symbols from the current block are zeros
    if lengthNonZero==0
        zze(1+blockIncrement) = EOB;
    else
        % case 2: non-zero exists in block, the first DC component is non-zero
        if indexNonZero(1)==1
            zze(1+blockIncrement) = zz_curent(1);  % encode the first DC component separately
            zzeIndexIncrement = 2;  % init of the zze index
        else
            % case 3: non-zero exists in block, the first DC component and its followings are zeros
            run = indexNonZero(1) - 2;
            zze(1+blockIncrement:3+blockIncrement) = [0, run, zz_curent(indexNonZero(1))];
            zzeIndexIncrement = 4;  % init of the zze index
        end
        
        % continue case 2&3: loop all of the non-zero symbols
        for i=2:lengthNonZero
            % number of zeros between neighboring two non-zero symbols
            run = indexNonZero(i) - indexNonZero(i-1) - 1;
            % if no zero between the neighboring non-zeros
            if run==0
                zze(zzeIndexIncrement+blockIncrement) = zz_curent(indexNonZero(i));
                zzeIndexIncrement = zzeIndexIncrement + 1;
                % if zero exists, then update both zeros and non-zeros
            else
                zze(zzeIndexIncrement+blockIncrement: ...
                    zzeIndexIncrement+2+blockIncrement) = ...
                    [0, run-1, zz_curent(indexNonZero(i))];
                zzeIndexIncrement = zzeIndexIncrement + 3;
            end
        end
        % insert EOB at the end for each block
        if indexNonZero(end) ~= numSymbolInBlock
            zze(numSymbolInBlock+blockIncrement) = EOB;
        end
    end
end
% after looping all blocks: remove the initialization of NaNs
numNonNaN = find(~isnan(zze));
zze = zze(numNonNaN);
end

function dst = ZeroRunDec_EoB(src, EoB)
%  Function Name : ZeroRunDec1.m zero run level decoder
%  Input         : src (zero run encoded sequence 1xM with EoB signs)
%                  EoB (end of block sign)
%
%  Output        : dst (reconstructed zig-zag scanned sequence 1xN)

numElement = size(src, 2);  % number of zero-run encoded symbols
numSymbolInBlock = 64;

% initialize with larger number of elements (efficiency)
dst = zeros(1, numElement*numSymbolInBlock) * NaN;

counterBlock = 1;  % initialization of block counter
counterIndex = 1;  % initialization of index number, should in range [1, 64]
flag_skip = 0;  % determine whether to skip the current iteration

% loop for all zero-run encoded symbols
for i=1:numElement
    current_symbol = src(i);
    blockIncrement = (counterBlock-1) * numSymbolInBlock;
    % check if it's necessary to execute the current iteration
    if flag_skip == 1
        flag_skip = 0;  % reset
        continue
    end
    
    switch current_symbol
        case EoB  % case 1: the zero-run symbol is EOB
            rest_length = numSymbolInBlock - counterIndex + 1;
            dst(counterIndex+blockIncrement : numSymbolInBlock+blockIncrement) = zeros(1, rest_length);
            counterIndex = 1;  % reset the index to the beginning
            counterBlock = counterBlock + 1;
            continue  % once decode an EOC symbol, jump to the next block
            
        case 0  % case 2: the zero-run symbol is a sequence of zeros
            run = src(i+1);
            dst(counterIndex+blockIncrement : counterIndex+run+blockIncrement) = zeros(1, run+1);
            counterIndex = counterIndex + 1 + run;
            flag_skip = 1;  % no need to update the next zero-run symbol
            
        otherwise  % case 3: the zero-run symbol is non-zero
            dst(counterIndex+blockIncrement) = current_symbol;
            counterIndex = counterIndex + 1;
    end
    
    % check if the current block is full
    if counterIndex > 64
        counterIndex = 1;  % reset the index to the beginning position
        counterBlock = counterBlock + 1;
        continue
    end
end
% after looping all blocks: remove the initialization of NaNs
numNonNaN = find(~isnan(dst));
dst = dst(numNonNaN);
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

%% PSNR
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

%% For Huffman: special cass (1D) for computing PMF
function pmf = stats_marg_1D(sequence, range)
% Input:   sequence (1D sequence)
%          range (range and step value of histogram calculaton, e.g. 0:255)
% Output:  pmf (probability mass function)
num = length(sequence); % denominator: height*width*channel
pmf = hist(sequence, range); % count PMF in individual channel
pmf = pmf./num; % sum up the 3 values for one pixel, normalization
% pmf(pmf==0) = []; % easier for calculating entropy (avoid NaN)
end