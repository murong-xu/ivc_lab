%% Load all relevant data from folder
videoFolder = './sequences/foreman20_40_RGB/';
videoPath  = dir([videoFolder, '*.bmp']);

videoLength = length(videoPath);
originalFrame = cell(videoLength, 1);
for i = 1:videoLength
    originalFrame{i}= double(imread([videoFolder, videoPath(i).name]));
end
originalYCbCr = cellfun(@ictRGB2YCbCr, originalFrame, 'Uniform', false);
lena_small = double(imread('./IVC_labs_starting_point/data/images/lena_small.tif'));
%% Loop for various compression scales
scales =1;
% scales = [0.07, 0.1, 0.2, 0.4, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 4.5];
for scaleIdx = 1 : numel(scales)
    qScale   = scales(scaleIdx);
    recFrame = cell(videoLength, 1);
    finalRecFrame = cell(videoLength, 1);
    finalRecFrameRGB = cell(videoLength, 1);
    averagePSNR = zeros(numel(scales), 1);
    averageRate = zeros(numel(scales), 1);
    %% Part 1: encode 1st frame using still image compression
    % run-level encoded results in 1D vector [Y, Cb, Cr]
    firstFrame = originalFrame{1};
    lena_small_ycbcr = ictRGB2YCbCr(lena_small);
    firstFrame_ycbcr = ictRGB2YCbCr(firstFrame);
    zeroRunLenaSmall = IntraEncode(lena_small_ycbcr, qScale);
    zeroRunFirstFrame = IntraEncode(firstFrame_ycbcr, qScale);
    recFrame{1} = IntraDecode(zeroRunFirstFrame, size(firstFrame), qScale);  % YCbCr
    %% Part 1: use pmf of zeroRunLenaSmall to build and train huffman table
    rangeFirstFrame = min(zeroRunFirstFrame):max(zeroRunFirstFrame);  % based on the actual range of encoded image
    pmfFirstFrame = stats_marg_1D(zeroRunLenaSmall, rangeFirstFrame);  % use marg. PMF for 1D
    % Huffman Table
    [BinaryTreeFirstFrame, HuffCodeFirstFrame, BinCodeFirstFrame, CodelengthsFirstFrame] = buildHuffman(pmfFirstFrame');
    %% Part 1: use trained table to encode zeroRunFirstFrame to get the bytestream
    % encoding
    lowerBoundFirstFrame = min(zeroRunFirstFrame);  % based on the actual range of encoded image
    bytestreamZeroRun = enc_huffman_new(zeroRunFirstFrame-lowerBoundFirstFrame+1, BinCodeFirstFrame, CodelengthsFirstFrame);
    
    % calculate bit rate
    rate(1) = (numel(bytestreamZeroRun)*8) / (numel(firstFrame)/3);
    
    % decoding
    decZeroRun = double(reshape(dec_huffman_new(bytestreamZeroRun, BinaryTreeFirstFrame,...
        max(size(zeroRunFirstFrame(:)))), size(zeroRunFirstFrame))) - 1 + lowerBoundFirstFrame;
    %% Part 1: image reconstruction
    finalRecFrame{1} = IntraDecode(decZeroRun, size(firstFrame), qScale);  % YCbCr
    finalRecFrameRGB{1} = ictYCbCr2RGB(finalRecFrame{1});
    PSNR(1) = calcPSNR(firstFrame, finalRecFrameRGB{1});
    
    
    %% Part 2: encode the rest of frames
    rangeHuffman = -1000 : 4000;  % global lower & upper bound
    for i = 2 : videoLength
        refYCbCr = recFrame{i-1};  % last reconst. frame
        currentRGB = originalFrame{i};
        currentYCbCr = originalYCbCr{i};
        MV = SSD(refYCbCr(:,:,1), currentYCbCr(:,:,1));  % only Y component
        mcp = SSD_rec(refYCbCr, MV);  % predicted current img, YCbCr
        errYCbCr = currentYCbCr - mcp;  % prediction error
        zeroRun = IntraEncode(errYCbCr, qScale);
        recFrame{i} = IntraDecode(zeroRun, size(currentYCbCr), qScale) + mcp;  % YCbCr
        %% Part 2: build Huffman table, compute only once
        if i == 2
            % use pmf to build and train huffman table
            pmfMV = hist(MV(:),rangeHuffman);
            pmfMV = pmfMV./sum(pmfMV);
            pmfZeroRun = hist(zeroRun(:),rangeHuffman);
            pmfZeroRun = pmfZeroRun ./sum(pmfZeroRun);
            
            % Huffman Table
            [BinaryTreeMV, HuffCodeMV, BinCodeMV, CodelengthsMV] = buildHuffman(pmfMV);
            [BinaryTreeZeroRun, HuffCodeZeroRun, BinCodeZeroRun, CodelengthsZeroRun] = buildHuffman(pmfZeroRun);
        end
        %% Part 2: use trained table to encode & decode MV and residual image
        % encoding
        lowerBoundHuffman = -1000;  % global lowerbound
        bytestreamZeroRun = enc_huffman_new(zeroRun-lowerBoundHuffman+1, BinCodeZeroRun, CodelengthsZeroRun);
        bytestreamMV = enc_huffman_new(MV-lowerBoundHuffman+1, BinCodeMV, CodelengthsMV);
        
        % calculate bit rate
        rateZeroRun = (numel(bytestreamZeroRun)*8) / (numel(currentYCbCr)./3); 
        rateMV = (numel(bytestreamMV)*8) / (numel(currentYCbCr)./3);
        
        % decoding
        decZeroRun = double(reshape(dec_huffman_new(bytestreamZeroRun, BinaryTreeZeroRun,...
            max(size(zeroRun(:)))), size(zeroRun))) - 1 + lowerBoundHuffman;
        decMV = double(reshape(dec_huffman_new(bytestreamMV, BinaryTreeMV,...
            max(size(MV(:)))), size(MV))) - 1 + lowerBoundHuffman;
        %% Part 2: image reconstruction
        mcp2 = SSD_rec(finalRecFrame{i-1}, decMV);
        finalRecFrame{i} = IntraDecode(decZeroRun, size(currentYCbCr), qScale) + mcp2;  % YCbCr
        finalRecFrameRGB{i} = ictYCbCr2RGB(finalRecFrame{i});
        rate(i) = rateZeroRun + rateMV;
        PSNR(i) = calcPSNR(currentRGB, finalRecFrameRGB{i});
        
    end
    %% Final: evaluation on the whole sequence: average PSNR and Bitrate
    averageRate(scaleIdx) = mean(rate);
    averagePSNR(scaleIdx) = mean(PSNR);
    fprintf('QP: %.1f bit-rate: %.2f bits/pixel PSNR: %.2fdB\n', ...
        qScale, averageRate(scaleIdx), averagePSNR(scaleIdx))
end


%% Intra Encoding & Decoding
function dst = IntraEncode(image, qScale)
%  Function Name : IntraEncode.m
%  Input         : image (Original YCbCr Image)
%                  qScale(quantization scale)
%  Output        : dst   (sequences after zero-run encoding, 1xN)

[sizeY, sizeX, ~] = size(image);
blockLength = 8;
numBlockY = sizeY ./ blockLength;
numBlockX = sizeX ./ blockLength;

% % RGB to YCbCr
% img = ictRGB2YCbCr(image);
img = image;

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
EOB = 4000;
dst_Y = ZeroRunEnc_EoB(sequenceZigZag_Y, EOB);
dst_Cb = ZeroRunEnc_EoB(sequenceZigZag_Cb, EOB);
dst_Cr = ZeroRunEnc_EoB(sequenceZigZag_Cr, EOB);
dst = [dst_Y, dst_Cb, dst_Cr];
end

function dst = IntraDecode(image, img_size, qScale)
%  Function Name : IntraDecode.m
%  Input         : image (zero-run encoded image, 1xN)
%                  img_size (original image size)
%                  qScale(quantization scale)
%  Output        : dst   (decoded image, YCbCr)

sizeY = img_size(1);
sizeX = img_size(2);
blockLength = 8;
symbolInBlock = blockLength.^2;
numBlockY = sizeY ./ blockLength;
numBlockX = sizeX ./ blockLength;

% Run-level Decoding
EOB = 4000;
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

% % YCbCr to RGB
% dst = ictYCbCr2RGB(img_IDCT);
dst = img_IDCT;
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
currentZzePos = 1;

% loop for all blocks
for k = 1 : numBlock
    numFollowingZeros = -1;
    for i = 1:numSymbolInBlock
        currentZzPos = (k-1)*numSymbolInBlock + i;
        % case 1: current input is non-zero
        if zz(currentZzPos) ~= 0
            % check if there are existing successive zeros before
            if numFollowingZeros ~= -1
                zze(currentZzePos) = numFollowingZeros;
                currentZzePos = currentZzePos + 1;
            end
            zze(currentZzePos) = zz(currentZzPos);  % non-zero
            currentZzePos = currentZzePos + 1;  % position increment
            numFollowingZeros = -1;
        else
            % case 2: current input is a zero
            % case 2.1: the current zero is a "first" zero
            if numFollowingZeros == -1
                zze(currentZzePos) = 0;
                currentZzePos = currentZzePos + 1;
                numFollowingZeros = 0;
            else
                % case 2.2: the current zero is a "following" zero
                numFollowingZeros = numFollowingZeros + 1;
            end
        end
    end
    % case 3: when end of block still find successive zeros
    if numFollowingZeros ~= -1
        zze(currentZzePos-1) = EOB;
    end
end
end

function dst = ZeroRunDec_EoB(src, EoB)
%  Function Name : ZeroRunDec1.m zero run level decoder
%  Input         : src (zero run encoded sequence 1xM with EoB signs)
%                  EoB (end of block sign)
%
%  Output        : dst (reconstructed zig-zag scanned sequence 1xN)

numElement = size(src, 2);  % number of zero-run encoded symbols
numSymbolInBlock = 64;

currentSrcPos = 1;
currentDstPos = 1;
indexRest = 1;
for i = 1 : numElement
    % case 1: current input is non-zero
    if src(currentSrcPos) ~= 0
        % case 1.1: current input is not EOB
        if src(currentSrcPos) ~= EoB
            dst(currentDstPos) = src(currentSrcPos);
            currentDstPos = currentDstPos + 1;  % increment
            currentSrcPos = currentSrcPos + 1;
            indexRest = indexRest + 1;
            if indexRest >= numSymbolInBlock+1  % block reset
                indexRest = 1;
            end
        else
            % case 1.2: current input is EOB
            indexRestZero = numSymbolInBlock - indexRest;
            dst(currentDstPos : currentDstPos+indexRestZero) = 0;
            indexRest = 1;  % reset
            currentDstPos = currentDstPos + indexRestZero + 1;
            currentSrcPos = currentSrcPos + 1;
        end
    else
        % case 2: current input is a zero
        indexRestZero = src(currentSrcPos + 1);
        dst(currentDstPos : currentDstPos+indexRestZero) = 0;
        indexRest = indexRest + indexRestZero + 1;
        currentDstPos = currentDstPos + indexRestZero + 1;
        currentSrcPos = currentSrcPos + 2;
    end
    if currentSrcPos > numElement
        break
    end
end
end

%% SSD
function motion_vectors_indices = SSD(ref_image, image)
%  Input         : ref_image(Reference Image, size: height x width)
%                  image (Current Image, size: height x width)
%
%  Output        : motion_vectors_indices (Motion Vector Indices, size: (height/8) x (width/8) x 1 )

height = size(image, 1);
width = size(image, 2);
numBlockX = width ./ 8;
numBlockY = height ./ 8;
motion_vectors_indices = zeros(numBlockY, numBlockX);

positionBlockX = 1 : 8 : width;  % position of block
positionBlockY = 1 : 8 : height;
% motion vector indexing for range +-4, row-wise
indexTable = reshape(1:81, 9, 9)';

for row = 1:numBlockY
    for col = 1:numBlockX
        locX = positionBlockX(col);  % position of current img block
        locY = positionBlockY(row);
        imgBlock = image(locY:locY+7, locX:locX+7);  % current image block
        % loop in search field +-4
        flag_skip = 0;
        for refX = locX-4:locX+4
            if (refX<1) | (refX>(width-7))  % exceed border
                continue
            end
            for refY = locY-4:locY+4
                if (refY<1) | (refY>(height-7))  % exceed border
                    continue
                end
                % current reference block
                refBlock = ref_image(refY:refY+7, refX:refX+7);
                diff = (imgBlock - refBlock).^2;
                SSD = sum(diff(:));
                if ~flag_skip
                    SSD_min = SSD;  % initialize the first threshold
                    flag_skip = 1;  % only once
                end
                if SSD <= SSD_min
                    SSD_min = SSD;
                    bestX = refX;  % absolute position
                    bestY = refY;
                end
            end
        end
        % relative vector, center is (4,4)
        vector = [bestX, bestY] - [locX, locY] + 5;
        motion_vectors_indices(row, col) = indexTable(vector(2), vector(1));
    end
end
end

function rec_image = SSD_rec(ref_image, motion_vectors)
%  Input         : ref_image(Reference Image, YCbCr image)
%                  motion_vectors
%
%  Output        : rec_image (Reconstructed current image, YCbCr image)
height = size(ref_image, 1);
width = size(ref_image, 2);
numBlockX = width ./ 8;
numBlockY = height ./ 8;
rec_image = zeros(height, width, 3);

positionBlockX = 1 : 8 : width;  % position of block
positionBlockY = 1 : 8 : height;
% motion vector indexing for range +-4, row-wise
indexTable = reshape(1:81, 9, 9)';

for row = 1:numBlockY
    for col = 1:numBlockX
        index = motion_vectors(row, col);
        [vectorY, vectorX] = find(indexTable==index);  % not centered
        vectorX = vectorX - 5;  % centered to (5,5), relative vector
        vectorY = vectorY - 5;
        locX = positionBlockX(col);  % position of current img block
        locY = positionBlockY(row);
        refX = locX + vectorX;  % position of reference block
        refY = locY + vectorY;
        refBlock = ref_image(refY:refY+7, refX:refX+7, :);  % current ref block
        rec_image(locY:locY+7, locX:locX+7, :) = refBlock;
    end
end
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

%% Huffman
function [ BinaryTree, HuffCode, BinCode, Codelengths] = buildHuffman( p )

global y

p=p(:)/sum(p)+eps;              % normalize histogram
p1=p;                           % working copy

c=cell(length(p1),1);			% generate cell structure

for i=1:length(p1)				% initialize structure
    c{i}=i;
end

while size(c)-2					% build Huffman tree
    [p1,i]=sort(p1);			% Sort probabilities
    c=c(i);						% Reorder tree.
    c{2}={c{1},c{2}};           % merge branch 1 to 2
    c(1)=[];	                % omit 1
    p1(2)=p1(1)+p1(2);          % merge Probabilities 1 and 2
    p1(1)=[];	                % remove 1
end

%cell(length(p),1);              % generate cell structure
getcodes(c,[]);                  % recurse to find codes
code=char(y);

[numCodes maxlength] = size(code); % get maximum codeword length

% generate byte coded huffman table
% code

length_b=0;
HuffCode=zeros(1,numCodes);
for symbol=1:numCodes
    for bit=1:maxlength
        length_b=bit;
        if(code(symbol,bit)==char(49)) HuffCode(symbol) = HuffCode(symbol)+2^(bit-1)*(double(code(symbol,bit))-48);
        elseif(code(symbol,bit)==char(48))
        else
            length_b=bit-1;
            break;
        end
    end
    Codelengths(symbol)=length_b;
end

BinaryTree = c;
BinCode = code;

%clear global y;

return
end

%----------------------------------------------------------------
function getcodes(a,dum)
global y                            % in every level: use the same y
if isa(a,'cell')                    % if there are more branches...go on
    getcodes(a{1},[dum 0]);    %
    getcodes(a{2},[dum 1]);
else
    y{a}=char(48+dum);
end
end


function [bytestream] = enc_huffman_new( data, BinCode, Codelengths)

a = BinCode(data(:),:)';
b = a(:);
mat = zeros(ceil(length(b)/8)*8,1);
p  = 1;
for i = 1:length(b)
    if b(i)~=' '
        mat(p,1) = b(i)-48;
        p = p+1;
    end
end
p = p-1;
mat = mat(1:ceil(p/8)*8);
d = reshape(mat,8,ceil(p/8))';
multi = [1 2 4 8 16 32 64 128];
bytestream = sum(d.*repmat(multi,size(d,1),1),2);

end


function [output] = dec_huffman_new (bytestream, BinaryTree, nr_symbols)

output = zeros(1,nr_symbols);
ctemp = BinaryTree;

dec = zeros(size(bytestream,1),8);
for i = 8:-1:1
    dec(:,i) = rem(bytestream,2);
    bytestream = floor(bytestream/2);
end

dec = dec(:,end:-1:1)';
a = dec(:);

i = 1;
p = 1;
while(i <= nr_symbols)&&p<=max(size(a))
    while(isa(ctemp,'cell'))
        next = a(p)+1;
        p = p+1;
        ctemp = ctemp{next};
    end
    output(i) = ctemp;
    ctemp = BinaryTree;
    i=i+1;
end
end