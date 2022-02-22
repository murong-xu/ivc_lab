%% Description
% Run this script to plot the RD-curves for following methods:
% 1, Chp4 Baseline: Intra Coding
% 2, Chp5 Baseline: Video Coding
% 3, Intra Coding Optimization: Intra Prediction
% 4, Intra Coding Optimization: Adaptive Post-deblocking
% 5, Video Coding Optimization: Half-pel Motion Estimation
% 6, Video Coding Optimization: Quarter-pel Motion Estimation

%% Please specify the input here
videoFolder = './sequences/foreman20_40_RGB/';
lena_small = double(imread('./IVC_labs_starting_point/data/images/lena_small.tif'));

scales = [0.06, 0.07, 0.1, 0.2, 0.4, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 4.5];

%% Load all relevant data
videoPath  = dir([videoFolder, '*.bmp']);
videoLength = length(videoPath);

sz = size(double(imread([videoFolder, videoPath(1).name])));

originalFrame = zeros(sz(1), sz(2), 3*videoLength);
originalYCbCr = zeros(sz(1), sz(2), 3*videoLength);
for i = 1:videoLength
    originalFrame(:,:,(3*i-2):(3*i))= double(imread([videoFolder, videoPath(i).name]));
    originalYCbCr(:,:,(3*i-2):(3*i)) = ictRGB2YCbCr(originalFrame(:,:,(3*i-2):(3*i)));
end

lena_small_ycbcr = ictRGB2YCbCr(lena_small);

tic;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1, Chp4 Baseline: Intra Coding
numPixel = numel(originalFrame(:, :, 1:3))./3;
fprintf('Chp4 Baseline: Intra Coding \n');
for scaleIdx = 1 : numel(scales)
    qScale   = scales(scaleIdx);
    bpp = 0;
    psnr_temp = 0;
    %% use pmf of lena_small to build and train huffman table
    lena_small  = IntraEncode(lena_small_ycbcr, qScale);
    range_small = -1000:4000;
    pmf = hist(lena_small(:), range_small);
    pmf = pmf ./ sum(pmf);
    [BinaryTree, HuffCode, BinCode, Codelengths] = buildHuffman(pmf);
    %% loop for all frames
    for i = 1 : videoLength
        foreman_ycbcr = originalYCbCr(:, :, (3*i-2):(3*i));
        k = IntraEncode(foreman_ycbcr, qScale);
        % huffman encoding
        lowerBound = -1000;  % based on the actual range of encoded image
        bytestream = enc_huffman_new(k-lowerBound+1, BinCode, Codelengths);
        % calculate bit rate
        bpp = bpp + (numel(bytestream)*8) / numPixel;
        % huffman decoding
        k_rec = double(reshape(dec_huffman_new(bytestream, BinaryTree,...
            max(size(k(:)))), size(k))) - 1 + lowerBound;
        %% image reconstruction
        I_rec = IntraDecode(k_rec, sz, qScale);
        I_rec = ictYCbCr2RGB(I_rec);
        psnr_temp = psnr_temp + calcPSNR(originalFrame(:, :, (3*i-2):(3*i)), I_rec);
    end
    %% calculate the overall performance
    bitrate_intraOld(scaleIdx) = bpp ./ videoLength;
    psnr_intraOld(scaleIdx) = psnr_temp ./ videoLength;
    fprintf('QP: %.1f bit-rate: %.2f bits/pixel PSNR: %.2fdB\n', ...
        qScale, bitrate_intraOld(scaleIdx), psnr_intraOld(scaleIdx));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2, Chp5 Baseline: Video Coding
fprintf('Chp5 Baseline: Video Coding \n');
for scaleIdx = 1 : numel(scales)
    qScale   = scales(scaleIdx);
    rate = zeros(1,videoLength);
    PSNR = zeros(1,videoLength);
    %% Part 1: encode the 1st frame using intra mode
    firstFrame = originalFrame(:, :, 1:3);  % RGB
    firstFrame_ycbcr = originalYCbCr(:, :, 1:3);  % YCbCr
    zeroRunFirstFrame = IntraEncode(firstFrame_ycbcr, qScale);
    %% Part 1: build Huffman table based on the first frame
    pmfFirstFrame = hist(zeroRunFirstFrame(:),min(zeroRunFirstFrame):max(zeroRunFirstFrame));
    pmfFirstFrame = pmfFirstFrame./sum(pmfFirstFrame);
    [BinaryTreeFirstFrame, HuffCodeFirstFrame, BinCodeFirstFrame, ...
        CodelengthsFirstFrame] = buildHuffman(pmfFirstFrame');
    %% Part 1: Huffman encoding and decoding
    % encoding
    lowerBoundFirstFrame = min(zeroRunFirstFrame);
    bytestreamZeroRun = enc_huffman_new(zeroRunFirstFrame-lowerBoundFirstFrame+1,...
        BinCodeFirstFrame, CodelengthsFirstFrame);
    % calculate bit rate for the 1st frame
    rate(1) = (numel(bytestreamZeroRun)*8) / (numel(firstFrame)/3);
    % decoding
    decZeroRun = double(reshape(dec_huffman_new(bytestreamZeroRun, BinaryTreeFirstFrame,...
        max(size(zeroRunFirstFrame(:)))), size(zeroRunFirstFrame))) - 1 + lowerBoundFirstFrame;
    %% Part 1: image reconstruction
    finalRecFrame = IntraDecode(decZeroRun, size(firstFrame), qScale);  % YCbCr
    finalRecFrameRGB = ictYCbCr2RGB(finalRecFrame);
    PSNR(1) = calcPSNR(firstFrame, finalRecFrameRGB);
    
    %% Part 2: encode the rest of frames
    rangeHuffman = -1000 : 4000;  % global lower & upper bound
    for i = 2 : videoLength
        refYCbCr = finalRecFrame;  % previous reconst. frame
        currentRGB = originalFrame(:, :, (3*i-2):(3*i));
        currentYCbCr = originalYCbCr(:, :, (3*i-2):(3*i));
        MV = SSD(refYCbCr(:,:,1), currentYCbCr(:,:,1));  % only Y component
        mcp = SSD_rec(refYCbCr, MV);  % predicted current img, YCbCr
        errYCbCr = currentYCbCr - mcp;  % prediction error
        zeroRun = IntraEncode(errYCbCr, qScale);
        %% Part 2: build Huffman table, compute only once
        if i == 2
            pmfMV = hist(MV(:),rangeHuffman);
            pmfMV = pmfMV./sum(pmfMV);
            pmfZeroRun = hist(zeroRun(:),rangeHuffman);
            pmfZeroRun = pmfZeroRun ./sum(pmfZeroRun);
            [BinaryTreeMV, HuffCodeMV, BinCodeMV, CodelengthsMV] = buildHuffman(pmfMV);
            [BinaryTreeZeroRun, HuffCodeZeroRun, BinCodeZeroRun, ...
                CodelengthsZeroRun] = buildHuffman(pmfZeroRun);
        end
        %% Part 2: use trained table to encode & decode MV and residual image
        % encoding
        lowerBoundHuffman = -1000;  % global lowerbound
        bytestreamZeroRun = enc_huffman_new(zeroRun-lowerBoundHuffman+1,...
            BinCodeZeroRun, CodelengthsZeroRun);
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
        mcp2 = SSD_rec(refYCbCr, decMV);
        finalRecFrame = IntraDecode(decZeroRun, size(currentYCbCr), qScale) + mcp2;  % YCbCr
        finalRecFrameRGB = ictYCbCr2RGB(finalRecFrame);
        rate(i) = rateZeroRun + rateMV;
        PSNR(i) = calcPSNR(currentRGB, finalRecFrameRGB);
        
    end
    %% Final: evaluation on the whole sequence: average PSNR and Bitrate
    bitrate_videoOld(scaleIdx) = mean(rate);
    psnr_videoOld(scaleIdx) = mean(PSNR);
    fprintf('QP: %.1f bit-rate: %.2f bits/pixel PSNR: %.2fdB\n', ...
        qScale, bitrate_videoOld(scaleIdx), psnr_videoOld(scaleIdx))
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3, Intra Coding Optimization: Intra Prediction
numPixel = numel(originalFrame(:, :, 1:3))./3;
[height, width, ~] = size(originalFrame(:, :, 1:3));
fprintf('Intra coding optimization: Intra prediction \n');
for scaleIdx = 1 : numel(scales)
    qScale = scales(scaleIdx);
    bpp = 0;
    psnr_temp = 0;
    %% loop for all frames
    for i = 1 : videoLength
        %% Intra prediction: encoder side
        foreman_ycbcr = originalYCbCr(:, :, (3*i-2):(3*i));
        % intra luma prediction: using 4x4 block
        imgPaddedY = imgPadding(foreman_ycbcr, 'luma');
        numRowBlkY = height./4;
        numColBlkY = width./4;
        modeMatrixY = zeros(numRowBlkY, numColBlkY);
        residualY = zeros(height,width); % Intra prediction error image of Y
        for row = 2:4:height+1
            for col = 2:4:width+1
                currentBlk = imgPaddedY(row:row+3, col:col+3);
                % try the 9 possible modes and find the best prediction
                [predBlk,mode] = intraPredLuma(imgPaddedY, row, col);
                % record the prediction error
                residualY(row-1:row+2,col-1:col+2) = currentBlk - predBlk;
                % record the used mode (0~8)
                modeMatrixY(ceil(row/4), ceil(col/4)) = mode;
            end
        end
        % intra chroma prediction: using 8x8 block
        imgPaddedCb = imgPadding(foreman_ycbcr, 'cb');
        imgPaddedCr = imgPadding(foreman_ycbcr, 'cr');
        numRowBlkC = height./8;
        numColBlkC = width./8;
        modeMatrixCb = zeros(numRowBlkC, numColBlkC);
        modeMatrixCr = zeros(numRowBlkC, numColBlkC);
        residualCb = zeros(height,width); % Intra prediction error image of Cb
        residualCr = zeros(height,width); % Intra prediction error image of Cr
        for row = 2:8:height+1
            for col = 2:8:width+1
                currentBlkCb = imgPaddedCb(row:row+7,col:col+7);
                currentBlkCr = imgPaddedCr(row:row+7,col:col+7);
                % try the 4 possible modes and find the best prediction
                [predBlkCb,modeCb] = intraPredChroma(imgPaddedCb,row,col);
                [predBlkCr,modeCr] = intraPredChroma(imgPaddedCr,row,col);
                % record the prediction error
                residualCb(row-1:row+6,col-1:col+6) = currentBlkCb - predBlkCb;
                residualCr(row-1:row+6,col-1:col+6) = currentBlkCr - predBlkCr;
                % record the used mode (0~3)
                modeMatrixCb(ceil(row/8), ceil(col/8)) = modeCb;
                modeMatrixCr(ceil(row/8), ceil(col/8)) = modeCr;
            end
        end
        % encode the complete prediction error in YCbCr
        residual = cat(3, residualY, residualCb, residualCr);
        result = foreman_ycbcr - residual;
        residualEncode = IntraEncode(residual, qScale);
        % store the mode information in 1D array
        modeArray = [modeMatrixY(:)', modeMatrixCb(:)', modeMatrixCr(:)'];
        % store the head data (most top and left lines' pixel values)
        imgPaddedHead = [imgPaddedY(1,:), imgPaddedY(2:end, 1)', ...
            imgPaddedCb(1,:), imgPaddedCb(2:end, 1)', ...
            imgPaddedCr(1,:), imgPaddedCr(2:end, 1)'];
        imgPaddedHead = round(imgPaddedHead);
        %% build Huffman table, only do it for once
        if i == 1
            range_residual = -1000:4000;
            pmfRes = hist(residualEncode(:), range_residual);
            pmfRes = pmfRes ./ sum(pmfRes);
            [BinaryTreeRes, HuffCodeRes, BinCodeRes, CodelengthsRes] = buildHuffman(pmfRes);
            
            range_mode = 0:8;
            pmfMode = hist(modeArray(:), range_mode);
            pmfMode = pmfMode ./ sum(pmfMode);
            [BinaryTreeMode, HuffCodeMode, BinCodeMode, CodelengthsMode] ...
                = buildHuffman(pmfMode);
        end
        %% use trained table to get the bytestream
        % Huffman encoding
        lowerBound = -1000;
        resBytestream = enc_huffman_new(residualEncode-lowerBound+1, BinCodeRes, CodelengthsRes);
        headBytestream = enc_huffman_new(imgPaddedHead-lowerBound+1, BinCodeRes, CodelengthsRes);
        modeBytestream = enc_huffman_new(modeArray+1, BinCodeMode, CodelengthsMode);
        
        % calculate bit rate
        bppRes = (numel(resBytestream)*8) / numPixel;
        bppMode = (numel(modeBytestream)*8) / numPixel;
        bppHead = (numel(headBytestream)*8) / numPixel;
        bpp = bpp + bppRes + bppMode + bppHead;
        
        % Huffman decoding
        resRec = double(reshape(dec_huffman_new(resBytestream, BinaryTreeRes,...
            max(size(residualEncode(:)))), size(residualEncode))) - 1 + lowerBound;
        headRec = double(reshape(dec_huffman_new(headBytestream, BinaryTreeRes,...
            max(size(imgPaddedHead(:)))), size(imgPaddedHead))) - 1 + lowerBound;
        modeRec = double(reshape(dec_huffman_new(modeBytestream, BinaryTreeMode,...
            max(size(modeArray(:)))), size(modeArray))) - 1;
        
        %% Intra prediction: decoder side
        residualRec = IntraDecode(resRec, sz, qScale);
        residualY = residualRec(:,:,1);
        residualCb = residualRec(:,:,2);
        residualCr = residualRec(:,:,3);
        
        recY = zeros(height+1,width+5);
        numRowBlkY = height./4;
        numColBlkY = width./4;
        recCb = zeros(height+1,width+1);
        recCr = zeros(height+1,width+1);
        numRowBlkC = height./8;
        numColBlkC = width./8;
        % padding, add head information
        recY(1,:) = headRec(1:width+5);
        recY(2:end,1) = headRec(width+6:width+5+height)';
        recCb(1,:) = headRec(width+6+height:2*width+6+height);
        recCb(2:end,1) = headRec(2*width+7+height:2*width+6+2*height)';
        recCr(1,:) = headRec(2*width+7+2*height:3*width+7+2*height);
        recCr(2:end,1) = headRec(3*width+8+2*height:end)';
        
        modeMatrixYRec = reshape(modeRec(1:numRowBlkY*numColBlkY),[numRowBlkY,numColBlkY]);
        modeMatrixCbRec = reshape(modeRec(numRowBlkY*numColBlkY+1:...
            numRowBlkY*numColBlkY+numRowBlkC*numColBlkC),[numRowBlkC,numColBlkC]);
        modeMatrixCrRec = reshape(modeRec(numRowBlkY*numColBlkY+numRowBlkC*numColBlkC+1:...
            end),[numRowBlkC,numColBlkC]);
        % reconstruction for luma
        for row = 2:4:height+1
            for col = 2:4:width+1
                % get the used mode for encoding current block
                mode = modeMatrixYRec(ceil(row/4), ceil(col/4));
                % based on mode and previous reconst. block to make
                % prediction
                predBlk = intraDecodeLuma(recY, row, col, mode);
                % update the reconstruction
                recY(row:row+3,col:col+3) = predBlk + residualY(row-1:row+2,col-1:col+2);
            end
        end
        % reconstruction for chroma
        for row = 2:8:height+1
            for col = 2:8:width+1
                modeCb = modeMatrixCbRec(ceil(row/8), ceil(col/8));
                modeCr = modeMatrixCrRec(ceil(row/8), ceil(col/8));
                currentBlkCb = intraDecodeChroma(recCb, row, col, modeCb);
                currentBlkCr = intraDecodeChroma(recCr, row, col, modeCr);
                recCb(row:row+7,col:col+7) = currentBlkCb + residualCb(row-1:row+6,col-1:col+6);
                recCr(row:row+7,col:col+7) = currentBlkCr + residualCr(row-1:row+6,col-1:col+6);
            end
        end
        rec = cat(3, recY(2:height+1, 2:width+1), recCb(2:height+1, 2:width+1), recCr(2:height+1, 2:width+1));
        I_rec = ictYCbCr2RGB(rec);
        psnr_temp = psnr_temp + calcPSNR(originalFrame(:, :, (3*i-2):(3*i)), I_rec);
    end
    bitrate_intraOpt(scaleIdx) = bpp ./ videoLength;
    psnr_intraOpt(scaleIdx) = psnr_temp ./ videoLength;
    fprintf('QP: %.1f bit-rate: %.2f bits/pixel PSNR: %.2fdB\n', ...
        qScale, bitrate_intraOpt(scaleIdx), psnr_intraOpt(scaleIdx))
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4, Intra Coding Optimization: Adaptive Post-deblocking
fprintf('Intra Opt: Adaptive Post-deblocking \n');
numPixel = numel(originalFrame(:, :, 1:3))./3;
% deblocking thresholds depend on different quantization level
% this is a pre-computed table including best choices
indexDeblocking = [1,1,1,1,1,28,29,35,39,45,51,51];
for scaleIdx = 1 : numel(scales)
    qScale   = scales(scaleIdx);
    bpp = 0;
    psnr_temp = 0;
    indexCurrent = indexDeblocking(scaleIdx);
    %% use pmf of lena_small to build and train huffman table
    lena_small  = IntraEncodeDeblock(lena_small_ycbcr, qScale);
    range_small = -1000:4000;
    pmf = hist(lena_small(:), range_small);
    pmf = pmf ./ sum(pmf);
    [BinaryTree, HuffCode, BinCode, Codelengths] = buildHuffman(pmf);
    %% loop for all frames
    for i = 1 : videoLength
        foreman_ycbcr = originalYCbCr(:, :, (3*i-2):(3*i));
        k = IntraEncodeDeblock(foreman_ycbcr, qScale);
        % huffman encoding
        lowerBound = -1000;  % based on the actual range of encoded image
        bytestream = enc_huffman_new(k-lowerBound+1, BinCode, Codelengths);
        % calculate bit rate
        bpp = bpp + (numel(bytestream)*8) / numPixel;
        % huffman decoding
        k_rec = double(reshape(dec_huffman_new(bytestream, BinaryTree,...
            max(size(k(:)))), size(k))) - 1 + lowerBound;
        %% image reconstruction
        I_rec = IntraDecodeDeblock(k_rec, sz, qScale);
        % use different deblocking threshold according to quantization
        % scale
        rec = deblock(I_rec, indexCurrent);
        finalRec = ictYCbCr2RGB(rec);
        currentPSNR = calcPSNR(originalFrame(:, :, (3*i-2):(3*i)), finalRec);
        % use the best deblocking result
        psnr_temp = psnr_temp + currentPSNR;
    end
    %% calculate the overall performance
    bitrate_intraDeblock(scaleIdx) = bpp ./ videoLength;
    psnr_intraDeblock(scaleIdx) = psnr_temp ./ videoLength;
    fprintf('QP: %.1f bit-rate: %.2f bits/pixel PSNR: %.2fdB\n', ...
        qScale, bitrate_intraDeblock(scaleIdx), psnr_intraDeblock(scaleIdx));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 5, Video Coding Optimization: Half-Pel
fprintf('Video Coding Optimization: Half-Pel \n');
for scaleIdx = 1 : numel(scales)
    qScale   = scales(scaleIdx);
    rate = zeros(1,videoLength);
    PSNR = zeros(1,videoLength);
    %% Part 1: encode the 1st frame using intra mode
    firstFrame = originalFrame(:, :, 1:3);  % RGB
    firstFrame_ycbcr = originalYCbCr(:, :, 1:3);  % YCbCr
    zeroRunFirstFrame = IntraEncode(firstFrame_ycbcr, qScale);
    %% Part 1: build Huffman table based on the first frame
    pmfFirstFrame = hist(zeroRunFirstFrame(:),min(zeroRunFirstFrame):max(zeroRunFirstFrame));
    pmfFirstFrame = pmfFirstFrame./sum(pmfFirstFrame);
    [BinaryTreeFirstFrame, HuffCodeFirstFrame, BinCodeFirstFrame, ...
        CodelengthsFirstFrame] = buildHuffman(pmfFirstFrame');
    %% Part 1: Huffman encoding and decoding
    % encoding
    lowerBoundFirstFrame = min(zeroRunFirstFrame);
    bytestreamZeroRun = enc_huffman_new(zeroRunFirstFrame-lowerBoundFirstFrame+1,...
        BinCodeFirstFrame, CodelengthsFirstFrame);
    % calculate bit rate for the 1st frame
    rate(1) = (numel(bytestreamZeroRun)*8) / (numel(firstFrame)/3);
    % decoding
    decZeroRun = double(reshape(dec_huffman_new(bytestreamZeroRun, BinaryTreeFirstFrame,...
        max(size(zeroRunFirstFrame(:)))), size(zeroRunFirstFrame))) - 1 + lowerBoundFirstFrame;
    %% Part 1: image reconstruction
    finalRecFrame = IntraDecode(decZeroRun, size(firstFrame), qScale);  % YCbCr
    finalRecFrameRGB = ictYCbCr2RGB(finalRecFrame);
    PSNR(1) = calcPSNR(firstFrame, finalRecFrameRGB);
    
    %% Part 2: encode the rest of frames
    rangeHuffman = -1000 : 4000;  % global lower & upper bound
    for i = 2 : videoLength
        refYCbCr = finalRecFrame;  % previous reconst. frame
        currentRGB = originalFrame(:, :, (3*i-2):(3*i));
        currentYCbCr = originalYCbCr(:, :, (3*i-2):(3*i));
        % perform half-pel motion estimation
        MV = halfPel(refYCbCr(:,:,1), currentYCbCr(:,:,1));  % only Y component
        mcp = halfPel_rec(refYCbCr, MV);  % predicted current img, YCbCr
        errYCbCr = currentYCbCr - mcp;  % prediction error
        zeroRun = IntraEncode(errYCbCr, qScale);
        %% Part 2: build Huffman table, compute only once
        if i == 2
            pmfMV = hist(MV(:),rangeHuffman);
            pmfMV = pmfMV./sum(pmfMV);
            pmfZeroRun = hist(zeroRun(:),rangeHuffman);
            pmfZeroRun = pmfZeroRun ./sum(pmfZeroRun);
            [BinaryTreeMV, HuffCodeMV, BinCodeMV, CodelengthsMV] = buildHuffman(pmfMV);
            [BinaryTreeZeroRun, HuffCodeZeroRun, BinCodeZeroRun, ...
                CodelengthsZeroRun] = buildHuffman(pmfZeroRun);
        end
        %% Part 2: use trained table to encode & decode MV and residual image
        % encoding
        lowerBoundHuffman = -1000;  % global lowerbound
        bytestreamZeroRun = enc_huffman_new(zeroRun-lowerBoundHuffman+1,...
            BinCodeZeroRun, CodelengthsZeroRun);
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
        mcp2 = halfPel_rec(refYCbCr, decMV);
        finalRecFrame = IntraDecode(decZeroRun, size(currentYCbCr), qScale) + mcp2;  % YCbCr
        finalRecFrameRGB = ictYCbCr2RGB(finalRecFrame);
        rate(i) = rateZeroRun + rateMV;
        PSNR(i) = calcPSNR(currentRGB, finalRecFrameRGB);
        
    end
    %% Final: evaluation on the whole sequence: average PSNR and Bitrate
    bitrate_halfPel(scaleIdx) = mean(rate);
    psnr_halfPel(scaleIdx) = mean(PSNR);
    fprintf('QP: %.1f bit-rate: %.2f bits/pixel PSNR: %.2fdB\n', ...
        qScale, bitrate_halfPel(scaleIdx), psnr_halfPel(scaleIdx))
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 6, Video Coding Optimization: Quarter-Pel
fprintf('Video Coding Optimization: Quarter-Pel \n');
for scaleIdx = 1 : numel(scales)
    qScale   = scales(scaleIdx);
    rate = zeros(1,videoLength);
    PSNR = zeros(1,videoLength);
    %% Part 1: encode the 1st frame using intra mode
    firstFrame = originalFrame(:, :, 1:3);  % RGB
    firstFrame_ycbcr = originalYCbCr(:, :, 1:3);  % YCbCr
    zeroRunFirstFrame = IntraEncode(firstFrame_ycbcr, qScale);
    %% Part 1: build Huffman table based on the first frame
    pmfFirstFrame = hist(zeroRunFirstFrame(:),min(zeroRunFirstFrame):max(zeroRunFirstFrame));
    pmfFirstFrame = pmfFirstFrame./sum(pmfFirstFrame);
    [BinaryTreeFirstFrame, HuffCodeFirstFrame, BinCodeFirstFrame, ...
        CodelengthsFirstFrame] = buildHuffman(pmfFirstFrame');
    %% Part 1: Huffman encoding and decoding
    % encoding
    lowerBoundFirstFrame = min(zeroRunFirstFrame);
    bytestreamZeroRun = enc_huffman_new(zeroRunFirstFrame-lowerBoundFirstFrame+1,...
        BinCodeFirstFrame, CodelengthsFirstFrame);
    % calculate bit rate for the 1st frame
    rate(1) = (numel(bytestreamZeroRun)*8) / (numel(firstFrame)/3);
    % decoding
    decZeroRun = double(reshape(dec_huffman_new(bytestreamZeroRun, BinaryTreeFirstFrame,...
        max(size(zeroRunFirstFrame(:)))), size(zeroRunFirstFrame))) - 1 + lowerBoundFirstFrame;
    %% Part 1: image reconstruction
    finalRecFrame = IntraDecode(decZeroRun, size(firstFrame), qScale);  % YCbCr
    finalRecFrameRGB = ictYCbCr2RGB(finalRecFrame);
    PSNR(1) = calcPSNR(firstFrame, finalRecFrameRGB);
    
    %% Part 2: encode the rest of frames
    rangeHuffman = -1000 : 4000;  % global lower & upper bound
    for i = 2 : videoLength
        refYCbCr = finalRecFrame;  % previous reconst. frame
        currentRGB = originalFrame(:, :, (3*i-2):(3*i));
        currentYCbCr = originalYCbCr(:, :, (3*i-2):(3*i));
        % perform quarter-pel motion estimation
        MV = quarterPel(refYCbCr(:,:,1), currentYCbCr(:,:,1));  % only Y component
        mcp = quarterPel_rec(refYCbCr, MV);  % predicted current img, YCbCr
        errYCbCr = currentYCbCr - mcp;  % prediction error
        zeroRun = IntraEncode(errYCbCr, qScale);
        %% Part 2: build Huffman table, compute only once
        if i == 2
            pmfMV = hist(MV(:),rangeHuffman);
            pmfMV = pmfMV./sum(pmfMV);
            pmfZeroRun = hist(zeroRun(:),rangeHuffman);
            pmfZeroRun = pmfZeroRun ./sum(pmfZeroRun);
            [BinaryTreeMV, HuffCodeMV, BinCodeMV, CodelengthsMV] = buildHuffman(pmfMV);
            [BinaryTreeZeroRun, HuffCodeZeroRun, BinCodeZeroRun, ...
                CodelengthsZeroRun] = buildHuffman(pmfZeroRun);
        end
        %% Part 2: use trained table to encode & decode MV and residual image
        % encoding
        lowerBoundHuffman = -1000;  % global lowerbound
        bytestreamZeroRun = enc_huffman_new(zeroRun-lowerBoundHuffman+1,...
            BinCodeZeroRun, CodelengthsZeroRun);
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
        mcp2 = quarterPel_rec(refYCbCr, decMV);
        finalRecFrame = IntraDecode(decZeroRun, size(currentYCbCr), qScale) + mcp2;  % YCbCr
        finalRecFrameRGB = ictYCbCr2RGB(finalRecFrame);
        rate(i) = rateZeroRun + rateMV;
        PSNR(i) = calcPSNR(currentRGB, finalRecFrameRGB);
        
    end
    %% Final: evaluation on the whole sequence: average PSNR and Bitrate
    bitrate_quarterPel(scaleIdx) = mean(rate);
    psnr_quarterPel(scaleIdx) = mean(PSNR);
    fprintf('QP: %.1f bit-rate: %.2f bits/pixel PSNR: %.2fdB\n', ...
        qScale, bitrate_quarterPel(scaleIdx), psnr_quarterPel(scaleIdx))
end


%% Save all results
toc;
save('result.mat','bitrate_halfPel','bitrate_intraDeblock','bitrate_intraOld',...
    'bitrate_intraOpt','bitrate_quarterPel','bitrate_videoOld','psnr_halfPel',...
    'psnr_intraDeblock','psnr_intraOld','psnr_intraOpt','psnr_quarterPel',...
    'psnr_videoOld');

%% Plot the RD-curve
figure(1);
intraOld = plot(bitrate_intraOld, psnr_intraOld, '-*', 'Color', 'k','LineWidth',1);
hold on
intraMacro = plot(bitrate_intraOpt, psnr_intraOpt, '-*', 'Color', 'c','LineWidth',1);
hold on
intraDebloc = plot(bitrate_intraDeblock, psnr_intraDeblock, '-*', 'Color', 'g','LineWidth',1);
hold on
videoOld = plot(bitrate_videoOld, psnr_videoOld, '-o', 'Color', 'k','LineWidth',1);
hold on
videoHalf = plot(bitrate_halfPel, psnr_halfPel, '-o', 'Color', 'b','LineWidth',1);
hold on
videoQuarter = plot(bitrate_quarterPel, psnr_quarterPel, '-o', 'Color', 'r','LineWidth',1);
hold on

ylim([25 50])
xlim([0 4.5])
hold on;
grid on;
xlabel('bitrate [bit/pixel]','fontsize',16);
ylabel('PSNR [dB]','fontsize',16);
legend([intraOld,intraMacro,intraDebloc,videoOld,videoHalf,videoQuarter], {'Chp4 Baseline: Intra',...
    'Intra Opt: Intra Prediction','Intra Opt: Adaptive Post-deblocking','Chp5 Baseline: Video','Video Opt: Half-pel','Video Opt: Quarter-pel'},...
    'fontsize',10, 'Location','best');
title('RD performance of Optimization, Foreman','fontsize',16);

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

%% Intra Encoding & Decoding for Deblocking Method
function dst = IntraEncodeDeblock(image, qScale)
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
    @(block_struct) Quant8x8Deblock(block_struct.data, qScale));

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

function dst = IntraDecodeDeblock(image, img_size, qScale)
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
img_IQuan = blockproc(img_IZigZag, [blockLength,blockLength], @(block_struct) DeQuant8x8Deblock(block_struct.data, qScale));

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

%% Quantization for Deblocking
function quant = Quant8x8Deblock(dct_block, qScale)
%  Input         : dct_block (Original Coefficients, 8x8x3), YCbCr
%                  qScale (Quantization Parameter, scalar)
%
%  Output        : quant (Quantized Coefficients, 8x8x3)
quant = zeros(8, 8, 3);
quanTable_L = [16 16 16 16 17 18 21 24;
    16 16 16 16 17 19 22 25;
    16 16 17 18 20 22 25 29;
    16 16 18 21 24 27 31 36;
    17 17 20 24 30 35 41 47;
    18 19 22 27 35 44 54 65;
    21 22 25 31 41 54 70 88;
    24 25 29 36 47 65 88 115];
quanTable_C = [17,18,24,47,99,99,99,99; 18,21,26,66,99,99,99,99;...
    24,13,56,99,99,99,99,99; 47,66,99,99,99,99,99,99;...
    99,99,99,99,99,99,99,99; 99,99,99,99,99,99,99,99;...
    99,99,99,99,99,99,99,99; 99,99,99,99,99,99,99,99];
quant(:, :, 1) = round(dct_block(:, :, 1) ./ (quanTable_L .* qScale));
quant(:, :, 2) = round(dct_block(:, :, 2) ./ (quanTable_C .* qScale));
quant(:, :, 3) = round(dct_block(:, :, 3) ./ (quanTable_C .* qScale));
end

function dct_block = DeQuant8x8Deblock(quant_block, qScale)
%  Function Name : DeQuant8x8.m
%  Input         : quant_block  (Quantized Block, 8x8x3)
%                  qScale       (Quantization Parameter, scalar)
%
%  Output        : dct_block    (Dequantized DCT coefficients, 8x8x3)
dct_block = zeros(8, 8, 3);
quanTable_L = [16 16 16 16 17 18 21 24;
    16 16 16 16 17 19 22 25;
    16 16 17 18 20 22 25 29;
    16 16 18 21 24 27 31 36;
    17 17 20 24 30 35 41 47;
    18 19 22 27 35 44 54 65;
    21 22 25 31 41 54 70 88;
    24 25 29 36 47 65 88 115];
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