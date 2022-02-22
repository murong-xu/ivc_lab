%% E5.1: Encode the frame using still image compression

videoFolder = './sequences/foreman20_40_RGB/';
videoPath  = dir([videoFolder, '*.bmp']);
videoLength = length(videoPath);
originalFrame = cell(videoLength, 1);
for i = 1:videoLength
    originalFrame{i}= double(imread([videoFolder, videoPath(i).name]));
end
originalYCbCr = cellfun(@ictRGB2YCbCr, originalFrame, 'Uniform', false);

lena_small = double(imread('./IVC_labs_starting_point/data/images/lena_small.tif'));
lena_small_ycbcr = ictRGB2YCbCr(lena_small);

scales = [0.07, 0.1, 0.2, 0.4, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 4.5];
for scaleIdx = 1 : numel(scales)
    qScale   = scales(scaleIdx);
    bpp = 0;
    psnr_temp = 0;
    lena_small  = IntraEncode(lena_small_ycbcr, qScale);
    %% use pmf of lena_small to build and train huffman table
    range_small = -1000:4000;  % based on the actual range of encoded image
    pmf = hist(lena_small(:), range_small);
    pmf = pmf ./ sum(pmf);  % use marg. PMF for 1D
    % Huffman Table
    [BinaryTree, HuffCode, BinCode, Codelengths] = buildHuffman(pmf);
    for i = 1 : videoLength
        % k and k_small: run-level encoded results in 1D vector [Y, Cb, Cr]
        foreman_ycbcr = originalYCbCr{i};
        k        = IntraEncode(foreman_ycbcr, qScale);
        %% use trained table to encode k to get the bytestream
        % encoding
        lowerBound = -1000;  % based on the actual range of encoded image
        bytestream = enc_huffman_new(k-lowerBound+1, BinCode, Codelengths);
        
        % calculate bit rate
        bpp = bpp + (numel(bytestream)*8) / (numel(originalFrame{i})./3);
        
        % decoding
        k_rec = double(reshape(dec_huffman_new(bytestream, BinaryTree,...
            max(size(k(:)))), size(k))) - 1 + lowerBound;
        %% image reconstruction
        I_rec = IntraDecode(k_rec, size(originalFrame{i}), qScale);
        I_rec = ictYCbCr2RGB(I_rec);
        psnr_temp = psnr_temp + calcPSNR(originalFrame{i}, I_rec);
    end
    bitPerPixel(scaleIdx) = bpp ./ videoLength;
    PSNR(scaleIdx) = psnr_temp ./ videoLength;
    fprintf('QP: %.1f bit-rate: %.2f bits/pixel PSNR: %.2fdB\n', ...
        qScale, bitPerPixel(scaleIdx), PSNR(scaleIdx))
end