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
