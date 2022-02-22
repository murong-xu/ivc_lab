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