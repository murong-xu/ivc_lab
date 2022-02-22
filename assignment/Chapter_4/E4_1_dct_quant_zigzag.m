%% E4.1.a
imageLena = double(imread('./IVC_labs_starting_point/data/images/lena.tif'));
% load('block1.mat');
block1 = imageLena(1:8, 1:8, :);
DCT8x8(block1);
% load('block2.mat');
block2 = imageLena(401:408, 401:408, :);
DCT8x8(block2);
% load('block3.mat');
block3 = imageLena(201:208, 201:208, :);
DCT8x8(block3);
fprintf("The syntax of the code seems to be correct, next run the assessment to verify the correctness");

%% E4.1.a
block1 = randi(255,8,8,3);
coeff1 = DCT8x8(block1);
rec1   = IDCT8x8(coeff1);
fprintf("The syntax of the code seems to be correct, next run the assessment to verify the correctness");

%% E4.1.b
dct_block_1 = Quant8x8(coeff1, 1);
dct_block_2 = Quant8x8(coeff1, 2);
fprintf("The syntax of the code seems to be correct, next run the assessment to verify the correctness");

%% E4.1.b
dct_block = randi(255,8,8,3);
q_block   = Quant8x8(dct_block, 1);
rec_block = DeQuant8x8(q_block, 1);
fprintf("The syntax of the code seems to be correct, next run the assessment to verify the correctness");

%% E4.1.c
% % load('quant.mat');
% ZigZag8x8(q_block);
% fprintf("The syntax of the code seems to be correct, next run the assessment to verify the correctness");

%% E4.1.c
quant = randi(255,8,8,3);
zz    = ZigZag8x8(quant);
coeff = DeZigZag8x8(zz);
fprintf("The syntax of the code seems to be correct, next run the assessment to verify the correctness");

%% E4.1.a
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
%% E4.2 a
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

%% E4.1.b
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

%% E4.1.b
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

%% E4.1.c
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
    quantBlock = quant(:, :, channel)
    zz(ZigZag(:),  channel) = quantBlock(:);
end
end

%% E4.1.c
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