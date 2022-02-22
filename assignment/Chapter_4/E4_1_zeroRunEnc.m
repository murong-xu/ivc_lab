%% E4.1.d
EOB = 1000;
A = [1 0 0 2 3 4 5 1 3 0 0 0 0 0 0 0 0 1 2 3 4 0 0 0 0 0 1  4 6 8 2,...
    4 30 0 0 0 0 2 3 5 6 7 8 9 0 3 8 2 9 29 2 0 0 0 0 0 1 3 9 4 3 0 0 0];

B = [1 0 0 2 3 4 5 1 3 0 0 0 0 0 0 0 0 1 2 3 4 0 0 0 0 0 1  4 6 8 2,...
    4 30 0 0 0 0 2  3 5 6 7 8 9 0 3 8 2 9 29 2 0 0 0 0 0 1 3 9 4 3 0 0 1];

C = [1 0 0 2 3 4 5 1 3 0 0 0 0 0 0 0 0 1 2 3 4 0 0 0 0 0 1  4 6 8 2,...
    4 30 0 0 0 0 2 3 5 6 7 8 9 0 3 8 2 9 29 2 0 0 0 0 0 1 3 9 4 3 0 1 0];

D = [2 -2 3 1 -1 2 0 0 -1 2 -1 -1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0, ...
    0 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0, ...
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0, ...
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];

A_solution = [1 0 1 2 3 4 5 1 3 0 7 1 2 3 4 0 4 1  4 6 8 2,...
    4 30 0 3 2 3 5 6 7 8 9 0 0 3 8 2 9 29 2 0 4 1 3 9 4 3 EOB];

B_solution = [1 0 1 2 3 4 5 1 3 0 7 1 2 3 4 0 4 1  4 6 8 2,...
    4 30 0 3 2 3 5 6 7 8 9 0 0 3 8 2 9 29 2 0 4 1 3 9 4 3 0 1 1];

C_solution = [1 0 1 2 3 4 5 1 3 0 7 1 2 3 4 0 4 1  4 6 8 2,...
    4 30 0 3 2 3 5 6 7 8 9 0 0 3 8 2 9 29 2 0 4 1 3 9 4 3 0 0 1 EOB];

D_solution = [2 -2 3 1 -1 2 0 1 -1 2 -1 -1 1 0 24 -1 0 23 -1 EOB];

A_self = ZeroRunEnc_EoB(A, EOB);
B_self = ZeroRunEnc_EoB(B, EOB);
C_self = ZeroRunEnc_EoB(C, EOB);
D_self = ZeroRunEnc_EoB(D, EOB);

% test on a sequence
load('foreman10_residual_zero_run.mat');
load('foreman10_residual_zig_zag.mat');
zero_run_enc = ZeroRunEnc_EoB(foreman10_residual_zig_zag, EOB);
fprintf("Your solution:\n");
zero_run_enc(:,100);
fprintf("Our solution:\n");
foreman10_residual_zero_run(:,100);

fprintf("The syntax of the code seems to be correct, next run the assessment to verify the correctness");

%% E4.1.d
EOB = 1000;
A = [1 0 1 2 3 4 5 1 3 0 7 1 2 3 4 0 4 1  4 6 8 2,...
    4 30 0 3 2 3 5 6 7 8 9 0 0 3 8 2 9 29 2 0 4 1 3 9 4 3 EOB];

B = [1 0 1 2 3 4 5 1 3 0 7 1 2 3 4 0 4 1  4 6 8 2,...
    4 30 0 3 2 3 5 6 7 8 9 0 0 3 8 2 9 29 2 0 4 1 3 9 4 3 0 1 1];

C = [1 0 1 2 3 4 5 1 3 0 7 1 2 3 4 0 4 1  4 6 8 2,...
    4 30 0 3 2 3 5 6 7 8 9 0 0 3 8 2 9 29 2 0 4 1 3 9 4 3 0 0 1 EOB];

A_solution = [1 0 0 2 3 4 5 1 3 0 0 0 0 0 0 0 0 1 2 3 4 0 0 0 0 0 1  4 6 8 2,...
    4 30 0 0 0 0 2 3 5 6 7 8 9 0 3 8 2 9 29 2 0 0 0 0 0 1 3 9 4 3 0 0 0];

B_solution = [1 0 0 2 3 4 5 1 3 0 0 0 0 0 0 0 0 1 2 3 4 0 0 0 0 0 1  4 6 8 2,...
    4 30 0 0 0 0 2  3 5 6 7 8 9 0 3 8 2 9 29 2 0 0 0 0 0 1 3 9 4 3 0 0 1];

C_solution = [1 0 0 2 3 4 5 1 3 0 0 0 0 0 0 0 0 1 2 3 4 0 0 0 0 0 1  4 6 8 2,...
    4 30 0 0 0 0 2 3 5 6 7 8 9 0 3 8 2 9 29 2 0 0 0 0 0 1 3 9 4 3 0 1 0];

% Run learner solution.
zzA = ZeroRunDec_EoB(A, EOB);
zzB = ZeroRunDec_EoB(B, EOB);
zzC = ZeroRunDec_EoB(C, EOB);

assert(isequal(length(A_solution),length(zzA)), 'The length of reconstructed A is %d, but the length of original A is %d', length(zzA), length(A));
assert(isequal(length(B_solution),length(zzB)), 'The length of reconstructed B is %d, but the length of original B is %d', length(zzB), length(B));
assert(isequal(length(C_solution),length(zzC)), 'The length of reconstructed C is %d, but the length of original C is %d', length(zzC), length(C));

% test on a sequence
load('foreman10_residual_zero_run.mat');
load('foreman10_residual_zig_zag.mat');
zero_run_enc = ZeroRunEnc_EoB(foreman10_residual_zig_zag, EOB);
decoded = ZeroRunDec_EoB(zero_run_enc, EOB);

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
    quantBlock = quant(:, :, channel);
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

%% E4.1.d
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

%% E4.1.d
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