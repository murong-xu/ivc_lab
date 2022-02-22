function [finalPred,mode] = intraPredChroma(img, row, col)
% [finalPred,mode] = intraPredChroma(img, row, col)
% make intra prediction using 8x8 block. Based on the 4 possible prediction
% modes and select the best one with lowest SAD compared to current block.
% Simplified version of H.264 standard.
%
% Input:
% img  : the whole image
% row  : current row coordinate in img (top left of block)
% col  : current column coordinate in img
%
% Output:
% finalPred: prediction result of current block
% mode   : integer number in range [0,3]

SAD = 10000000;  % initialization
finalPred = zeros(8,8);
currentBlk = img(row:row+7, col:col+7);

% loop for all possible modes
for modeNr = 0:3
    predBlk = zeros(8,8);
    switch modeNr
        case 0  % vertical
            for c = 1:8
                predBlk(:,c) = img(row-1, col+c-1);
            end
            d = sum(sum(abs(predBlk - currentBlk)));
            if d < SAD
                SAD = d;
                finalPred = predBlk;
                mode = 0;
            end
        case 1  % horizontal
            for r = 1:8
                predBlk(r,:) = img(row+r-1, col-1);
            end
            d = sum(sum(abs(predBlk - currentBlk)));
            if d < SAD
                SAD = d;
                finalPred = predBlk;
                mode = 1;
            end
        case 2  % DC
            s = sum(img(row-1,col:col+7)) + sum(img(row:row+7,col-1));
            predBlk(:,:) = round(s/16);
            d = sum(sum(abs(predBlk - currentBlk)));
            if d < SAD
                SAD = d;
                finalPred = predBlk;
                mode = 2;
            end
        case 3  % plane
            H = (img(row-1, col+4)-img(row-1, col+2))...
                + 2*(img(row-1, col+5)-img(row-1, col+1))...
                + 3*(img(row-1, col+6)-img(row-1, col))...
                + 4*(img(row-1, col+7)-img(row-1, col-1));
            V = (img(row+4, col-1)-img(row+2, col-1))...
                + 2*(img(row+5, col-1)-img(row+1, col-1))...
                + 3*(img(row+6, col-1)-img(row, col-1))...
                + 4*(img(row+7, col-1)-img(row-1, col-1));
            a = 16*(img(row+7, col-1)+img(row-1, col+7));
            b = (5*H+32)./64;
            c = (5*V+32)./64;
            for i = 0:7
                for j = 0:7
                    predBlk(i+1,j+1) = (a+b*(j-3)+c*(i-3)+16)./32;
                end
            end
            d = sum(sum(abs(predBlk - currentBlk)));
            if d < SAD
                SAD = d;
                finalPred = predBlk;
                mode = 3;
            end
    end
end