function [finalPred,mode] = intraPredLuma(img, row, col)
% [finalPred,mode] = intraPredLuma(img, predBlk, row, col)
% make intra prediction using 4x4 block. Based on the 9 possible prediction
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
% mode   : integer number in range [0,8]

SAD = 10000000;  % initialization
finalPred = zeros(4, 4);
currentBlk = img(row:row+3, col:col+3);

% loop for all possible modes
for modeNr = 0 : 8
    predBlk = zeros(4,4);
    switch modeNr
        case 0  % vertical
            for c = 1:4
                predBlk(:,c) = img(row-1, col+c-1);
            end
            d = sum(sum(abs(predBlk - currentBlk)));
            if d < SAD
                SAD = d;
                finalPred = predBlk;
                mode = 0;
            end
        case 1  % horizontal
            for r = 1:4
                predBlk(r,:) = img(row+r-1, col-1);
            end
            d = sum(sum(abs(predBlk - currentBlk)));
            if d < SAD
                SAD = d;
                finalPred = predBlk;
                mode = 1;
            end
        case 2  % DC
            s = sum(img(row-1,col:col+3)) + sum(img(row:row+3,col-1));
            predBlk(:,:) = round(s/8);
            d = sum(sum(abs(predBlk - currentBlk)));
            if d < SAD
                SAD = d;
                finalPred = predBlk;
                mode = 2;
            end
        case 3  % diagonal down-left
            for r = 1:4
                c = r + 3;
                predBlk(r,:) = img(row-1, col+r:col+c);
            end
            d = sum(sum(abs(predBlk - currentBlk)));
            if d < SAD
                SAD = d;
                finalPred = predBlk;
                mode = 3;
            end
        case 4  % diagonal down-right
            predBlk(4,1) = img(row+2,col-1);
            predBlk(1,4) = img(row-1,col+2);
            for r = 1:4
                predBlk(r,r) = img(row -1,col-1);
            end
            for r = 2:4
                predBlk(r,r-1) = img(row,col-1);
            end
            for r = 3:4
                predBlk(r,r-2) = img(row+1,col-1);
            end
            for c = 2:4
                predBlk(c-1,c) = img(row-1,col);
            end
            for c = 3:4
                predBlk(c-2,c) = img(row-1,col+1);
            end
            d = sum(sum(abs(predBlk - currentBlk)));
            if d < SAD
                SAD = d;
                finalPred = predBlk;
                mode = 4;
            end
        case 5  % vertical-right
            predBlk(1,4) = img(row-1,col+3);
            predBlk(4,1) = img(row+1,col-1);
            for r = 1:4
                predBlk(r,r) = img(row-1,col);
            end
            for r = 1:3
                predBlk(r,r+1) = img(row-1,col+1);
            end
            for r = 1:2
                predBlk(r,r+2) = img(row-1,col+2);
            end
            for r = 2:4
                predBlk(r,r-1) = img(row-1,col-1);
            end
            for r = 3:4
                predBlk(r,r-2) = img(row,col-1);
            end
            d = sum(sum(abs(predBlk - currentBlk)));
            if d < SAD
                SAD = d;
                finalPred = predBlk;
                mode = 5;
            end
        case 6  % horizontal-down
            predBlk(4,1) = img(row+3,col-1);
            predBlk(1,4) = img(row-1,col+1);
            for r = 1:4
                predBlk(r,r) = img(row,col-1);
            end
            for r = 2:4
                predBlk(r,r-1) = img(row+1,col-1);
            end
            for r = 3:4
                predBlk(r,r-2) = img(row+2,col-1);
            end
            for r = 1:3
                predBlk(r,r+1) = img(row-1,col-1);
            end
            for r = 1:2
                predBlk(r,r+2) = img(row-1,col);
            end
            d = sum(sum(abs(predBlk - currentBlk)));
            if d < SAD
                SAD = d;
                finalPred = predBlk;
                mode = 6;
            end
        case 7  % vertical-left
            predBlk(1,1) = img(row-1,col);
            predBlk(1,2) = img(row-1,col+1);
            predBlk(2,1) = img(row-1,col+1);
            predBlk(3,1) = img(row-1,col+1);
            predBlk(1,3) = img(row-1,col+2);
            predBlk(2,2) = img(row-1,col+2);
            predBlk(3,2) = img(row-1,col+2);
            predBlk(4,1) = img(row-1,col+2);
            predBlk(1,4) = img(row-1,col+3);
            predBlk(2,3) = img(row-1,col+3);
            predBlk(3,3) = img(row-1,col+3);
            predBlk(4,2) = img(row-1,col+3);
            predBlk(2,4) = img(row-1,col+4);
            predBlk(3,4) = img(row-1,col+4);
            predBlk(4,3) = img(row-1,col+4);
            predBlk(4,4) = img(row-1,col+5);
            d = sum(sum(abs(predBlk - currentBlk)));
            if d < SAD
                SAD = d;
                finalPred = predBlk;
                mode = 7;
            end
        case 8  % horizontal-up
            predBlk(1,1) = img(row,col-1);
            predBlk(2,1) = img(row+1,col-1);
            predBlk(1,2) = img(row+1,col-1);
            predBlk(1,3) = img(row+1,col-1);
            predBlk(3,1) = img(row+2,col-1);
            predBlk(2,2) = img(row+2,col-1);
            predBlk(2,3) = img(row+2,col-1);
            predBlk(1,4) = img(row+2,col-1);
            predBlk(4,1) = img(row+3,col-1);
            predBlk(3,2) = img(row+3,col-1);
            predBlk(3,3) = img(row+3,col-1);
            predBlk(2,4) = img(row+3,col-1);
            predBlk(4,2) = img(row+3,col-1);
            predBlk(4,3) = img(row+3,col-1);
            predBlk(3,4) = img(row+3,col-1);
            predBlk(4,4) = img(row+3,col-1);
            d = sum(sum(abs(predBlk - currentBlk)));
            if d < SAD
                SAD = d;
                finalPred = predBlk;
                mode = 8;
            end
    end
end