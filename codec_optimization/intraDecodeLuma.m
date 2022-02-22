function [finalPred] = intraDecodeLuma(img, row, col, mode)
% [finalPred] = intraDecodeLuma(img, row, col, mode)
% make intra prediction using 4x4 block based on the given prediction
% mode in range [0,8].
% Simplified version of H.264 standard.
%
% Input:
% img  : the whole image
% row  : current row coordinate in img (top left of block)
% col  : current column coordinate in img
% mode : the used mode for prediction, in range of [0,8].
%
% Output:
% finalPred: prediction result of current block

finalPred = zeros(4,4);
switch mode
    case 0
        % Vertical Mode
        for c = 1:4
            finalPred(:,c) = img(row-1,col+c-1);
        end
    case 1
        % Horizontal Mode
        for r = 1:4
            finalPred(r,:) = img(row+r-1, col-1);
        end
    case 2
        % DC (Average) Mode
        s = sum(img(row-1,col:col+3)) + sum(img(row:row+3,col-1));
        finalPred(:,:) = round(s/8);
    case 3
        % Diagonal down left Mode
        for r = 1:4
            c = r+3;
            finalPred(r,1:4) = img(row-1, col+r:col+c);
        end
    case 4
        % Diagonal down right Mode
        finalPred(4,1) = img(row+2,col-1);
        finalPred(1,4) = img(row-1,col+2);
        for r = 1:4
            finalPred(r,r) = img(row-1,col-1);
        end
        for r = 2:4
            finalPred(r,r-1) = img(row,col-1);
        end
        for r = 3:4
            finalPred(r,r-2) = img(row+1,col-1);
        end
        for c = 2:4
            finalPred(c-1,c) = img(row-1,col);
        end
        for c = 3:4
            finalPred(c-2,c) = img(row-1,col+1);
        end
    case 5
        % Vertical right Mode
        finalPred(1,4) = img(row-1,col+3);
        finalPred(4,1) = img(row+1,col-1);
        for r = 1:4
            finalPred(r,r) = img(row-1,col);
        end
        for r = 1:3
            finalPred(r,r+1) = img(row-1,col+1);
        end
        for r = 1:2
            finalPred(r,r+2) = img(row-1,col+2);
        end
        for r = 2:4
            finalPred(r,r-1) = img(row-1,col-1);
        end
        for r = 3:4
            finalPred(r,r-2) = img(row,col-1);
        end
    case 6
        % horizontal down Mode
        finalPred(4,1) = img(row+3,col-1);
        finalPred(1,4) = img(row-1,col+1);
        for r = 1:4
            finalPred(r,r) = img(row,col-1);
        end
        for r = 2:4
            finalPred(r,r-1) = img(row+1,col-1);
        end
        for r = 3:4
            finalPred(r,r-2) = img(row+2,col-1);
        end
        for r = 1:3
            finalPred(r,r+1) = img(row-1,col-1);
        end
        for r = 1:2
            finalPred(r,r+2) = img(row-1,col);
        end
    case 7
        % Vertical left Mode
        finalPred(1,1) = img(row-1,col);
        finalPred(1,2) = img(row-1,col+1);
        finalPred(2,1) = img(row-1,col+1);
        finalPred(3,1) = img(row-1,col+1);
        finalPred(1,3) = img(row-1,col+2);
        finalPred(2,2) = img(row-1,col+2);
        finalPred(3,2) = img(row-1,col+2);
        finalPred(4,1) = img(row-1,col+2);
        finalPred(1,4) = img(row-1,col+3);
        finalPred(2,3) = img(row-1,col+3);
        finalPred(3,3) = img(row-1,col+3);
        finalPred(4,2) = img(row-1,col+3);
        finalPred(2,4) = img(row-1,col+4);
        finalPred(3,4) = img(row-1,col+4);
        finalPred(4,3) = img(row-1,col+4);
        finalPred(4,4) = img(row-1,col+5);
    case 8
        % horizontal up Mode
        finalPred(1,1) = img(row,col-1);
        finalPred(2,1) = img(row+1,col-1);
        finalPred(1,2) = img(row+1,col-1);
        finalPred(1,3) = img(row+1,col-1);
        finalPred(3,1) = img(row+2,col-1);
        finalPred(2,2) = img(row+2,col-1);
        finalPred(2,3) = img(row+2,col-1);
        finalPred(1,4) = img(row+2,col-1);
        finalPred(4,1) = img(row+3,col-1);
        finalPred(3,2) = img(row+3,col-1);
        finalPred(3,3) = img(row+3,col-1);
        finalPred(2,4) = img(row+3,col-1);
        finalPred(4,2) = img(row+3,col-1);
        finalPred(4,3) = img(row+3,col-1);
        finalPred(3,4) = img(row+3,col-1);
        finalPred(4,4) = img(row+3,col-1);
end