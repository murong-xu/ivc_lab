function [finalPred] = intraDecodeChroma(img, row, col, mode)
% [finalPred] = intraDecodeChroma(img, row, col, mode)
% make intra prediction using 8x8 block based on the given prediction
% mode in range [0,3].
% Simplified version of H.264 standard.
%
% Input:
% img  : the whole image
% row  : current row coordinate in img (top left of block)
% col  : current column coordinate in img
% mode : the used mode for prediction, in range of [0,3].
%
% Output:
% finalPred: prediction result of current block

finalPred = zeros(8,8);
switch mode
    case 0
        % Vertical Mode
        for c = 1:8
            finalPred(:,c) = img(row-1,col+c-1);
        end
    case 1
        % Horizontal Mode
        for r = 1:8
            finalPred(r,:) = img(row+r-1, col-1);
        end
    case 2
        % DC (Average) Mode
        s = sum(img(row-1,col:col+7)) + sum(img(row:row+7,col-1));
        finalPred(:,:) = round(s/16);
    case 3
        %Plane for N=8
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
                finalPred(i+1,j+1) = (a+b*(j-3)+c*(i-3)+16)./32;
            end
        end
end