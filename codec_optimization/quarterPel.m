function motion_vectors_indices = quarterPel(ref_image, image)
% motion_vectors_indices = quarterPel(ref_image, image)
% First do the integer-search, then performing half-pel and quarter-pel
% search based on the previous found best integer result.
%  Input         : ref_image(Reference Image, size: height x width)
%                  image (Current Image, size: height x width)
%
%  Output        : motion_vectors_indices

height = size(image, 1);
width = size(image, 2);
numBlockX = width ./ 8;
numBlockY = height ./ 8;
motion_vectors_indices = zeros(numBlockY, numBlockX);

positionBlockX = 1 : 8 : width;  % position of block
positionBlockY = 1 : 8 : height;

% bi-linear interpolation for the whole image
[x1,y1] = meshgrid(1:width, 1:height);
[x2,y2] = meshgrid(1:0.5:width, 1:0.5:height);
[x4,y4] = meshgrid(1:0.25:width, 1:0.25:height);
refInterp2 = interp2(x1,y1,ref_image,x2,y2,'linear');
% "quarter-linear" interpolation for the whole image
refInterp4 = interp2(x2,y2,refInterp2,x4,y4,'linear');
% define search range for half-pel
searchPattern2 = zeros(2, 4);
searchPattern2(:,1) = [-0.5, -0.5];
searchPattern2(:,2) = [-0.5, 0.5];
searchPattern2(:,3) = [0.5, -0.5];
searchPattern2(:,4) = [0.5, 0.5];
% define search range for quarter-pel
searchPattern4 = zeros(2, 4);
searchPattern4(:,1) = [-0.25, -0.25];
searchPattern4(:,2) = [-0.25, 0.25];
searchPattern4(:,3) = [0.25, -0.25];
searchPattern4(:,4) = [0.25, 0.25];

% motion vector indexing for range +-4, row-wise
% the actual distance between adjacent indices is 0.25 pixel
indexTable = reshape(1:39^2, 39, 39)';

for row = 1:numBlockY
    for col = 1:numBlockX
        locX = positionBlockX(col);  % position of current img block
        locY = positionBlockY(row);
        imgBlock = image(locY:locY+7, locX:locX+7);  % current image block
        %% Step 1: integer-pixel, loop in search field +-4
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
        %% Step 2: half-pel search (based on the best integer result)
        halfBestX = bestX;
        halfBestY = bestY;
        for search = 1:4
            newRefX = bestX + searchPattern2(1, search);
            newRefY = bestY + searchPattern2(2, search);
            if (newRefX<1) | (newRefX>(width-8))  % exceed border
                continue
            end
            if (newRefY<1) | (newRefY>(height-8)) % exceed border
                continue
            end
            refInterpLocX = newRefX./0.5 - 1;
            refInterpLocY = newRefY./0.5 - 1;
            newRefBlock = refInterp2(refInterpLocY:2:refInterpLocY+14, refInterpLocX:2:refInterpLocX+14);
            dd = (imgBlock - round(newRefBlock)).^2;
            d = sum(dd(:));
            if d < SSD_min
                SSD_min = d;
                halfBestX = newRefX;
                halfBestY = newRefY;
            end
        end
        %% Step 3: quarter-pel search (based on the best half-pel result)
        finalBestX = halfBestX;
        finalBestY = halfBestY;
        for search = 1:4
            newRefX = halfBestX + searchPattern4(1, search);
            newRefY = halfBestY + searchPattern4(2, search);
            if (newRefX<1) | (newRefX>(width-8))  % exceed border
                continue
            end
            if (newRefY<1) | (newRefY>(height-8)) % exceed border
                continue
            end
            refInterpLocX = newRefX./0.25 - 3;
            refInterpLocY = newRefY./0.25 - 3;
            newRefBlock = refInterp4(refInterpLocY:4:refInterpLocY+28, ...
                refInterpLocX:4:refInterpLocX+28);
            dd = (imgBlock - round(newRefBlock)).^2;
            d = sum(dd(:));
            if d < SSD_min
                SSD_min = d;
                finalBestX = newRefX;
                finalBestY = newRefY;
            end
        end
        % relative vector, center is (9,9)
        vector = ([finalBestX, finalBestY] - [locX, locY])./0.25 + 20;
        motion_vectors_indices(row, col) = indexTable(vector(2), vector(1));
    end
end
end