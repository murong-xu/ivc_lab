function motion_vectors_indices = halfPel(ref_image, image)
% motion_vectors_indices = halfPel(ref_image, image)
% First do the integer-search, then performing half-pel search based on the
% previous found best integer result
%
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
refInterp = interp2(x1,y1,ref_image,x2,y2,'linear');
% define search range for half-pel
searchPattern = zeros(2, 4);
searchPattern(:,1) = [-0.5, -0.5];
searchPattern(:,2) = [-0.5, 0.5];
searchPattern(:,3) = [0.5, -0.5];
searchPattern(:,4) = [0.5, 0.5];
% motion vector indexing for range +-4, row-wise
% the actual distance between adjacent indices is 0.5 pixel
indexTable = reshape(1:19^2, 19, 19)';

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
        finalBestX = bestX;
        finalBestY = bestY;
        for search = 1:4
            newRefX = bestX + searchPattern(1, search);
            newRefY = bestY + searchPattern(2, search);
            if (newRefX<1) | (newRefX>(width-8))  % exceed border
                continue
            end
            if (newRefY<1) | (newRefY>(height-8)) % exceed border
                continue
            end
            refInterpLocX = newRefX./0.5 - 1;
            refInterpLocY = newRefY./0.5 - 1;
            newRefBlock = refInterp(refInterpLocY:2:refInterpLocY+14,...
                refInterpLocX:2:refInterpLocX+14);
            dd = (imgBlock - round(newRefBlock)).^2;
            d = sum(dd(:));
            if d < SSD_min
                SSD_min = d;
                finalBestX = newRefX;
                finalBestY = newRefY;
            end
        end
        % relative vector, center is (9,9)
        vector = ([finalBestX, finalBestY] - [locX, locY])./0.5 + 10;
        motion_vectors_indices(row, col) = indexTable(vector(2), vector(1));
    end
end
end