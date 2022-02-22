%% E5.1
% load('image_ycbcr');
% load('ref_image_ycbcr');
% mv_indices = SSD(reference_image(:,:,1), image(:,:,1));
% a = reshape(1:256, 16,16);
% b = reshape(1:256, 16,16);
% mv_indices = SSD(a, b);
% fprintf("The syntax of the code seems to be correct, next run the assessment to verify the correctness");

%% E5.1
function motion_vectors_indices = SSD(ref_image, image)
%  Input         : ref_image(Reference Image, size: height x width)
%                  image (Current Image, size: height x width)
%
%  Output        : motion_vectors_indices (Motion Vector Indices, size: (height/8) x (width/8) x 1 )

height = size(image, 1);
width = size(image, 2);
numBlockX = width ./ 8;
numBlockY = height ./ 8;
motion_vectors_indices = zeros(numBlockY, numBlockX);

positionBlockX = 1 : 8 : width;  % position of block
positionBlockY = 1 : 8 : height;
% motion vector indexing for range +-4, row-wise
indexTable = reshape(1:81, 9, 9)';

for row = 1:numBlockY
    for col = 1:numBlockX
        locX = positionBlockX(col);  % position of current img block
        locY = positionBlockY(row);
        imgBlock = image(locY:locY+7, locX:locX+7);  % current image block
        % loop in search field +-4
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
        % relative vector, center is (4,4)
        vector = [bestX, bestY] - [locX, locY] + 5;
        motion_vectors_indices(row, col) = indexTable(vector(2), vector(1));
    end
end
end