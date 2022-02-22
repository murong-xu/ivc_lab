%% E5.1
% rec_image = SSD_rec(b, mv_indices);
% fprintf("The syntax of the code seems to be correct, next run the assessment to verify the correctness");

%% E5.1
function rec_image = SSD_rec(ref_image, motion_vectors)
%  Input         : ref_image(Reference Image, YCbCr image)
%                  motion_vectors
%
%  Output        : rec_image (Reconstructed current image, YCbCr image)
height = size(ref_image, 1);
width = size(ref_image, 2);
numBlockX = width ./ 8;
numBlockY = height ./ 8;
rec_image = zeros(height, width, 3);

positionBlockX = 1 : 8 : width;  % position of block
positionBlockY = 1 : 8 : height;
% motion vector indexing for range +-4, row-wise
indexTable = reshape(1:81, 9, 9)';

for row = 1:numBlockY
    for col = 1:numBlockX
        index = motion_vectors(row, col);
        [vectorY, vectorX] = find(indexTable==index);  % not centered
        vectorX = vectorX - 5;  % centered to (5,5), relative vector
        vectorY = vectorY - 5;
        locX = positionBlockX(col);  % position of current img block
        locY = positionBlockY(row);
        refX = locX + vectorX;  % position of reference block
        refY = locY + vectorY;
        refBlock = ref_image(refY:refY+7, refX:refX+7, :);  % current ref block
        rec_image(locY:locY+7, locX:locX+7, :) = refBlock;
    end
end
end