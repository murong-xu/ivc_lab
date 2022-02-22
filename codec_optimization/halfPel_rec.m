function rec_image = halfPel_rec(ref_image, motion_vectors)
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

% bi-linear interpolation for the whole image
[x1,y1] = meshgrid(1:width, 1:height);
[x2,y2] = meshgrid(1:0.5:width, 1:0.5:height);
refInterpY = interp2(x1,y1,ref_image(:,:,1),x2,y2,'linear');
refInterpCb = interp2(x1,y1,ref_image(:,:,2),x2,y2,'linear');
refInterpCr = interp2(x1,y1,ref_image(:,:,3),x2,y2,'linear');
refInterp = cat(3, refInterpY, refInterpCb, refInterpCr);
% motion vector indexing for range +-4, row-wise
% the actual distance between adjacent indices is 0.5 pixel
indexTable = reshape(1:19^2, 19, 19)';

for row = 1:numBlockY
    for col = 1:numBlockX
        index = motion_vectors(row, col);
        [vectorY, vectorX] = find(indexTable==index);  % not centered
        vectorX = (vectorX - 10).*0.5;  % centered to (10,10), relative vector
        vectorY = (vectorY - 10).*0.5;
        locX = positionBlockX(col);  % position of current img block
        locY = positionBlockY(row);
        refX = (locX + vectorX)./0.5 - 1;  % position of reference block
        refY = (locY + vectorY)./0.5 - 1;
        refBlock = refInterp(refY:2:refY+14, refX:2:refX+14, :);  % current ref block
        rec_image(locY:locY+7, locX:locX+7, :) = refBlock;
    end
end
end