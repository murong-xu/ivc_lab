function rec_image = quarterPel_rec(ref_image, motion_vectors)
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

[x1,y1] = meshgrid(1:width, 1:height);
[x2,y2] = meshgrid(1:0.5:width, 1:0.5:height);
[x4,y4] = meshgrid(1:0.25:width, 1:0.25:height);
refInterpY = interp2(x1,y1,ref_image(:,:,1),x2,y2,'linear');
refInterpCb = interp2(x1,y1,ref_image(:,:,2),x2,y2,'linear');
refInterpCr = interp2(x1,y1,ref_image(:,:,3),x2,y2,'linear');
refInterpY = interp2(x2,y2,refInterpY,x4,y4,'linear');
refInterpCb = interp2(x2,y2,refInterpCb,x4,y4,'linear');
refInterpCr = interp2(x2,y2,refInterpCr,x4,y4,'linear');
refInterp = cat(3, refInterpY, refInterpCb, refInterpCr);
% motion vector indexing for range +-4, row-wise
% the actual distance between adjacent indices is 0.25 pixel
indexTable = reshape(1:39^2, 39, 39)';

for row = 1:numBlockY
    for col = 1:numBlockX
        index = motion_vectors(row, col);
        [vectorY, vectorX] = find(indexTable==index);  % not centered
        vectorX = (vectorX - 20).*0.25;  % centered to (20,20), relative vector
        vectorY = (vectorY - 20).*0.25;
        locX = positionBlockX(col);  % position of current img block
        locY = positionBlockY(row);
        refX = (locX + vectorX)./0.25 - 3;  % position of reference block
        refY = (locY + vectorY)./0.25 - 3;
        refBlock = refInterp(refY:4:refY+28, refX:4:refX+28, :);  % current ref block
        rec_image(locY:locY+7, locX:locX+7, :) = refBlock;
    end
end
end