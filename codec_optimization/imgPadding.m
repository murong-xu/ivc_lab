function imgPadded = imgPadding(img, mode)
% Input         : img          (Original Image)
%                 mode         (intra prediction for y/cb/cr)
%
% Output        : imgPadded    (padded image)

[height, width, ~] = size(img);
switch mode
    case 'luma'
        imgPadded = zeros(height+1,width+5);
        imgY = img(:,:,1);
        imgPadded(2:height+1,2:width+1) = imgY;
        imgPadded(1,2:width+1) = imgY(1,:);
        imgPadded(2:height+1,1) = imgY(:,1);
        imgPadded(1,1) = imgY(1,1);
        for counter = 2:5
            imgPadded(2:height+1,width+counter) = imgY(:,width);
        end
        for counter = 2:5
            imgPadded(1,width+counter) = imgPadded(2,width+counter);
        end
    case 'cb'
        imgPadded = zeros(height+1,width+1);
        imgCb = img(:,:,2);
        imgPadded(2:height+1,2:width+1) = imgCb;
        imgPadded(1,2:width+1) = imgCb(1,:);
        imgPadded(2:height+1,1) = imgCb(:,1);
        imgPadded(1,1) = imgCb(1,1);
    case 'cr'
        imgPadded = zeros(height+1,width+1);
        imgCr = img(:,:,3);
        imgPadded(2:height+1,2:width+1) = imgCr;
        imgPadded(1,2:width+1) = imgCr(1,:);
        imgPadded(2:height+1,1) = imgCr(:,1);
        imgPadded(1,1) = imgCr(1,1);
end
end