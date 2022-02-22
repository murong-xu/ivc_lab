function pmf = stats_marg_2D(image, range)
% Input:   image (original image)
%          range (range and step value of histogram calculaton, e.g. 0:255)
% Output:  pmf (probability mass function)
sz = size(image);
num = length(image(:)); % denominator: height*width*channel
image_new = reshape(image, sz(1)*sz(2), 1);
pmf = hist(image_new, range); % count PMF in individual channel
pmf = pmf./num; % sum up the 3 values for one pixel, normalization
% pmf(pmf==0) = []; % easier for calculating entropy (avoid NaN)
end