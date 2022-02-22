%% For Huffman: special case (1D) for computing PMF
function pmf = stats_marg_1D(sequence, range)
% Input:   sequence (1D sequence)
%          range (range and step value of histogram calculaton, e.g. 0:255)
% Output:  pmf (probability mass function)
num = length(sequence); % denominator: height*width*channel
pmf = hist(sequence, range); % count PMF in individual channel
pmf = pmf./num; % sum up the 3 values for one pixel, normalization
% pmf(pmf==0) = []; % easier for calculating entropy (avoid NaN)
end