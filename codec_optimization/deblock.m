function deblocked = deblock(img, index)
% deblocked = deblock(img, index)
% Perform post-deblocking horizontally and vertically, using the choosen
% deblocking parameters alpha and beta.
%
%  Input         : img (image YCbCr)
%                  index (determine which parameters in the table)
%
%  Output        : deblocked (resulting image)
alphaTable = [0	 0	0	0	0	0	0	0	0	0	0	0	0	0	0 ...
    0	4	4	5	6	7	8	9	10	12	13  15	17	20	22	25	28	...
    32	36	40	45	50	56	63	71	80	90	101	113	127	144	162	182	203	226	255	255];
betaTable = [0	0	0	0	0	0	0	0	0	0	0	0	0	0	0 ...
    0	2	2	2	3	3	3	3	4	4	4   6	6	7	7	8	8 ...
    9	9	10	10	11	11	12	12	13	13	14	14	15	15	16	16	17	17	18	18];

[height, width, ~] = size(img);

alpha = alphaTable(index);
beta = betaTable(index);

deblocked = img;
blockSize = 8;
%% Deblocking along Vertical Direction
for channel = 1:3
    for row = blockSize : blockSize : height-blockSize
        for col = 1 : blockSize : width
            upperBlk = img(row-blockSize+1:row,col:col+blockSize-1,channel);
            downBlk = img(row+1:row+blockSize,col:col+blockSize-1,channel);
            for k = 1:blockSize
                p0 = upperBlk(blockSize,k);
                p1 = upperBlk(blockSize-1,k);
                p2 = upperBlk(blockSize-2,k);
                p3 = upperBlk(blockSize-3,k);
                q0 = downBlk(1,k);
                q1 = downBlk(2,k);
                q2 = downBlk(3,k);
                q3 = downBlk(4,k);
                diff1 = abs(p0 - q0);
                diff2 = abs(p2 - p0);
                diff3 = abs(q2 - q0);
                diff4 = abs(p1 - p0);
                diff5 = abs(q1 - q0);
                if (diff1<alpha) && (diff4<beta) && (diff5<beta) && ...
                        (diff2<beta) && (diff3<beta) % strong deblocking
                    p0_new = (p2+2*p1+2*p0+2*q0+q1+4)./8;
                    p1_new = (p2+p1+p0+q0+2)./4;
                    p2_new = (2*p3+3*p2+p1+p0+q0+4)./8;
                    deblocked(row-2:row,col+k-1,channel) = [p2_new, p1_new, p0_new]';
                    q0_new = (q2+2*q1+2*q0+2*p0+p1+4)./8;
                    q1_new = (q2+q1+q0+p0+2)./4;
                    q2_new = (2*q3+3*q2+q1+q0+p0+4)./8;
                    deblocked(row+1:row+3,col+k-1,channel) = [q0_new, q1_new, q2_new]';
                elseif (diff1<alpha) && (diff4<beta) && (diff5<beta)% weak deblocking
                    p0_new = (2*p1+p0+q1+2)./4;
                    q0_new = (2*q1+q0+p1+2)./4;
                    deblocked(row:row+1,col+k-1,channel) = [p0_new, q0_new]';
                end
            end
        end
    end
end
%% Deblocking along Horizontal Direction
for channel = 1:3
    for col = blockSize:blockSize:width-blockSize
        for row = 1:blockSize:height
            leftBlk = img(row:row+blockSize-1,col-blockSize+1:col,channel);
            rightBlk = img(row:row+blockSize-1,col+1:col+blockSize,channel);
            for k = 1:blockSize
                p0 = leftBlk(k,blockSize);
                p1 = leftBlk(k,blockSize-1);
                p2 = leftBlk(k,blockSize-2);
                p3 = leftBlk(k,blockSize-3);
                q0 = rightBlk(k,1);
                q1 = rightBlk(k,2);
                q2 = rightBlk(k,3);
                q3 = rightBlk(k,4);
                diff1 = abs(p0 - q0);
                diff2 = abs(p2 - p0);
                diff3 = abs(q2 - q0);
                diff4 = abs(p1 - p0);
                diff5 = abs(q1 - q0);
                if (diff1<alpha) && (diff4<beta) && (diff5<beta) && ...
                        (diff2<beta) && (diff3<beta) % strong deblocking
                    p0_new = (p2+2*p1+2*p0+2*q0+q1+4)./8;
                    p1_new = (p2+p1+p0+q0+2)./4;
                    p2_new = (2*p3+3*p2+p1+p0+q0+4)./8;
                    deblocked(row+k-1,col-2:col,channel) = [p2_new, p1_new, p0_new];
                    q0_new = (q2+2*q1+2*q0+2*p0+p1+4)./8;
                    q1_new = (q2+q1+q0+p0+2)./4;
                    q2_new = (2*q3+3*q2+q1+q0+p0+4)./8;
                    deblocked(row+k-1,col+1:col+3,channel) = [q0_new, q1_new, q2_new];
                elseif (diff1<alpha) && (diff4<beta) && (diff5<beta)% weak deblocking
                    p0_new = (2*p1+p0+q1+2)./4;
                    q0_new = (2*q1+q0+p1+2)./4;
                    deblocked(row+k-1,col:col+1,channel) = [p0_new, q0_new];
                end
            end
        end
    end
end
end