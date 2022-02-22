function zze = ZeroRunEnc_EoB(zz, EOB)
%  Input         : zz (Zig-zag scanned sequence, 1xN)
%                  EOB (End Of Block symbol, scalar)
%
%  Output        : zze (zero-run-level encoded sequence, 1xM)
numElement = size(zz, 2);
numSymbolInBlock = 64;
numBlock = numElement ./ numSymbolInBlock;
currentZzePos = 1;

% loop for all blocks
for k = 1 : numBlock
    numFollowingZeros = -1;
    for i = 1:numSymbolInBlock
        currentZzPos = (k-1)*numSymbolInBlock + i;
        % case 1: current input is non-zero
        if zz(currentZzPos) ~= 0
            % check if there are existing successive zeros before
            if numFollowingZeros ~= -1
                zze(currentZzePos) = numFollowingZeros;
                currentZzePos = currentZzePos + 1;
            end
            zze(currentZzePos) = zz(currentZzPos);  % non-zero
            currentZzePos = currentZzePos + 1;  % position increment
            numFollowingZeros = -1;
        else
            % case 2: current input is a zero
            % case 2.1: the current zero is a "first" zero
            if numFollowingZeros == -1
                zze(currentZzePos) = 0;
                currentZzePos = currentZzePos + 1;
                numFollowingZeros = 0;
            else
                % case 2.2: the current zero is a "following" zero
                numFollowingZeros = numFollowingZeros + 1;
            end
        end
    end
    % case 3: when end of block still find successive zeros
    if numFollowingZeros ~= -1
        zze(currentZzePos-1) = EOB;
    end
end
end
