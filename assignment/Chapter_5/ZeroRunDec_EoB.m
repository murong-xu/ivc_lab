function dst = ZeroRunDec_EoB(src, EoB)
%  Function Name : ZeroRunDec1.m zero run level decoder
%  Input         : src (zero run encoded sequence 1xM with EoB signs)
%                  EoB (end of block sign)
%
%  Output        : dst (reconstructed zig-zag scanned sequence 1xN)

numElement = size(src, 2);  % number of zero-run encoded symbols
numSymbolInBlock = 64;

currentSrcPos = 1;
currentDstPos = 1;
indexRest = 1;
for i = 1 : numElement
    % case 1: current input is non-zero
    if src(currentSrcPos) ~= 0
        % case 1.1: current input is not EOB
        if src(currentSrcPos) ~= EoB
            dst(currentDstPos) = src(currentSrcPos);
            currentDstPos = currentDstPos + 1;  % increment
            currentSrcPos = currentSrcPos + 1;
            indexRest = indexRest + 1;
            if indexRest >= numSymbolInBlock+1  % block reset
                indexRest = 1;
            end
        else
            % case 1.2: current input is EOB
            indexRestZero = numSymbolInBlock - indexRest;
            dst(currentDstPos : currentDstPos+indexRestZero) = 0;
            indexRest = 1;  % reset
            currentDstPos = currentDstPos + indexRestZero + 1;
            currentSrcPos = currentSrcPos + 1;
        end
    else
        % case 2: current input is a zero
        indexRestZero = src(currentSrcPos + 1);
        dst(currentDstPos : currentDstPos+indexRestZero) = 0;
        indexRest = indexRest + indexRestZero + 1;
        currentDstPos = currentDstPos + indexRestZero + 1;
        currentSrcPos = currentSrcPos + 2;
    end
    if currentSrcPos > numElement
        break
    end
end
end
