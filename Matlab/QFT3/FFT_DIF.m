function d = FFT_DIF(y) %#codegen
% fast calculation of d=qft(y) using DIF recursion
% input: y ... complex N x 1 vector 
% output: d ... complex N x 1 vector
    arguments (Input)
        y (:,1) double {coder.mustBeComplex}
    end
    N = length(y);
    if N==1, d=y; return; end % L=0 
    if N/2 ~= round(N/2) 
        error('expect vector of even length'); 
    end
    ak = y(1:N/2);
    bk = y(N/2+1:end);
    w = exp(1i*2*pi*(0:N/2-1)'/N); 
    d=[FFT_DIF(ak + bk); FFT_DIF((ak - bk) .* w)];
end