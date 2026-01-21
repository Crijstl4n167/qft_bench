function res = invQFT7(qs)
% inverse QFT using Matlabs fft: equivalent to swapping back the bits into 
% reverse order and applying the inverse of the QFT gate of the lecture

    % Remark: because of our convention of using exp(2*pi*i/N) as the N-th 
    % root of unity instead of exp(-2*pi*i/N) one can perform a scaled fft 
    % as the inverse of ours
    res = 1/sqrt(length(qs)) * fft(qs);
end