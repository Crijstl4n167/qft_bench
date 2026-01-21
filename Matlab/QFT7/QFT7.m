function res = QFT7(qs)
% QFT using Matlabs fft is equivalent to applying the QFT gate of the
% lecture and swapping back the bits in the correct position

    % Remark: because of our convention of using exp(2*pi*i/N) as the N-th 
    % root of unity instead of exp(-2*pi*i/N) one can perform a scaled ifft
    % as the equivalent of our QFT
    res = sqrt(length(qs)) * ifft(qs);
end