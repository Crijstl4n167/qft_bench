function res = invQFT3(qs)
% QFT using the FFT algorithm of Decimation in Frequency
    arguments (Input)
        qs (:,1) double
    end
    N = length(qs);
    res = sqrt(N) * iFFT_DIF(qs);
end