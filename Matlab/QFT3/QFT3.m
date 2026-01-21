function res = QFT3(qs) %#codegen
% QFT using the FFT algorithm of Decimation in Frequency
    arguments (Input)
        qs (:,1) double {coder.mustBeComplex}
    end
    N = length(qs);
    res = 1/sqrt(N) * FFT_DIF(qs);
end