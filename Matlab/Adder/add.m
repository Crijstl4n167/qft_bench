function res = add(qs, k, QFT, invQFT, bitorder)
% Simulate m + k on a quantum computer
    arguments (Input)
        qs (:,1) double = num2qs(1, 2)
        k (1,1) {mustBeInteger} = 1
        QFT function_handle = @(x) sqrt(length(x)) * ifft(x)
        invQFT function_handle = @(x) 1/sqrt(length(x)) * fft(x)
        bitorder {mustBeMember(bitorder, [-1, 1])} = 1
    end

    n = round(log2(length(qs)));

    if bitorder == 1
        loop = 1:n;
    elseif bitorder == -1
        loop = n:-1:1;
    end

    rots = 1;
    for j = loop
        rots = kron(rots, [1; exp(2*pi*k*1i/2^j)]);
    end

    res = invQFT(rots .* QFT(qs));
end