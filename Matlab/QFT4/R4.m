function qg = R4(k)
% returns the 1 q-bit R gate defined as [1, 0; 0, exp(2*pi*i/2^k)]
    qg = sparse([1, 0; 0, exp(2*pi*1i/2^k)]);
end