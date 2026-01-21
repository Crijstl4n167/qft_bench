function qg = R5(k)
% returns the 1 q-bit R gate defined as [1, 0; 0, exp(2*pi*i/2^k)]
% only diagonal returned as vector for efficiency
    qg = [1; exp(2*pi*1i/2^k)];
end