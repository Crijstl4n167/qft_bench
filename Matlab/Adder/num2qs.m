function qs = num2qs(k, n)
    arguments (Input)
        k (1,1) {mustBeInteger} = 0
        n (1,1) {mustBeInteger} = max(ceil(log2(k+1)), 1)
    end

    if k >= 2^n
        error('%d not representable in %d bits', k, n);
    else
        qs = complex([zeros(k,1); 1; zeros(2^n-k-1, 1)]);
    end
end