function res = invQFT2(qs)
% inverse QFT using sparse matrices
    n = round(log2(length(qs)));
    res = complex(qs);

    for i = n-1:-1:0
        for j = 2:n-i
            res = controlled2(i+j-1, i, conj(R2(j)), n) * res;
        end
        res = H2(i,n) * res;
    end
end