function res = QFT2(qs)
% QFT using sparse matrices
    n = round(log2(length(qs)));
    res = complex(qs);

    for i = 0:n-1
        res = H2(i,n) * res;
        for j = 2:n-i
            res = controlled2(i+j-1, i, R2(j), n) * res;
        end
    end
end