function res = invQFT1(qs)
% applies the dense version of the inverse QFT to the input qs
    n = round(log2(length(qs)));
    res = qs;
    for i = n-1:-1:0
        for j = 2:n-i
            res = controlled1(i+j-1, i, conj(R1(j)), n) * res;
        end
        res = H1(i,n) * res;
    end
end