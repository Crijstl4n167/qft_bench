function res = QFT1(qs)
% applies the dense version of the QFT to the input qs
    n = round(log2(length(qs)));
    res = qs;
    for i = 0:n-1
        res = H1(i,n) * res;
        for j = 2:n-i
            res = controlled1(i+j-1, i, R1(j), n) * res;
        end
    end
end