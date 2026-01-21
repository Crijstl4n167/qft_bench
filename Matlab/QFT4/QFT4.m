function res = QFT4(qs)
% stacked QFT implementation V1
    arguments (Input)
        qs (:,:) double
    end

    n = round(log2(length(qs)));
    res = complex(qs);
    M1 = sparse([1, 0; 0, 0]);
    M2 = sparse([0, 0; 0, 1]);
    Rs = cellfun(@(x) R4(x), num2cell(1:n), UniformOutput=false);
    for i = 0:n-1
        if i > 0
            U1 = repmat({speye(2)}, 1, n);
            U2 = U1;
            U1{i+1} = M1;
            U2{i+1} = M2;
            U2(1:i) = Rs(i+1:-1:2);
            res = (multikron4(U1) +  multikron4(U2)) * res;
        end
        res = H4(i, n) * res;
    end
end