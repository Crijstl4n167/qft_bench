function res = invQFT5(qs)
% Stacked QFT implementation V2
% all diagonal gates are stored as vectors of the diagonal to improve
% efficency
% inverse thus reverse order and conjugate (all gates are symmetric thus no
% transpose)
    arguments (Input)
        qs (:,:) double {coder.mustBeComplex}
    end

    n = round(log2(size(qs, 1)));
    res = complex(qs);
    M1 = [1; 0];
    M2 = [0; 1];
    Rs = cellfun(@(x) conj(R5(x)), num2cell(1:n), UniformOutput=false);

    for i = n-1:-1:0
        res = applyH(res, i, n);
        if i > 0
            U1 = repmat({[1; 1]}, 1, n);
            U2 = U1;
            U1{i+1} = M1;
            U2{i+1} = M2;
            U2(1:i) = Rs(i+1:-1:2);
            res = (multikron5(U1) +  multikron5(U2)) .* res;
        end
    end
end