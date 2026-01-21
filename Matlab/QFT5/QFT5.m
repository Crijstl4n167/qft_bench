function res = QFT5(qs)
% Stacked QFT implementation V2
% all diagonal gates are stored as vectors of the diagonal to improve
% efficency
    arguments (Input)
        qs (:,:) double {coder.mustBeComplex}
    end

    n = round(log2(size(qs, 1)));
    res = complex(qs);
    M1 = [1; 0];
    M2 = [0; 1];
    Rs = cellfun(@(x) R5(x), num2cell(1:n), UniformOutput=false);
    for i = 0:n-1
        if i > 0
            U1 = repmat({[1; 1]}, 1, n);
            U2 = U1;
            U1{i+1} = M1;
            U2{i+1} = M2;
            % U2(1:i) = Rs(i+1:-1:2);
            for j = 1:i
                U2{j} = Rs{i+2-j};
            end
            res = (multikron5(U1) +  multikron5(U2)) .* res;
        end
        res = applyH(res, i, n);
    end
end