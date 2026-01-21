function qs = applyH(qs, k, n) %#codegen
    % qs: state vector (size 2^n)
    % k: target qubit index (0-indexed)
    % n: total qubits
    arguments (Input)
        qs (:,1) double {coder.mustBeComplex}
        k (1,1) double = 0
        n (1,1) double = log2(size(qs, 1))
    end
    
    h = 1/sqrt(2);
    blk_len = 2^(n-k-1);
    for i = 1 : 2*blk_len : 2^n
        for j = 0:blk_len-1
            v1 = qs(i+j);
            v2 = qs(i+j+blk_len);
            qs(i+j) = h * (v1 + v2);
            qs(i+j+blk_len) = h * (v1 - v2);
        end
    end
end