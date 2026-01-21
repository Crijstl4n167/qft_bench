function res = QFT6_GPU(qs)
% Stacked QFT implementation V3 (in place)
    arguments (Input)
        qs (:,1) double {coder.mustBeComplex}
    end
    
    blk_len = length(qs);
    n = round(log2(blk_len));
    h = 1/sqrt(2);
    
    res = gpuArray(complex(qs));
    Rs = cellfun(@(x) gpuArray([1; exp(2*pi*1i/2^x)]), num2cell(1:n), UniformOutput=false);
    Rs{1} = gpuArray(1);
    rots = Rs{1};

    for i = 0:n-1
        blk_len = blk_len/2; % ... = 2^(n-i-1)

        % qshape to isolate the target bit as the 2nd dimension
        % [Inner stride, Target Bit (2), Outer stride]
        % qs(:, 1, :) is where i-th bit is 0
        % qs(:, 2, :) is where i-th bit is 1
        res = reshape(res, blk_len, 2, []);
        v1 = res(:, 1, :);

        % form rotation gates
        rots = kron(Rs{i+1}, rots);
 
        % apply rotations to the more signigicant bits of the qs where i-th
        % bit is 1
        v2 = res(:,2,:) .* reshape(rots, 1, 1, []);

        % Apply Hadamard: |0> -> h*(v1+v2), |1> -> h*(v1-v2)
        res(:, 1, :) = h .* (v1 + v2);
        res(:, 2, :) = h .* (v1 - v2);
        res = res(:);
    end
end