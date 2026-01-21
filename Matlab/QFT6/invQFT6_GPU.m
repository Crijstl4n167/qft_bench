function res = invQFT6_GPU(qs)
% Stacked QFT implementation V3
% as inverse is just the conjugate transpose and all the gates are
% symmetric we just need to run through them in reverse and conjugate
    arguments (Input)
        qs (:,1) double {coder.mustBeComplex}
    end

    n = round(log2(length(qs)));
    h = 1/sqrt(2);

    res = gpuArray(complex(qs));
    Rs = cellfun(@(x) gpuArray([1; exp(-2*pi*1i/2^x)]), num2cell(1:n), UniformOutput=false);
    Rs{1} = gpuArray(1);
    rots = Rs{1};

    for i = n-1:-1:0
        blk_len = 2^(n-i-1);

        % Reshape to isolate the target bit as the 2nd dimension
        % [Inner stride, Target Bit (2), Outer stride]
        % res(:, 1, :) is where i-th bit is 0
        % res(:, 2, :) is where i-th bit is 1
        res = reshape(res, blk_len, 2, []);
        v1 = res(:, 1, :);
        v2 = res(:, 2, :);
        % prepare rotations
        rots = kron(Rs{i+1}, rots);

        % Apply H and Rotations
        res(:, 1, :) = h * (v1 + v2);
        res(:, 2, :) = h * (v1 - v2) .* reshape(rots, 1, 1, []);

        res = res(:);
    end
end