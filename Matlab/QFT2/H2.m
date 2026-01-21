%% H.m
function qg = H2(k, n)
% returns a sparse 2^n matrix representing the circuit of 
% applying an H gate on q-bits k which is a subset of {0, ..., n-1}
    arguments (Input)
        k (1,:) = 0
        n (1,1) = max(k)+1
    end
    
    h = 1/sqrt(2) * [1, 1; 1, -1];
    k = k + 1;
    qg = 1;
    for i = 1:n
        if ismember(i, k)
            qg = kron(qg, h);
        else
            qg = kron(qg, speye(2));
        end
    end    
end