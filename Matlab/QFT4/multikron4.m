%% multikron.m
function res = multikron4(U)
% input cell array of gates
    res = 1;
    for i = 1:length(U)
        res = kron(res, U{i});
    end
end