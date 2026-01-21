%% multikron.m
function res = multikron1(U)
% input cell array of gates
    res = 1;
    for i = 1:length(U)
        res = kron(res, U{i});
    end
end