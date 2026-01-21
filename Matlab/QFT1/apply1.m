function res = apply1(gates, qs)
% gates is a cell array of gates to be applied to qs
% qs is original the quantum state
% res is the resulting qs (res = gates{n} * ... * gates{1} * qs)
    res = qs;
    for i = 1:length(gates)
        res = gates{i} * res;
    end
end