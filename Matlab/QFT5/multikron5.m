%% multikron.m
function res = multikron5(U)
% input cell array of gates
    arguments (Input)
        U (1,:) cell
    end
    res = U{1};
    for i = 2:length(U)
        res = kron(res, U{i});
    end
end

% function res = multikron5(U) %#codegen
% % input cell array of gates
%     arguments (Input)
%         U (1,:) cell
%     end
% 
%     n = length(U);
%     res = complex(zeros(2^n, 1));
%     res(1) = 1;
%     for i = 1:n
%         res(1:2^i) = kron(res(1:2^(i-1)), U{i});
%     end
% end