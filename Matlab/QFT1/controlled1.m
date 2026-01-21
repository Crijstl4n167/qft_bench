function qg = controlled1(cntrl, tar, gate, n)
    M1 = [1, 0; 0, 0];
    M2 = [0, 0; 0, 1];
    % shift to base 1 indexing
    cntrl = cntrl + 1; tar = tar + 1;
    U1 = repmat({eye(2)}, 1, n);
    U2 = U1;
    U1{cntrl} = M1;
    U2{cntrl} = M2;
    U2{tar} = gate;
    qg = multikron1(U1) + multikron1(U2);
end