function compare3(n) %#codegen
    arguments (Input)
        n (1,1) double = 15
    end

    qs = rand(2^n, 1) + rand(2^n, 1) * 1i;
    qs = qs/norm(qs);
    
    tic
    QFT3(qs);
    time = toc;
    disp(time);
end
