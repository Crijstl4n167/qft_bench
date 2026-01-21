addpath(genpath('../QFT7'))
filename = 'qft7_gpu_results.csv';

% Optional: Write a header first (if the file doesn't exist yet)
if ~isfile(filename)
    writematrix('QFT7 on GPU', filename); 
end

for n = 1:28
    % 1. Setup Data
    qs = rand(2^n, 1) + rand(2^n, 1) * 1i;
    qs = qs/norm(qs);
    qs = gpuArray(qs);

    % 2. Run and Time
    tic
    QFT7(qs);
    time = toc;

    % 3. Memory Cleanup (Crucial for preventing crashes!)
    % Clear large variables before the next iteration to free up RAM
    clear qs; clear ans;
    
    % 4. Display progress to Command Window
    fprintf('QFT7 on GPU: n = %d, Time = %.5fs\n', n, time);
    
    % 5. WRITE TO CSV IMMEDIATELY
    % Create a row vector: [n, time]
    dataRow = [n, time]; 
    
    % Append this row to the file
    writematrix(dataRow, filename, 'WriteMode', 'append');
end