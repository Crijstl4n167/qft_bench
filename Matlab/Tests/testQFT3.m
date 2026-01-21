addpath(genpath('../QFT3'))
filename = 'qft3_results.csv';

% Optional: Write a header first (if the file doesn't exist yet)
if ~isfile(filename)
    writematrix('QFT3', filename); 
end

for n = 1:24
    % 1. Setup Data
    qs = rand(2^n, 1) + rand(2^n, 1) * 1i;
    qs = qs/norm(qs);
    
    % 2. Run and Time
    tic
    QFT3(qs);
    time = toc;

    % 3. Memory Cleanup (Crucial for preventing crashes!)
    % Clear large variables before the next iteration to free up RAM
    clear qs; clear ans;
    
    % 4. Display progress to Command Window
    fprintf('QFT3: n = %d, Time = %.5fs\n', n, time);
    
    % 5. WRITE TO CSV IMMEDIATELY
    % Create a row vector: [n, time]
    dataRow = [n, time]; 
    
    % Append this row to the file
    writematrix(dataRow, filename, 'WriteMode', 'append');
end