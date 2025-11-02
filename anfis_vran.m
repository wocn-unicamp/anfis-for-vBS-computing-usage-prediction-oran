%% ================== Core Functions ==================

function [success, mse, r2, evs] = calculate_metrics(actual, predicted)
    % Classification-style success (threshold 0.5)
    success = mean(abs(predicted - actual) < 0.5, 'all');
    % Mean Squared Error
    mse = mean((predicted - actual).^2, 'all');
    % R-squared
    ssres = sum((actual - predicted).^2, 'all');
    sstot = sum((actual - mean(actual, 'all')).^2, 'all');
    r2 = 1 - ssres / sstot;
    % Explained Variance Score
    evs = 1 - var(actual - predicted, 0, 'all') / var(actual, 0, 'all');
end

function [testOutput, valOutput, results] = evaluate_algorithm(fis, X_val, X_test, y_test, row_name, results, n)
    valOutput = evalfis(fis, X_val);
    testOutput = evalfis(fis, X_test);
    [success, mse, r2, evs] = calculate_metrics(y_test, testOutput);
    new_row = {sprintf('%s_%d', row_name, n), success, mse, r2, evs, length(fis.rule)};
    results = [results; new_row];
end

function [results, gralOut] = evaluate_overall(y_test, table_head, results, gralOut)
    assert(isequal(size(gralOut), size(y_test)), 'gralOut and y_test must have the same dimensions');
    [success, mse, r2, evs] = calculate_metrics(y_test, gralOut);
    rmse = sqrt(mse);
    success_results = cell2table({'OVERALL', success, rmse, r2, evs, size(y_test, 2) * 2}, ...
                                 'VariableNames', table_head);
    results = [results; success_results];
    fprintf('Overall Evaluation:\nMSE: %.4f\nRMSE: %.4f\nR2: %.4f\nEVS: %.4f\nSuccess Rate: %.2f%%\n', ...
            mse, rmse, r2, evs, success*100);
end

function fis = fuzzycm(X_train, y_train, clusters)
    opts = genfisOptions('FCMClustering', 'NumClusters', clusters, 'Verbose', true);
    fis = genfis(X_train, y_train, opts);
end

function trainfis = train_neuron(fis, X_train, y_train, epochs)
    [in,out,~] = getTunableSettings(fis);
    opt = tunefisOptions("Method","anfis","OptimizationType","tuning");
    opt.MethodOptions.EpochNumber = epochs;
    opt.Display = "none";
    trainfis = tunefis(fis,[in;out],X_train,y_train,opt);
end

function [gralOut, learnOut, resul_data] = initGen(epochs, clusters, resul_data)
    labels = resul_data.labels;
    features = size(resul_data.df_train, 2);
    cols_to_remove = features - labels + (1:labels);

    X_train_all = resul_data.df_train;
    X_val_all   = resul_data.df_val;
    X_test_all  = resul_data.df_test;

    gralOut  = zeros(size(X_test_all, 1), labels);
    learnOut = zeros(size(X_val_all, 1), labels);

    for n = 1:labels
        fprintf('Generating model for label %02d...\n\n', n);

        X_train = X_train_all(:, ~ismember(1:features, cols_to_remove(n)));
        X_val   = X_val_all(:, ~ismember(1:features, cols_to_remove(n)));
        X_test  = X_test_all(:, ~ismember(1:features, cols_to_remove(n)));

        y_train = resul_data.df_train(:, cols_to_remove(n));
        y_val   = resul_data.df_val(:, cols_to_remove(n));
        y_test  = resul_data.df_test(:, cols_to_remove(n));

        try
            resul_data.fis = fuzzycm(X_train, y_train, clusters);
            resul_data.fis = train_neuron(resul_data.fis, X_train, y_train, epochs);
        catch ME
            fprintf('Error in model training for label %d: %s\n', n, ME.message);
            continue;
        end

        if resul_data.eval == 1
            [gralOut(:,n), learnOut(:,n), resul_data.results] = ...
                evaluate_algorithm(resul_data.fis, X_val, X_test, y_test, "FCM", resul_data.results, n);
        end
    end

    if resul_data.eval > 1
        [resul_data.results, gralOut] = ...
            evaluate_overall(y_val, resul_data.results.Properties.VariableNames, resul_data.results, gralOut);
    end
end

%% ================== Main Script ==================

clear; close all; clc;

% Constants
LABELS   = 6;
EPOCHS   = 5;
CLUSTERS = 3;
DATA_PATH = 'D:\Documents\MATLAB\Fuzzy_Projects\vran\vranf.csv';

% Load dataset
data = readmatrix(DATA_PATH);
assert(~isempty(data), 'Failed to read data or file is empty.');

% Split into train/val/test
inst = size(data, 1);
rng('default');
shuffled_indices = randperm(inst);
test_size = round(0.2 * inst);
train_val_size = inst - test_size;
train_size = round(0.75 * train_val_size);

df_train = data(shuffled_indices(1:train_size), :);
df_val   = data(shuffled_indices(train_size+1:train_val_size), :);
df_test  = data(shuffled_indices(end-test_size+1:end), :);

% Results table
table_head = {'Method','Successful Prediction','Mean Squared Error', ...
              'Coefficient of Determination','Explained Variance Score (EVS)','Total Rules'};
results = array2table(zeros(0, length(table_head)), 'VariableNames', table_head);

resul_data = struct('df_train', df_train, 'df_val', df_val, 'df_test', df_test, ...
                    'labels', LABELS, 'results', results, 'fis', [], 'eval', 1);

% Train and evaluate
[gralOut, learnOut, resul_data] = initGen(EPOCHS, CLUSTERS, resul_data);

% Display results
disp(resul_data.results);

% Save outputs
output_cols = size(data, 2)-LABELS+1:size(data, 2);
Ytest = df_test(:, output_cols);
Yval  = df_val(:, output_cols);

writematrix(gralOut, 'CSV/Y_test_pred.csv');
writematrix(learnOut, 'CSV/Y_val_pred.csv');
writematrix(Ytest, 'CSV/Y_test.csv');
writematrix(Yval, 'CSV/Y_val.csv');

% Plot membership functions (up to 25 inputs)
num_inputs = numel(resul_data.fis.Inputs);
for i = 1:min(25, num_inputs)
    subplot(5, 5, i);
    [xOut, yOut] = plotmf(resul_data.fis, 'input', i);
    for j = 1:size(xOut, 2)
        plot(xOut(:,j), yOut(:,j)); hold on;
    end
    title(['Input ' num2str(i)]);
    xlabel(''); ylabel('');
    hold off;
end
exportgraphics(gcf, 'membership_functions.eps');
