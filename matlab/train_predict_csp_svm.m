%% ============================================================
%  FINAL MODEL: Top-72 CSP Features + RBF SVM
%  Trains on ALL labeled data, predicts on test set, writes predictions.csv
%  with CORRECT label names (left_hand/right_hand, not left/right)
%% ============================================================

clear; clc; close all;

%% ------------------------------------------------------------
% FILES
% ---------------------------------------------------------------
trainFile = 'step13_csp_train_features.csv';
testFile  = 'step13_csp_test_features.csv';

fprintf('Loading training data...\n');
trainTbl = readtable(trainFile);
fprintf('Loading test data...\n');
testTbl = readtable(testFile);

fprintf('Training size: %d rows x %d columns\n', height(trainTbl), width(trainTbl));
fprintf('Test size: %d rows x %d columns\n', height(testTbl), width(testTbl));

%% ------------------------------------------------------------
% LABELS AND FEATURES
% ---------------------------------------------------------------
Y = categorical(trainTbl.label);
fprintf('\nClass counts:\n');
disp(countcats(Y));
disp(categories(Y));

varNames = trainTbl.Properties.VariableNames;
excludeNames = {'label','trial_id','test_id','subject','session'};

featureMask = true(1,width(trainTbl));
for k = 1:numel(excludeNames)
    idx = strcmpi(varNames,excludeNames{k});
    featureMask(idx) = false;
end
featureNamesFiltered = varNames(featureMask);   % CORRECTLY filtered name list
X = trainTbl{:,featureMask};

testVarNames = testTbl.Properties.VariableNames;
testFeatureMask = true(1,width(testTbl));
for k = 1:numel(excludeNames)
    idx = strcmpi(testVarNames,excludeNames{k});
    testFeatureMask(idx) = false;
end
Xtest = testTbl{:,testFeatureMask};

fprintf('\nTraining features: %d, Test features: %d\n', size(X,2), size(Xtest,2));
if size(X,2) ~= size(Xtest,2)
    error('Train/test feature count mismatch - check column alignment.');
end

%% ------------------------------------------------------------
% RANK FEATURES (F-statistic, on ALL labeled data this time -
% for the FINAL model we don't need a held-out split for ranking)
% ---------------------------------------------------------------
fprintf('\nRanking features...\n');
classes = categories(Y);
numFeatures = size(X,2);
F = zeros(numFeatures,1);

for j = 1:numFeatures
    x = X(:,j);
    grandMean = mean(x);
    between = 0; within = 0;
    for c = 1:numel(classes)
        idx = Y == classes{c};
        xc = x(idx);
        nc = sum(idx);
        if nc > 0
            mc = mean(xc);
            between = between + nc*(mc-grandMean)^2;
            within = within + sum((xc-mc).^2);
        end
    end
    dfBetween = numel(classes)-1;
    dfWithin = numel(x)-numel(classes);
   if within > 0
    F(j) = (between/dfBetween) / (within/dfWithin);
else
    F(j) = 0;
end
end

[~, rankedIdx] = sort(F,'descend');

%% ------------------------------------------------------------
% USE TOP 72 FEATURES (your best validated result)
% ---------------------------------------------------------------
bestN = 72;
bestFeatureIdx = rankedIdx(1:bestN);

fprintf('\nUsing top %d features:\n', bestN);
for k = 1:bestN
    fprintf('%3d. %s  F=%.3f\n', k, featureNamesFiltered{bestFeatureIdx(k)}, F(bestFeatureIdx(k)));
end

Xfinal = X(:,bestFeatureIdx);
XtestFinal = Xtest(:,bestFeatureIdx);

%% ------------------------------------------------------------
% STANDARDIZE AND TRAIN FINAL MODEL ON ALL DATA
% ---------------------------------------------------------------
muFinal = mean(Xfinal,1);
sigmaFinal = std(Xfinal,[],1);
sigmaFinal(sigmaFinal < 1e-12) = 1;

XfinalS = (Xfinal-muFinal)./sigmaFinal;
XtestS  = (XtestFinal-muFinal)./sigmaFinal;

fprintf('\nTraining final RBF SVM on ALL %d trials...\n', size(XfinalS,1));

finalModel = fitcecoc(XfinalS, Y, ...
    'Learners', templateSVM('KernelFunction','rbf', 'KernelScale','auto', ...
        'BoxConstraint',1, 'Standardize',false), ...
    'Coding','onevsone');

save('final_csp_svm_model.mat', 'finalModel', 'muFinal', 'sigmaFinal', 'bestFeatureIdx', 'featureNamesFiltered');
fprintf('Model saved.\n');

%% ------------------------------------------------------------
% PREDICT TEST TRIALS
% ---------------------------------------------------------------
fprintf('\nPredicting test trials...\n');
testPred = predict(finalModel, XtestS);

fprintf('\nPredicted class distribution (raw labels):\n');
predCategories = categories(testPred);
for k = 1:numel(predCategories)
    n = sum(testPred == predCategories{k});
    fprintf('%-10s %4d (%.2f%%)\n', predCategories{k}, n, 100*n/numel(testPred));
end

%% ------------------------------------------------------------
% *** CRITICAL FIX: map to REQUIRED label names ***
% Challenge requires exactly: left_hand, right_hand, feet, idle
% ---------------------------------------------------------------
predictedLabels = string(testPred);
predictedLabels = strrep(predictedLabels, 'left', 'left_hand');
predictedLabels = strrep(predictedLabels, 'right', 'right_hand');
% Safety: verify no double-replacement happened (e.g. "left_hand_hand")
predictedLabels = strrep(predictedLabels, 'left_hand_hand', 'left_hand');
predictedLabels = strrep(predictedLabels, 'right_hand_hand', 'right_hand');

fprintf('\nFinal label set (MUST be exactly these 4):\n');
disp(unique(predictedLabels));

%% ------------------------------------------------------------
% CREATE SUBMISSION
% ---------------------------------------------------------------
testIDs = string(testTbl.test_id);

submission = table(testIDs, predictedLabels, 'VariableNames', {'test_id','predicted_label'});

fprintf('\nSubmission preview:\n');
disp(submission(1:min(10,height(submission)), :));

if height(submission) ~= 2000
    error('ERROR: submission must contain exactly 2000 rows. Found %d.', height(submission));
end
if width(submission) ~= 2
    error('ERROR: submission must contain exactly 2 columns.');
end

VALID_LABELS = ["left_hand","right_hand","feet","idle"];
if ~all(ismember(unique(predictedLabels), VALID_LABELS))
    error('INVALID LABELS DETECTED - check the mapping above before saving.');
end

writetable(submission, 'predictions.csv');

fprintf('\n============================================================\n');
fprintf('SUBMISSION CREATED SUCCESSFULLY - predictions.csv\n');
fprintf('============================================================\n');
fprintf('Rows: %d, Columns: %d\n', height(submission), width(submission));
fprintf('All labels valid: left_hand, right_hand, feet, idle\n');
