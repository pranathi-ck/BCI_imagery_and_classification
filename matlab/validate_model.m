%% ============================================================
%  VALIDATION ACCURACY: Stratified K-Fold CV on the EXACT pipeline
%  (F-stat feature ranking -> top-72 -> z-score -> RBF SVM ECOC)
%
%  Why this script exists: the submitted final_csp_svm_model.mat is
%  trained on 100% of labeled data (no held-out set), so it has no
%  honest accuracy number of its own. This script re-runs the same
%  methodology under proper k-fold cross-validation - feature ranking
%  and standardization are refit INSIDE each fold on the training
%  split only, so no fold ever leaks into its own evaluation - to
%  produce the validation accuracy required for accuracy_report.csv.
%% ============================================================

clear; clc; close all;

%% ------------------------------------------------------------
% CONFIG
% ---------------------------------------------------------------
trainFile   = 'step13_csp_train_features.csv';
outCsv      = 'accuracy_report.csv';
outFig      = 'accuracy_report.png';
K           = 5;      % number of folds
bestN       = 72;      % must match the final model's feature count
rngSeed     = 42;

rng(rngSeed);

%% ------------------------------------------------------------
% LOAD DATA
% ---------------------------------------------------------------
fprintf('Loading training data...\n');
trainTbl = readtable(trainFile);
fprintf('Training size: %d rows x %d columns\n', height(trainTbl), width(trainTbl));

Y = categorical(trainTbl.label);
classes = categories(Y);
numClasses = numel(classes);

varNames = trainTbl.Properties.VariableNames;
excludeNames = {'label','trial_id','test_id','subject','session'};
featureMask = true(1,width(trainTbl));
for k = 1:numel(excludeNames)
    idx = strcmpi(varNames,excludeNames{k});
    featureMask(idx) = false;
end
featureNames = varNames(featureMask);
X = trainTbl{:,featureMask};

fprintf('Features available: %d | Using top %d per fold | %d-fold stratified CV\n', ...
    size(X,2), bestN, K);

%% ------------------------------------------------------------
% STRATIFIED K-FOLD SPLIT
% ---------------------------------------------------------------
cv = cvpartition(Y, 'KFold', K, 'Stratify', true);

foldAcc = zeros(K,1);
% aggregate confusion matrix across all folds (rows=true, cols=pred)
aggConfMat = zeros(numClasses, numClasses);

fprintf('\nRunning %d-fold cross-validation (this refits feature ranking,\n', K);
fprintf('standardization, and the SVM inside every fold to avoid leakage)...\n\n');

for fold = 1:K
    tFoldStart = tic;

    trainIdx = training(cv, fold);
    testIdx  = test(cv, fold);

    Xtr = X(trainIdx,:);  Ytr = Y(trainIdx);
    Xte = X(testIdx,:);   Yte = Y(testIdx);

    % --- Feature ranking (F-statistic), computed on TRAIN split only ---
    numFeatures = size(Xtr,2);
    F = zeros(numFeatures,1);
    for j = 1:numFeatures
        x = Xtr(:,j);
        grandMean = mean(x);
        between = 0; within = 0;
        for c = 1:numClasses
            idx = Ytr == classes{c};
            xc = x(idx);
            nc = sum(idx);
            if nc > 0
                mc = mean(xc);
                between = between + nc*(mc-grandMean)^2;
                within = within + sum((xc-mc).^2);
            end
        end
        dfBetween = numClasses-1;
        dfWithin = numel(x)-numClasses;
        if within > 0
            F(j) = (between/dfBetween) / (within/dfWithin);
        else
            F(j) = 0;
        end
    end
    [~, rankedIdx] = sort(F,'descend');
    foldFeatureIdx = rankedIdx(1:bestN);

    % --- Standardize using TRAIN split statistics only ---
    XtrSel = Xtr(:,foldFeatureIdx);
    XteSel = Xte(:,foldFeatureIdx);
    mu    = mean(XtrSel,1);
    sigma = std(XtrSel,[],1);
    sigma(sigma < 1e-12) = 1;
    XtrS = (XtrSel - mu) ./ sigma;
    XteS = (XteSel - mu) ./ sigma;

    % --- Train the same model architecture as the final model ---
    model = fitcecoc(XtrS, Ytr, ...
        'Learners', templateSVM('KernelFunction','rbf', 'KernelScale','auto', ...
            'BoxConstraint',1, 'Standardize',false), ...
        'Coding','onevsone');

    % --- Evaluate ---
    Ypred = predict(model, XteS);
    foldAcc(fold) = mean(Ypred == Yte);

    C = confusionmat(Yte, Ypred, 'Order', classes);
    aggConfMat = aggConfMat + C;

    fprintf('Fold %d/%d: accuracy = %.4f  (%.1f s)\n', fold, K, foldAcc(fold), toc(tFoldStart));
end

%% ------------------------------------------------------------
% AGGREGATE METRICS
% ---------------------------------------------------------------
accMean = mean(foldAcc);
accStd  = std(foldAcc);
accMin  = min(foldAcc);
accMax  = max(foldAcc);

fprintf('\n============================================================\n');
fprintf('CROSS-VALIDATED ACCURACY: %.4f +/- %.4f  (min %.4f, max %.4f)\n', ...
    accMean, accStd, accMin, accMax);
fprintf('============================================================\n');

% Per-class precision/recall/F1 from the aggregated confusion matrix
precision = zeros(numClasses,1);
recall    = zeros(numClasses,1);
f1        = zeros(numClasses,1);
support   = sum(aggConfMat,2);

for c = 1:numClasses
    tp = aggConfMat(c,c);
    fp = sum(aggConfMat(:,c)) - tp;
    fn = sum(aggConfMat(c,:)) - tp;
    if (tp+fp) > 0, precision(c) = tp/(tp+fp); else, precision(c) = 0; end
    if (tp+fn) > 0, recall(c)    = tp/(tp+fn); else, recall(c) = 0; end
    if (precision(c)+recall(c)) > 0
        f1(c) = 2*precision(c)*recall(c)/(precision(c)+recall(c));
    else
        f1(c) = 0;
    end
end

macroP = mean(precision); macroR = mean(recall); macroF1 = mean(f1);
w = support/sum(support);
weightedP = sum(precision.*w); weightedR = sum(recall.*w); weightedF1 = sum(f1.*w);

fprintf('\nPer-class metrics:\n');
for c = 1:numClasses
    fprintf('%-12s precision=%.4f  recall=%.4f  f1=%.4f  support=%d\n', ...
        classes{c}, precision(c), recall(c), f1(c), support(c));
end

%% ------------------------------------------------------------
% WRITE accuracy_report.csv
% ---------------------------------------------------------------
metricNames = {};
metricValues = [];

metricNames(end+1) = {'cv_folds'};            metricValues(end+1) = K;
metricNames(end+1) = {'num_trials'};          metricValues(end+1) = numel(Y);
metricNames(end+1) = {'num_features_used'};   metricValues(end+1) = bestN;
metricNames(end+1) = {'accuracy_mean'};       metricValues(end+1) = accMean;
metricNames(end+1) = {'accuracy_std'};        metricValues(end+1) = accStd;
metricNames(end+1) = {'accuracy_min'};        metricValues(end+1) = accMin;
metricNames(end+1) = {'accuracy_max'};        metricValues(end+1) = accMax;
metricNames(end+1) = {'macro_precision'};     metricValues(end+1) = macroP;
metricNames(end+1) = {'macro_recall'};        metricValues(end+1) = macroR;
metricNames(end+1) = {'macro_f1'};            metricValues(end+1) = macroF1;
metricNames(end+1) = {'weighted_precision'};  metricValues(end+1) = weightedP;
metricNames(end+1) = {'weighted_recall'};     metricValues(end+1) = weightedR;
metricNames(end+1) = {'weighted_f1'};         metricValues(end+1) = weightedF1;

for c = 1:numClasses
    metricNames(end+1) = {[char(classes{c}) '_precision']}; metricValues(end+1) = precision(c);
    metricNames(end+1) = {[char(classes{c}) '_recall']};    metricValues(end+1) = recall(c);
    metricNames(end+1) = {[char(classes{c}) '_f1']};        metricValues(end+1) = f1(c);
    metricNames(end+1) = {[char(classes{c}) '_support']};   metricValues(end+1) = support(c);
end
for fold = 1:K
    metricNames(end+1) = {sprintf('fold%d_accuracy',fold)}; metricValues(end+1) = foldAcc(fold);
end

reportTbl = table(metricNames', metricValues', 'VariableNames', {'metric','value'});
writetable(reportTbl, outCsv);
fprintf('\nWrote %s (%d rows)\n', outCsv, height(reportTbl));

%% ------------------------------------------------------------
% VISUALIZATION: confusion matrix heatmap + per-fold accuracy
% ---------------------------------------------------------------
figAcc = figure('Color','w','Position',[100 100 1100 480], 'Name','Validation Accuracy Report');

subplot(1,2,1);
confPct = 100*aggConfMat ./ sum(aggConfMat,2);
imagesc(confPct); colormap(gca, 'parula'); colorbar;
set(gca,'XTick',1:numClasses,'XTickLabel',classes, ...
        'YTick',1:numClasses,'YTickLabel',classes,'FontSize',9);
xlabel('Predicted'); ylabel('True');
title(sprintf('Confusion Matrix (%% of true class)\nAggregated across %d folds', K), ...
    'FontWeight','bold','FontSize',10);
for i = 1:numClasses
    for j = 1:numClasses
        text(j,i, sprintf('%.1f%%',confPct(i,j)), 'HorizontalAlignment','center', ...
            'Color', 'w', 'FontSize', 9, 'FontWeight','bold');
    end
end

subplot(1,2,2);
bar(foldAcc, 'FaceColor',[0.2 0.5 0.8]);
hold on;
yline(accMean, 'r--', sprintf('mean = %.4f', accMean), 'LineWidth',1.5, 'FontSize',9);
hold off;
ylim([max(0,min(foldAcc)-0.05), 1]);
set(gca,'XTick',1:K);
xlabel('Fold'); ylabel('Accuracy');
title(sprintf('Per-Fold Accuracy (%.4f +/- %.4f)', accMean, accStd), ...
    'FontWeight','bold','FontSize',10);
grid on; box off;

sgtitle('Cross-Validated Performance of the Final Model Architecture', ...
    'FontWeight','bold','FontSize',13);

exportgraphics(figAcc, outFig, 'Resolution', 200);
fprintf('Wrote %s\n', outFig);

fprintf('\nDone. Use accuracy_mean from %s as the validation accuracy for submission.\n', outCsv);