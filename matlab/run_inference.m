%% ============================================================
%  INFERENCE ONLY: Load trained CSP+SVM model and predict
%  Does NOT retrain anything. Loads final_csp_svm_model.mat,
%  applies the exact saved feature selection + standardization,
%  predicts, times the run, and writes predictions.csv
%% ============================================================

clear; clc; close all;

%% ------------------------------------------------------------
% CONFIG
% ---------------------------------------------------------------
modelFile = 'final_csp_svm_model.mat';
testFile  = 'step13_csp_test_features.csv';
outFile   = 'predictions.csv';

VALID_LABELS = ["left_hand","right_hand","feet","idle"];

%% ------------------------------------------------------------
% LOAD TRAINED MODEL (no training happens here)
% ---------------------------------------------------------------
fprintf('Loading trained model from %s ...\n', modelFile);
tLoadStart = tic;
S = load(modelFile, 'finalModel', 'muFinal', 'sigmaFinal', ...
                     'bestFeatureIdx', 'featureNamesFiltered');
loadTime = toc(tLoadStart);
fprintf('Model loaded in %.3f s\n', loadTime);

finalModel          = S.finalModel;
muFinal             = S.muFinal;
sigmaFinal          = S.sigmaFinal;
bestFeatureIdx      = S.bestFeatureIdx;
featureNamesFiltered = S.featureNamesFiltered;

%% ------------------------------------------------------------
% LOAD TEST/INFERENCE DATA
% ---------------------------------------------------------------
fprintf('Loading inference data from %s ...\n', testFile);
testTbl = readtable(testFile);
fprintf('Inference set: %d rows x %d columns\n', height(testTbl), width(testTbl));

excludeNames = {'label','trial_id','test_id','subject','session'};
testVarNames = testTbl.Properties.VariableNames;
testFeatureMask = true(1,width(testTbl));
for k = 1:numel(excludeNames)
    idx = strcmpi(testVarNames,excludeNames{k});
    testFeatureMask(idx) = false;
end
Xtest = testTbl{:,testFeatureMask};
testFeatureNames = testVarNames(testFeatureMask);

% ------------------------------------------------------------
% SAFETY CHECK: incoming feature columns must match what the
% model was trained on (name + order), otherwise silently
% mismatched columns could produce garbage predictions.
% ------------------------------------------------------------
if numel(testFeatureNames) ~= numel(featureNamesFiltered) || ...
   ~all(strcmpi(testFeatureNames, featureNamesFiltered))
    error(['Feature name/order mismatch between inference file and ', ...
           'training-time features. Check that %s matches the schema ', ...
           'used to train %s.'], testFile, modelFile);
end

if size(Xtest,2) < max(bestFeatureIdx)
    error('Inference file has fewer feature columns than the model expects.');
end

%% ------------------------------------------------------------
% APPLY SAVED FEATURE SELECTION + STANDARDIZATION
% (uses muFinal/sigmaFinal computed at TRAIN time - never
%  recomputed here, which is what makes this true inference)
% ---------------------------------------------------------------
XtestFinal = Xtest(:,bestFeatureIdx);
XtestS     = (XtestFinal - muFinal) ./ sigmaFinal;

%% ------------------------------------------------------------
% PREDICT + TIME IT (per-trial latency, useful for edge_metrics.csv)
% ---------------------------------------------------------------
fprintf('\nRunning inference on %d trials...\n', size(XtestS,1));
tPredStart = tic;
testPred = predict(finalModel, XtestS);
predTime = toc(tPredStart);

numTrials = size(XtestS,1);
fprintf('Inference complete: %.4f s total, %.4f ms/trial\n', ...
    predTime, 1000*predTime/numTrials);

fprintf('\nPredicted class distribution (raw labels):\n');
predCategories = categories(testPred);
for k = 1:numel(predCategories)
    n = sum(testPred == predCategories{k});
    fprintf('%-10s %4d (%.2f%%)\n', predCategories{k}, n, 100*n/numTrials);
end

%% ------------------------------------------------------------
% MAP TO REQUIRED LABEL NAMES (exact match, not strrep)
% Challenge requires exactly: left_hand, right_hand, feet, idle
% ---------------------------------------------------------------
rawLabels = string(testPred);
labelMap = containers.Map( ...
    {'left','right','feet','idle'}, ...
    {'left_hand','right_hand','feet','idle'});

predictedLabels = strings(numTrials,1);
for i = 1:numTrials
    key = char(rawLabels(i));
    if isKey(labelMap, key)
        predictedLabels(i) = labelMap(key);
    else
        error('Unrecognized raw label "%s" at row %d - update labelMap.', key, i);
    end
end

fprintf('\nFinal label set (MUST be exactly these 4):\n');
disp(unique(predictedLabels));

%% ------------------------------------------------------------
% WRITE SUBMISSION
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
if ~all(ismember(unique(predictedLabels), VALID_LABELS))
    error('INVALID LABELS DETECTED - check labelMap before saving.');
end

writetable(submission, outFile);

%% ------------------------------------------------------------
% VISUALIZATION
% Figures saved as PNG so they can be dropped straight into the
% technical report / inference video / SUMMARY.md
% ---------------------------------------------------------------
fprintf('\nGenerating figures...\n');

classNames  = ["left_hand","right_hand","feet","idle"];
classColors = [0.20 0.45 0.85;   % left_hand  - blue
               0.85 0.30 0.30;   % right_hand - red
               0.30 0.70 0.40;   % feet       - green
               0.90 0.60 0.10];  % idle       - amber

counts = zeros(1,numel(classNames));
for k = 1:numel(classNames)
    counts(k) = sum(predictedLabels == classNames(k));
end
pct = 100*counts/numTrials;

figMain = figure('Color','w','Position',[100 100 1200 800], ...
    'Name','BCI Inference Report');

% --- (1) Class distribution bar chart -------------------------
subplot(2,2,1);
b = bar(counts, 'FaceColor','flat');
b.CData = classColors;
set(gca,'XTickLabel',classNames,'FontSize',10,'TickLabelInterpreter','none');
ylabel('Trial count');
title('Predicted Class Distribution','FontWeight','bold');
for k = 1:numel(counts)
    text(k, counts(k)+15, sprintf('%d (%.1f%%)',counts(k),pct(k)), ...
        'HorizontalAlignment','center','FontSize',9);
end
box off; grid on; ax = gca; ax.GridAlpha = 0.15;

% --- (2) Class distribution pie chart --------------------------
subplot(2,2,2);
p = pie(counts, classNames);
colormap(gca, classColors);
title('Class Share','FontWeight','bold');

% --- (3) Latency gauge / bar -----------------------------------
subplot(2,2,3);
msPerTrial = 1000*predTime/numTrials;
barh([loadTime*1000, predTime*1000, msPerTrial], 'FaceColor',[0.4 0.4 0.4]);
set(gca,'YTickLabel',{'Model load (ms)','Total inference (ms)','Per-trial (ms)'}, ...
    'FontSize',9);
xlabel('Milliseconds');
title('Timing Breakdown','FontWeight','bold');
grid on; box off;

% --- (4) Confidence / margin view (if scores available) --------
subplot(2,2,4);
try
    [~, scores] = predict(finalModel, XtestS);
    maxScore = max(scores,[],2);
    histogram(maxScore, 30, 'FaceColor',[0.3 0.5 0.8], 'EdgeColor','w');
    xlabel('Winning class score'); ylabel('Number of trials');
    title('Prediction Confidence Distribution','FontWeight','bold');
    grid on; box off;
catch
    axis off;
    text(0.5,0.5,'Confidence scores not available for this model type', ...
        'HorizontalAlignment','center');
end

sgtitle(sprintf('BCI Motor Imagery - Inference Summary (%d trials)', numTrials), ...
    'FontWeight','bold','FontSize',14);

exportgraphics(figMain, 'inference_report.png', 'Resolution', 200);
fprintf('Saved figure: inference_report.png\n');

%% ------------------------------------------------------------
% SUMMARY (also handy content for edge_metrics.csv later)
% ---------------------------------------------------------------
fprintf('\n============================================================\n');
fprintf('INFERENCE COMPLETE - %s written\n', outFile);
fprintf('============================================================\n');
fprintf('Rows: %d, Columns: %d\n', height(submission), width(submission));
fprintf('Model load time: %.3f s\n', loadTime);
fprintf('Total inference time: %.4f s (%.4f ms/trial)\n', predTime, 1000*predTime/numTrials);
fprintf('All labels valid: left_hand, right_hand, feet, idle\n');
fprintf('Figure saved: inference_report.png\n');
