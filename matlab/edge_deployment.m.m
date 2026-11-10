%% ============================================================
%  STEP 1 OF EDGE DEPLOYMENT: Prepare model for code generation
%  Run this ONCE on your development machine (not the Pi).
%  MATLAB Coder cannot codegen a ClassificationECOC object directly -
%  it needs the "coder-ready" format produced by saveLearnerForCoder.
%% ============================================================

clear; clc;

fprintf('Loading trained model...\n');
S = load('final_csp_svm_model.mat', 'finalModel', 'muFinal', 'sigmaFinal', ...
    'bestFeatureIdx', 'featureNamesFiltered');

fprintf('Converting to coder-compatible format...\n');
saveLearnerForCoder(S.finalModel, 'coder_model.mat');

% Standardization stats and feature indices are plain numeric arrays,
% so they can be saved as a normal .mat and loaded with coder.load
% inside the entry-point function (see predictBCI.m).
muFinal = S.muFinal;
sigmaFinal = S.sigmaFinal;
bestFeatureIdx = S.bestFeatureIdx;
save('coder_preproc.mat', 'muFinal', 'sigmaFinal', 'bestFeatureIdx');

modelInfo = dir('coder_model.mat');
preprocInfo = dir('coder_preproc.mat');
fprintf('\nWrote coder_model.mat (%.1f KB) and coder_preproc.mat (%.1f KB)\n', ...
    modelInfo.bytes/1024, preprocInfo.bytes/1024);
fprintf('Next: run codegen_raspi.m to cross-compile for the Pi.\n');