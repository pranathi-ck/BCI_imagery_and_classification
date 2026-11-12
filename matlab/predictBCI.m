function labelOut = predictBCI(x) %#codegen
% PREDICTBCI  Codegen entry point: raw 96-feature CSP row in, class out.
%
% Mirrors inference.m exactly: select the same 72 features, apply the
% same saved standardization, predict with the same trained model.
% No training or feature ranking happens here - everything is frozen
% at the values computed on the development machine.
%
% x must be a 1x96 (or Nx96) double row of raw CSP features, in the
% SAME column order used during training (see featureNamesFiltered
% saved alongside the model - the calling code is responsible for
% building x in that order).

persistent model mu sigma featIdx

if isempty(model)
    model = loadLearnerForCoder('coder_model.mat');
    pp = coder.load('coder_preproc.mat', 'muFinal', 'sigmaFinal', 'bestFeatureIdx');
    mu = pp.muFinal;
    sigma = pp.sigmaFinal;
    featIdx = pp.bestFeatureIdx;
end

xSel = x(:, featIdx);
xStd = (xSel - mu) ./ sigma;

labelOut = predict(model, xStd);
end

% NOTE on categorical codegen: predict() on a categorical-trained
% ClassificationECOC returns a categorical, which recent MATLAB Coder
% releases (R2022a+) support for fixed-category outputs. If your
% MATLAB/Coder version rejects categorical as a codegen output type,
% swap the last line for a numeric class index instead:
%   [~, scoreOut] = predict(model, xStd);
%   [~, labelOut] = max(scoreOut, [], 2);   % labelOut: 1..4 uint8 code
% and keep a fixed mapping (1=feet,2=idle,3=left,4=right, matching
% categories(Y) order from training) in EDGE_README.md.