%% ============================================================
%  SIMULATED EDGE BENCHMARK (no physical hardware required)
%
%  Honesty note: this does NOT measure a Raspberry Pi or any other
%  board. It measures the compiled C implementation of the model
%  (via MATLAB Coder -> MEX), which removes MATLAB's interpreter
%  overhead and is the standard desktop proxy for "how fast would
%  the actual algorithm run once compiled to native code" when no
%  physical target is available. Absolute numbers will differ from
%  a real embedded CPU (a Cortex-A72 is slower per-core than a
%  desktop x86 core) - report this as a SIMULATED / feasibility
%  estimate, not a measured on-device result.
%% ============================================================

clear; clc; close all;

%% ------------------------------------------------------------
% CONFIG
% ---------------------------------------------------------------
testFile = 'step13_csp_test_features.csv';
outCsv   = 'edge_metrics.csv';
outFig   = 'edge_metrics.png';

% Rough scaling factor to translate host-CPU compiled timing into a
% ballpark ARM Cortex-A72 @ 1.5GHz estimate (Raspberry Pi 4 class).
% This is a coarse, clearly-labeled ESTIMATE based on typical
% single-core integer/float throughput ratios, not a measurement -
% delete this section entirely if you'd rather report host-only
% numbers and avoid implying board-specific accuracy.
ARM_SLOWDOWN_FACTOR = 4;  % typical range 3-6x for scalar/SVM-style workloads

%% ------------------------------------------------------------
% STEP 0: prepare coder-compatible model artifacts if not already done
% ---------------------------------------------------------------
if ~isfile('coder_model.mat') || ~isfile('coder_preproc.mat')
    fprintf('Coder artifacts not found - running save_for_coder.m first...\n');
    run('save_for_coder.m');
end

%% ------------------------------------------------------------
% STEP 1: generate a MEX version of predictBCI (real compiled C code,
% run in-process on this machine - no external hardware involved)
% ---------------------------------------------------------------
usedCompiled = true;
try
    fprintf('Compiling predictBCI.m to native code via MATLAB Coder (MEX)...\n');
    argType = coder.typeof(double(0), [1 96], [0 0]);
    codegen predictBCI -args {argType} -o predictBCI_mex -report
    fprintf('Compiled successfully: predictBCI_mex\n');
catch ME
    warning('Codegen/MEX compilation failed (%s). Falling back to timing the plain MATLAB function instead of compiled code.', ME.message);
    usedCompiled = false;
end

%% ------------------------------------------------------------
% STEP 2: load test data
% ---------------------------------------------------------------
testTbl = readtable(testFile);
excludeNames = {'label','trial_id','test_id','subject','session'};
varNames = testTbl.Properties.VariableNames;
mask = true(1,width(testTbl));
for k = 1:numel(excludeNames)
    mask(strcmpi(varNames,excludeNames{k})) = false;
end
Xtest = testTbl{:,mask};
numTrials = size(Xtest,1);

%% ------------------------------------------------------------
% STEP 3: time inference, ONE TRIAL AT A TIME (mimics streaming
% real-time use on a device, rather than one big batched call)
% ---------------------------------------------------------------
fprintf('\nTiming %d trials, one at a time...\n', numTrials);
latencies = zeros(numTrials,1);

if usedCompiled
    % warm-up call (first MEX call pays a one-time load cost)
    predictBCI_mex(Xtest(1,:));
    for i = 1:numTrials
        row = Xtest(i,:);
        tStart = tic;
        predictBCI_mex(row); %#ok<NASGU>
        latencies(i) = toc(tStart);
    end
else
    S = load('final_csp_svm_model.mat', 'finalModel', 'muFinal', 'sigmaFinal', 'bestFeatureIdx');
    for i = 1:numTrials
        row = Xtest(i,:);
        tStart = tic;
        rowSel = row(:, S.bestFeatureIdx);
        rowStd = (rowSel - S.muFinal) ./ S.sigmaFinal;
        predict(S.finalModel, rowStd); %#ok<NASGU>
        latencies(i) = toc(tStart);
    end
end

hostMeanMs   = 1000*mean(latencies);
hostMedianMs = 1000*median(latencies);
hostP95Ms    = 1000*prctile(latencies,95);
hostP99Ms    = 1000*prctile(latencies,99);
hostThroughputHz = 1/mean(latencies);

estArmMeanMs = hostMeanMs * ARM_SLOWDOWN_FACTOR;
estArmThroughputHz = hostThroughputHz / ARM_SLOWDOWN_FACTOR;

fprintf('\n============================================================\n');
fprintf('SIMULATED EDGE BENCHMARK (n=%d trials)\n', numTrials);
fprintf('Mode: %s\n', ternary(usedCompiled, 'Compiled C code (MEX), timed on host CPU', 'Plain MATLAB predict(), timed on host CPU'));
fprintf('============================================================\n');
fprintf('Host-measured mean latency:   %.4f ms\n', hostMeanMs);
fprintf('Host-measured median latency: %.4f ms\n', hostMedianMs);
fprintf('Host-measured P95 latency:    %.4f ms\n', hostP95Ms);
fprintf('Host-measured P99 latency:    %.4f ms\n', hostP99Ms);
fprintf('Host-measured throughput:     %.1f trials/sec\n', hostThroughputHz);
fprintf('\n[ESTIMATE, not measured] Scaled ~%.0fx for a Pi 4-class ARM CPU:\n', ARM_SLOWDOWN_FACTOR);
fprintf('  Estimated mean latency:    %.4f ms\n', estArmMeanMs);
fprintf('  Estimated throughput:      %.1f trials/sec\n', estArmThroughputHz);

%% ------------------------------------------------------------
% MODEL / CODE SIZE (measured, not estimated)
% ---------------------------------------------------------------
sizeKb = @(f) ternary(isfile(f), dir(f).bytes/1024, 0);
modelMatKb   = sizeKb('final_csp_svm_model.mat');
coderModelKb = sizeKb('coder_model.mat');
coderPrepKb  = sizeKb('coder_preproc.mat');

generatedCodeKb = 0;
if isfolder('predictBCI_mex')
    d = dir(fullfile('predictBCI_mex','**','*.c*'));
    generatedCodeKb = sum([d.bytes])/1024;
end
mexFileKb = sizeKb(['predictBCI_mex.' mexext]);

fprintf('\nModel/code footprint:\n');
fprintf('  Original .mat model:        %.1f KB\n', modelMatKb);
fprintf('  Coder-ready model+preproc:  %.1f KB\n', coderModelKb + coderPrepKb);
fprintf('  Generated C/C++ source:     %.1f KB\n', generatedCodeKb);
fprintf('  Compiled MEX artifact:      %.1f KB\n', mexFileKb);

%% ------------------------------------------------------------
% WRITE edge_metrics.csv (clearly labeled as simulated)
% ---------------------------------------------------------------
metric = { ...
    'benchmark_type'; 'target_hardware'; 'num_trials'; ...
    'host_mean_latency_ms'; 'host_median_latency_ms'; ...
    'host_p95_latency_ms'; 'host_p99_latency_ms'; 'host_throughput_trials_per_sec'; ...
    'estimated_arm_slowdown_factor'; 'estimated_arm_mean_latency_ms'; ...
    'estimated_arm_throughput_trials_per_sec'; ...
    'model_mat_size_kb'; 'coder_model_size_kb'; ...
    'generated_c_source_size_kb'; 'compiled_mex_size_kb'};

value = { ...
    ternary(usedCompiled,'simulated_compiled_C_host_timed','simulated_matlab_host_timed'); ...
    'None (no physical device available - host-compiled-code simulation)'; ...
    numTrials; ...
    hostMeanMs; hostMedianMs; hostP95Ms; hostP99Ms; hostThroughputHz; ...
    ARM_SLOWDOWN_FACTOR; estArmMeanMs; estArmThroughputHz; ...
    modelMatKb; coderModelKb + coderPrepKb; ...
    generatedCodeKb; mexFileKb};

edgeTbl = table(metric, value);
writetable(edgeTbl, outCsv);
fprintf('\nWrote %s\n', outCsv);

%% ------------------------------------------------------------
% VISUALIZATION
% ---------------------------------------------------------------
figEdge = figure('Color','w','Position',[100 100 1000 380], 'Name','Simulated Edge Benchmark');

subplot(1,2,1);
histogram(1000*latencies, 30, 'FaceColor',[0.3 0.5 0.8], 'EdgeColor','w');
xline(hostMeanMs, 'r--', sprintf('mean %.3f ms',hostMeanMs), 'LineWidth',1.5);
xlabel('Latency (ms, host-timed)'); ylabel('Trial count');
title('Per-Trial Latency (Simulated)', 'FontWeight','bold','FontSize',10);
grid on; box off;

subplot(1,2,2);
bar(categorical({'.mat model','Coder model','Generated C src','Compiled MEX'}), ...
    [modelMatKb, coderModelKb+coderPrepKb, generatedCodeKb, mexFileKb], ...
    'FaceColor',[0.3 0.7 0.4]);
ylabel('Size (KB)');
title('Model/Code Footprint', 'FontWeight','bold','FontSize',10);
grid on; box off;

sgtitle(sprintf('SIMULATED Edge Benchmark - no physical device (host: %.3f ms/trial)', ...
    hostMeanMs), 'FontWeight','bold','FontSize',12);

exportgraphics(figEdge, outFig, 'Resolution', 200);
fprintf('Wrote %s\n', outFig);

function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end