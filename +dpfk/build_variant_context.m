function context = build_variant_context(common, useDSFS, opts)
%BUILD_VARIANT_CONTEXT Select inputs and global parameters for one variant.

selectionTimer = tic;
samples = common.samples;
tr = common.trainIdx;

if useDSFS
    featureSelection = dpfk.select_features( ...
        samples.Xexo(tr, :), samples.yLag(tr), samples.y(tr), common.folds, ...
        opts.useTargetLag);
    selected = featureSelection.indices;
else
    p = size(samples.Xexo, 2);
    selected = 1:p;
    featureSelection.enabled = false;
    featureSelection.k = p;
    featureSelection.indices = selected;
    featureSelection.order = selected;
    featureSelection.scores = [];
    featureSelection.kendall = [];
    featureSelection.dcor = [];
    featureSelection.foldRMSE = [];
    featureSelection.meanRMSE = [];
    featureSelection.minimumIndex = [];
    featureSelection.oneSE = [];
end
featureSelection.enabled = useDSFS;

X = dpfk.assemble_inputs( ...
    samples.Xexo(:, selected), samples.yLag, opts.useTargetLag);
context.common = common;
context.useDSFS = useDSFS;
context.featureSelection = featureSelection;
context.selectedCandidateIndices = selected;
context.selectedOriginalIndices = common.config.exogenousIndices(selected);
context.selectedNames = samples.featureNames(selected);
context.Xtrain = X(common.trainIdx, :);
context.yTrain = samples.y(common.trainIdx);
context.Xtest = X(common.testIdx, :);
context.yTest = samples.y(common.testIdx);
context.globalSelection = dpfk.select_global_params( ...
    context.Xtrain, context.yTrain, common.folds, opts);
context.globalParams.C = context.globalSelection.C;
context.globalParams.gamma = context.globalSelection.gamma;
context.globalParams.alpha = context.globalSelection.alpha;
context.globalParams.kernelType = context.globalSelection.kernelType;
% This deterministic cache is reused by every PUF seed for this context.
context.nestedGlobalParams = dpfk.select_nested_global_params( ...
    context.Xtrain, context.yTrain, common.folds, opts);
context.selectionRuntime = toc(selectionTimer);
end
