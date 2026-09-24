function run = evaluate_global_context(context)
%EVALUATE_GLOBAL_CONTEXT Fit and test one deterministic global-only variant.

fitTimer = tic;
xp = dpfk.fit_minmax(context.Xtrain);
yp = dpfk.fit_minmax(context.yTrain);
XtrainN = dpfk.apply_minmax(context.Xtrain, xp);
XtestN = dpfk.apply_minmax(context.Xtest, xp);
yTrainN = dpfk.apply_minmax(context.yTrain, yp);

if isfield(context.globalParams, 'kernelType') && ~isempty(context.globalParams.kernelType)
    kt = context.globalParams.kernelType;
else
    kt = 'mixed';
end
model = dpfk.train_krvfl(XtrainN, yTrainN, ...
    context.globalParams.C, context.globalParams.gamma, ...
    context.globalParams.alpha, [], kt);
prediction = dpfk.inverse_minmax( ...
    dpfk.predict_krvfl(model, XtestN), yp);

run.seed = NaN;
run.clusterCount = NaN;
run.CLocal = NaN;
run.meanValidationRMSE = context.globalSelection.meanRMSE;
run.meanXB = NaN;
run.pufSelection = [];
run.centers = [];
run.membershipWeightSums = [];
run.prediction = prediction;
run.truth = context.yTest;
run.targetRows = context.common.samples.targetRows(context.common.testIdx);
run.metrics = dpfk.metrics(context.yTest, prediction, context.yTrain);
run.runtime = context.selectionRuntime + toc(fitTimer);
end
