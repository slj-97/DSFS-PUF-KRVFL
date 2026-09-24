function run = evaluate_puf_context(context, opts, seed)
%EVALUATE_PUF_CONTEXT Fit and test one stochastic PUF global-local variant.

seedTimer = tic;
pufSelection = dpfk.select_puf_params(context.Xtrain, context.yTrain, ...
    context.globalParams, context.common.folds, opts, seed, ...
    context.nestedGlobalParams);

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
globalModel = dpfk.train_krvfl(XtrainN, yTrainN, ...
    context.globalParams.C, context.globalParams.gamma, ...
    context.globalParams.alpha, [], kt);

finalCenterSeed = seed + 900000 + 10 * pufSelection.clusterCount;
puf = dpfk.optimize_puf_centers(XtrainN, pufSelection.clusterCount, ...
    opts.fuzzifier, opts.pso, finalCenterSeed);
oof = dpfk.make_oof_residuals(context.Xtrain, context.yTrain, ...
    context.globalParams, opts.innerFolds);
Xresidual = XtrainN(oof.indices, :);
residualNormalized = oof.residual / yp.range;
Uresidual = dpfk.memberships(Xresidual, puf.centers, opts.fuzzifier);
localModels = dpfk.train_local_models(Xresidual, residualNormalized, ...
    Uresidual, pufSelection.CLocal, context.globalParams.gamma, ...
    context.globalParams.alpha, kt);

Utest = dpfk.memberships(XtestN, puf.centers, opts.fuzzifier);
globalPrediction = dpfk.predict_krvfl(globalModel, XtestN);
localCorrection = dpfk.predict_local_correction(localModels, XtestN, Utest);
prediction = dpfk.inverse_minmax(globalPrediction + localCorrection, yp);

run.seed = seed;
run.clusterCount = pufSelection.clusterCount;
run.CLocal = pufSelection.CLocal;
run.meanValidationRMSE = pufSelection.meanRMSE;
run.meanXB = pufSelection.meanXB;
run.pufSelection = pufSelection;
run.centers = puf.centers;
run.membershipWeightSums = sum(Uresidual, 1);
run.prediction = prediction;
run.truth = context.yTest;
run.targetRows = context.common.samples.targetRows(context.common.testIdx);
run.metrics = dpfk.metrics(context.yTest, prediction, context.yTrain);
run.runtime = context.selectionRuntime + toc(seedTimer);
end
