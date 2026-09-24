function oof = make_oof_residuals(X, y, params, foldCount)
%MAKE_OOF_RESIDUALS Forward out-of-fold residuals in original y units.

if isfield(params, 'kernelType') && ~isempty(params.kernelType)
    kt = params.kernelType;
else
    kt = 'mixed';
end
folds = dpfk.expanding_folds(size(X, 1), foldCount);
prediction = nan(size(y));
for q = 1:numel(folds)
    tr = folds(q).trainIdx;
    va = folds(q).valIdx;
    xp = dpfk.fit_minmax(X(tr, :));
    yp = dpfk.fit_minmax(y(tr));
    Xtr = dpfk.apply_minmax(X(tr, :), xp);
    Xva = dpfk.apply_minmax(X(va, :), xp);
    ytr = dpfk.apply_minmax(y(tr), yp);
    model = dpfk.train_krvfl( ...
        Xtr, ytr, params.C, params.gamma, params.alpha, [], kt);
    predN = dpfk.predict_krvfl(model, Xva);
    prediction(va) = dpfk.inverse_minmax(predN, yp);
end
valid = find(isfinite(prediction));
oof.indices = valid;
oof.prediction = prediction(valid);
oof.residual = y(valid) - prediction(valid);
oof.allPrediction = prediction;
end
