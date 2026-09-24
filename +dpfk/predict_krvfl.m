function yhat = predict_krvfl(model, X)
%PREDICT_KRVFL Predict with a fitted dual KRVFL.

if isfield(model, 'kernelType') && ~isempty(model.kernelType)
    kt = model.kernelType;
else
    kt = 'mixed';
end
K = dpfk.kernel_matrix(X, model.Xtrain, model.gamma, model.alpha, kt);
yhat = K * model.beta;
end
