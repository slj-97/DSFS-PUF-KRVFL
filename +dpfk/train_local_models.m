function models = train_local_models(X, residual, memberships, C, gamma, alpha, kernelType)
%TRAIN_LOCAL_MODELS Fit all fuzzy-weighted residual KRVFLs.

if nargin < 7 || isempty(kernelType)
    kernelType = 'mixed';
end
clusterCount = size(memberships, 2);
models = cell(clusterCount, 1);
for j = 1:clusterCount
    models{j} = dpfk.train_krvfl( ...
        X, residual, C, gamma, alpha, memberships(:, j), kernelType);
end
end
