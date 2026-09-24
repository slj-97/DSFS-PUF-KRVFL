function model = train_krvfl(X, y, C, gamma, alpha, weights, kernelType)
%TRAIN_KRVFL Fit global or fuzzy-weighted dual KRVFL.

if nargin < 6
    weights = [];
end
if nargin < 7 || isempty(kernelType)
    kernelType = 'mixed';
end
y = y(:);
n = size(X, 1);
if numel(y) ~= n || C <= 0 || gamma <= 0 || alpha < 0 || alpha > 1
    error('dpfk:InvalidKRVFLInput', 'Invalid KRVFL dimensions or parameters.');
end
K = dpfk.kernel_matrix(X, X, gamma, alpha, kernelType);
lambda = 1 / C;
if isempty(weights)
    A = K + lambda * eye(n);
    b = y;
else
    weights = max(weights(:), 0);
    if numel(weights) ~= n
        error('dpfk:DimensionMismatch', 'Weights must match training rows.');
    end
    A = weights .* K + lambda * eye(n);
    b = weights .* y;
end
model.beta = A \ b;
model.Xtrain = X;
model.C = C;
model.gamma = gamma;
model.alpha = alpha;
model.kernelType = kernelType;
end
