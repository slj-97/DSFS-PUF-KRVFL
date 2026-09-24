function K = kernel_matrix(XA, XB, gamma, alpha, kernelType)
%KERNEL_MATRIX Kernel dispatch for the DSFS-PUF-KRVFL global regression stage.
%
%   kernel_matrix(XA, XB, gamma, alpha)                -> mixed (legacy default)
%   kernel_matrix(XA, XB, gamma, alpha, 'linear')      -> single kernels:
%       'linear','poly2','poly3','rbf','matern32','matern52','laplacian','rq'
%
% alpha is retained as the mixing coefficient ONLY for the legacy mixed
% kernel (alpha=0 -> pure Matern-3/2, alpha=1 -> pure linear). Single kernels
% ignore alpha, so callers can pass any value without changing behaviour.
%
% Single-kernel semantics:
%   linear     : <xa, xb>
%   poly2/3    : (1 + <xa, xb>)^d          (d = 2 or 3)
%   rbf        : exp(-gamma * ||xa-xb||^2)
%   matern32   : (1 + s) exp(-s),          s = sqrt(3)*gamma*r
%   matern52   : (1 + s + s^2/3) exp(-s),  s = sqrt(5)*gamma*r
%   laplacian  : exp(-gamma * ||xa-xb||_1)
%   rq         : (1 + D2/(2*a))^(-a),      a = max(1/gamma, 1)

if nargin < 5 || isempty(kernelType)
    kernelType = 'mixed';
end

D2 = dpfk.pairwise_squared_distance(XA, XB);
r = sqrt(max(D2, 0));

switch lower(kernelType)
    case 'mixed'
        Klin = XA * XB';
        s = sqrt(3) * gamma * r;
        Kmatern = (1 + s) .* exp(-s);
        if nargin < 4 || isempty(alpha)
            alpha = 0.5;
        end
        K = alpha * Klin + (1 - alpha) * Kmatern;

    case 'linear'
        K = XA * XB';

    case 'poly2'
        K = (1 + XA * XB') .^ 2;

    case 'poly3'
        K = (1 + XA * XB') .^ 3;

    case 'rbf'
        K = exp(-gamma * D2);

    case 'matern32'
        s = sqrt(3) * gamma * r;
        K = (1 + s) .* exp(-s);

    case 'matern52'
        s = sqrt(5) * gamma * r;
        K = (1 + s + (s.^2) / 3) .* exp(-s);

    case 'laplacian'
        K = exp(-gamma * local_l1_distance(XA, XB));

    case 'rq'
        alphaRQ = max(1 / max(gamma, eps), 1);
        K = (1 + D2 / (2 * alphaRQ)) .^ (-alphaRQ);

    otherwise
        error('dpfk:UnknownKernel', 'Unknown kernel type: %s', kernelType);
end
end

function D1 = local_l1_distance(XA, XB)
%LOCAL_L1_DISTANCE Row-wise Manhattan distance matrix, computed in row blocks
% so that very wide feature matrices do not allocate an NxMxD temporary.
nA = size(XA, 1);
nB = size(XB, 1);
D1 = zeros(nA, nB);
if nA == 0 || nB == 0
    return;
end
block = max(1, floor(1e7 / max(nB * size(XB, 2), 1)));
for s = 1:block:nA
    e = min(s + block - 1, nA);
    D1(s:e, :) = sum(abs(bsxfun(@minus, ...
        reshape(XA(s:e, :), [e - s + 1, 1, size(XA, 2)]), ...
        reshape(XB, [1, nB, size(XB, 2)]))), 3);
end
end
