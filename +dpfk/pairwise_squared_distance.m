function D2 = pairwise_squared_distance(XA, XB)
%PAIRWISE_SQUARED_DISTANCE Squared Euclidean distances between row vectors.
%
%   Rounding-level negative values produced by the expansion below are clamped
%   to zero.  The clamp is applied by logical indexing rather than by max(),
%   because max(NaN, 0) returns 0 in MATLAB and would silently turn a NaN
%   distance into a zero distance, i.e. into a spurious perfect match.

D2 = sum(XA.^2, 2) + sum(XB.^2, 2)' - 2 * (XA * XB');
D2(D2 < 0) = 0;
end
