function [scores, tauAbs, dcor] = dsfs_scores(X, y)
%DSFS_SCORES Probabilistic-sum fusion of |Kendall tau| and dCor.

p = size(X, 2);
tauAbs = zeros(p, 1);
dcor = zeros(p, 1);
for j = 1:p
    tauAbs(j) = abs(dpfk.kendall_tau_b(X(:, j), y));
    dcor(j) = dpfk.distance_correlation(X(:, j), y);
end
scores = tauAbs + dcor - tauAbs .* dcor;
scores(~isfinite(scores)) = 0;
scores = min(max(scores, 0), 1);
end
