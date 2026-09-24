function gamma = median_gamma(X)
%MEDIAN_GAMMA Inverse median nonzero pairwise distance (Matern length scale).

D2 = dpfk.pairwise_squared_distance(X, X);
D = sqrt(D2);
values = D(triu(true(size(D)), 1) & D > 0);
if isempty(values)
    gamma = 1;
else
    gamma = 1 / max(median(values), 1e-12);
end
end
