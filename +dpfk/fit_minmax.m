function params = fit_minmax(X)
%FIT_MINMAX Fit column-wise [0,1] normalization parameters.

params.minimum = min(X, [], 1);
rawRange = max(X, [], 1) - params.minimum;
params.constant = rawRange <= eps(max(1, max(abs(X), [], 1)));
params.range = rawRange;
params.range(params.constant) = 1;
end
