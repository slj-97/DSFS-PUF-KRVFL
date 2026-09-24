function value = puf_objective(X, centers, fuzzifier)
%PUF_OBJECTIVE Membership-eliminated fuzzy within-cluster objective.
%
%   Non-finite centres are rejected with +inf.  This guard is required for
%   correctness of any population-based search that calls this function: the
%   distance computation below maps a NaN configuration onto an all-zero
%   distance matrix, so without the guard an invalid candidate would be scored
%   as a perfect solution and the search would lock onto it.

if any(~isfinite(centers(:)))
    value = inf;
    return;
end

U = dpfk.memberships(X, centers, fuzzifier);
D2 = dpfk.pairwise_squared_distance(X, centers);
value = sum((U .^ fuzzifier) .* D2, 'all');
end
