function xb = xb_index(X, centers, fuzzifier)
%XB_INDEX Xie-Beni diagnostic; it never prefilters cluster candidates.

U = dpfk.memberships(X, centers, fuzzifier);
D2 = dpfk.pairwise_squared_distance(X, centers);
centerD2 = dpfk.pairwise_squared_distance(centers, centers);
centerD2(1:(size(centerD2, 1) + 1):end) = inf;
minimumSeparation = min(centerD2, [], 'all');
if minimumSeparation <= eps
    xb = inf;
else
    xb = sum((U .^ fuzzifier) .* D2, 'all') / ...
        (size(X, 1) * minimumSeparation);
end
end
