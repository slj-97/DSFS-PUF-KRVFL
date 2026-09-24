function U = memberships(X, centers, fuzzifier)
%MEMBERSHIPS Stable fuzzy C-means memberships (rows sum to one).

D2 = dpfk.pairwise_squared_distance(X, centers);
n = size(X, 1);
c = size(centers, 1);
U = zeros(n, c);
zeroTolerance = 1e-14;
for i = 1:n
    zeroCenters = D2(i, :) <= zeroTolerance;
    if any(zeroCenters)
        U(i, zeroCenters) = 1 / sum(zeroCenters);
    else
        invDistance = D2(i, :) .^ (-1 / (fuzzifier - 1));
        U(i, :) = invDistance / sum(invDistance);
    end
end
end
