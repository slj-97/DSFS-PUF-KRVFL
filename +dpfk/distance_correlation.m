function value = distance_correlation(x, y)
%DISTANCE_CORRELATION Classical biased sample distance correlation.

x = x(:);
y = y(:);
if numel(x) ~= numel(y)
    error('dpfk:DimensionMismatch', 'x and y must have equal length.');
end
if numel(x) < 2 || range(x) == 0 || range(y) == 0
    value = 0;
    return;
end

A = abs(x - x');
B = abs(y - y');
A = A - mean(A, 2) - mean(A, 1) + mean(A(:));
B = B - mean(B, 2) - mean(B, 1) + mean(B(:));

dCov2 = max(mean(A(:) .* B(:)), 0);
dVarX2 = mean(A(:).^2);
dVarY2 = mean(B(:).^2);
if dVarX2 <= 0 || dVarY2 <= 0
    value = 0;
else
    value = sqrt(dCov2 / sqrt(dVarX2 * dVarY2));
    value = min(max(real(value), 0), 1);
end
end
