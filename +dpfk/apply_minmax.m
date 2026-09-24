function Xn = apply_minmax(X, params)
%APPLY_MINMAX Apply training-derived normalization without clipping.

Xn = (X - params.minimum) ./ params.range;
if any(params.constant)
    Xn(:, params.constant) = 0;
end
end
