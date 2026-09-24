function X = inverse_minmax(Xn, params)
%INVERSE_MINMAX Reverse a fitted min-max transformation.

X = Xn .* params.range + params.minimum;
end
