function correction = predict_local_correction(models, X, memberships)
%PREDICT_LOCAL_CORRECTION Membership-weighted unscaled residual sum.

correction = zeros(size(X, 1), 1);
for j = 1:numel(models)
    localPrediction = dpfk.predict_krvfl(models{j}, X);
    correction = correction + memberships(:, j) .* localPrediction;
end
end
