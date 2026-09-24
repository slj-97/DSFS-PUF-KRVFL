function m = metrics(y, prediction, outerTrainingY)
%METRICS RMSE, MAE, protected MAPE, Willmott index, and R-squared.

y = y(:);
prediction = prediction(:);
errorValue = y - prediction;
m.RMSE = sqrt(mean(errorValue.^2));
m.MAE = mean(abs(errorValue));
epsilonY = 1e-8 * max(1, median(abs(outerTrainingY(:))));
m.MAPE = 100 * mean(abs(errorValue) ./ max(abs(y), epsilonY));
denominator = sum((abs(prediction - mean(y)) + abs(y - mean(y))).^2);
if denominator <= eps
    m.WI = double(all(abs(errorValue) <= eps));
else
    m.WI = 1 - sum(errorValue.^2) / denominator;
end
totalSumSquares = sum((y - mean(y)).^2);
if totalSumSquares <= eps(max(1, sum(y.^2)))
    m.R2 = NaN;
else
    m.R2 = 1 - sum(errorValue.^2) / totalSumSquares;
end
m.epsilonY = epsilonY;
end
