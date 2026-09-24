function folds = expanding_folds(sampleCount, foldCount)
%EXPANDING_FOLDS Create chronological expanding-window validation folds.

if nargin < 2
    foldCount = 3;
end
initialCount = floor(sampleCount / 2);
remaining = sampleCount - initialCount;
if remaining < foldCount
    error('dpfk:TooFewSamplesForFolds', ...
        '%d samples cannot form %d nonempty validation blocks.', ...
        sampleCount, foldCount);
end

blockSizes = floor(remaining / foldCount) * ones(1, foldCount);
blockSizes(1:mod(remaining, foldCount)) = ...
    blockSizes(1:mod(remaining, foldCount)) + 1;

folds = repmat(struct('trainIdx', [], 'valIdx', []), foldCount, 1);
cursor = initialCount + 1;
for q = 1:foldCount
    valIdx = cursor:(cursor + blockSizes(q) - 1);
    folds(q).trainIdx = (1:(cursor - 1))';
    folds(q).valIdx = valIdx';
    cursor = valIdx(end) + 1;
end
end
