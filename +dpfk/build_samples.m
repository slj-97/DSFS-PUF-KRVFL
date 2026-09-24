function samples = build_samples(data, useTargetLag)
%BUILD_SAMPLES Construct aligned one-step samples with optional target lag.

if nargin < 2
    useTargetLag = true;
end

raw = data.raw;
if size(raw, 1) < 2
    error('dpfk:TooFewRows', 'At least two original rows are required.');
end

Xexo = raw(2:end, data.exogenousIndices);
yLag = raw(1:end-1, data.targetIndex);
y = raw(2:end, data.targetIndex);
targetRows = (2:size(raw, 1))';

valid = all(isfinite(Xexo), 2) & isfinite(yLag) & isfinite(y);
if ~useTargetLag
    valid = all(isfinite(Xexo), 2) & isfinite(y);
end
samples.Xexo = Xexo(valid, :);
samples.yLag = yLag(valid);
samples.y = y(valid);
samples.targetRows = targetRows(valid);
samples.featureNames = data.exogenousNames;
samples.droppedRows = targetRows(~valid);
samples.useTargetLag = useTargetLag;

if size(samples.Xexo, 1) < 12
    error('dpfk:TooFewSamples', ...
        'Only %d complete one-step samples remain for %s.', ...
        size(samples.Xexo, 1), data.name);
end
end
