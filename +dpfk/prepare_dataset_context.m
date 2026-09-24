function common = prepare_dataset_context(config, baseDirectory, opts)
%PREPARE_DATASET_CONTEXT Load data and create one shared chronological split.

data = dpfk.load_dataset(config, baseDirectory);
samples = dpfk.build_samples(data, opts.useTargetLag);
n = numel(samples.y);
nTrain = floor(opts.outerTrainFraction * n);
if nTrain < 8 || nTrain >= n
    error('dpfk:InvalidOuterSplit', 'Invalid 70/30 split for %s.', config.name);
end

common.config = config;
common.data = data;
common.samples = samples;
common.trainIdx = (1:nTrain)';
common.testIdx = ((nTrain + 1):n)';
common.folds = dpfk.expanding_folds(nTrain, opts.innerFolds);
end
