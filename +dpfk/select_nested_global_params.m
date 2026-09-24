function nested = select_nested_global_params(X, y, folds, opts)
%SELECT_NESTED_GLOBAL_PARAMS Select global parameters inside each PUF fold.
% The returned cell array is deterministic for a fixed training context and
% can be reused across PUF seeds and coarse/fine local-parameter searches.

nested = cell(numel(folds), 1);
for q = 1:numel(folds)
    tr = folds(q).trainIdx;
    nestedFolds = dpfk.expanding_folds(numel(tr), opts.innerFolds);
    selected = dpfk.select_global_params(X(tr, :), y(tr), nestedFolds, opts);
    nested{q}.C = selected.C;
    nested{q}.gamma = selected.gamma;
    nested{q}.alpha = selected.alpha;
    nested{q}.kernelType = selected.kernelType;
end
end
