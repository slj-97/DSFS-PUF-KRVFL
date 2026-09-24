function selection = select_global_params(X, y, folds, opts)
%SELECT_GLOBAL_PARAMS Coarse-to-fine KRVFL validation over kernels + hyperparams.
%
% Searches kernelType x C x gamma x alpha on the supplied validation folds.
% When opts.kernelCandidates is a single entry (or absent) the legacy mixed
% kernel behaviour is preserved exactly.
%
% Returns the winning hyper-parameters plus the full candidate table so the
% manuscript can report which kernel each dataset selected.

if ~isfield(opts, 'kernelCandidates') || isempty(opts.kernelCandidates)
    kernelCandidates = {'mixed'};
else
    kernelCandidates = opts.kernelCandidates;
end

% --- Stage 1: coarse search over every kernel family -------------------------
bestCoarse = struct('meanRMSE', inf, 'log2C', NaN, 'log2Gamma', NaN, ...
    'alpha', NaN, 'kernelType', '');
coarseRows = table();
coarseErrors = [];

for kk = 1:numel(kernelCandidates)
    kt = kernelCandidates{kk};
    [rows, errors] = evaluate_grid(X, y, folds, ...
        opts.log2C.coarse, opts.log2Gamma.coarse, alpha_grid_for(kt, opts), kt);
    meanErr = mean(errors, 2);
    [m, i] = min(meanErr);
    if m < bestCoarse.meanRMSE
        bestCoarse.meanRMSE = m;
        bestCoarse.log2C = rows.log2C(i);
        bestCoarse.log2Gamma = rows.log2Gamma(i);
        bestCoarse.alpha = rows.alpha(i);
        bestCoarse.kernelType = kt;
    end
    if isempty(coarseRows)
        coarseRows = rows;
    else
        coarseRows = [coarseRows; rows]; %#ok<AGROW>
    end
    coarseErrors = [coarseErrors; errors]; %#ok<AGROW>
end

% --- Stage 2: fine search around the winning coarse point --------------------
fineLogC = unique(bestCoarse.log2C + opts.fineOffsets);
fineLogG = unique(bestCoarse.log2Gamma + opts.fineOffsets);
[fineRows, fineErrors] = evaluate_grid(X, y, folds, fineLogC, fineLogG, ...
    alpha_grid_for(bestCoarse.kernelType, opts), bestCoarse.kernelType);
fineMean = mean(fineErrors, 2);
bestFine = choose_best(fineRows, fineMean);

selection.C = 2 ^ bestFine.log2C;
selection.gamma = 2 ^ bestFine.log2Gamma;
selection.alpha = bestFine.alpha;
selection.kernelType = bestCoarse.kernelType;
selection.meanRMSE = bestFine.meanRMSE;
selection.coarseCandidates = coarseRows;
selection.coarseFoldRMSE = coarseErrors;
selection.fineCandidates = fineRows;
selection.fineFoldRMSE = fineErrors;
selection.kernelCandidates = kernelCandidates;
end

function ag = alpha_grid_for(kernelType, opts)
%ALPHA_GRID_FOR Single kernels ignore alpha, so collapse the grid to one value
% and save the whole factor of 5 in runtime.
if strcmpi(kernelType, 'mixed')
    ag = opts.alphaGrid;
else
    ag = 0.5;   % placeholder; unused by single kernels
end
end

function [rows, errors] = evaluate_grid(X, y, folds, logC, logG, alphaGrid, kernelType)
if nargin < 7 || isempty(kernelType)
    kernelType = 'mixed';
end
[cMesh, gMesh, aMesh] = ndgrid(logC, logG, alphaGrid);
rows = table(cMesh(:), gMesh(:), aMesh(:), ...
    repmat({kernelType}, numel(cMesh), 1), ...
    'VariableNames', {'log2C','log2Gamma','alpha','kernelType'});
errors = nan(height(rows), numel(folds));
for r = 1:height(rows)
    C = 2 ^ rows.log2C(r);
    gamma = 2 ^ rows.log2Gamma(r);
    alpha = rows.alpha(r);
    for q = 1:numel(folds)
        tr = folds(q).trainIdx;
        va = folds(q).valIdx;
        xp = dpfk.fit_minmax(X(tr, :));
        yp = dpfk.fit_minmax(y(tr));
        Xtr = dpfk.apply_minmax(X(tr, :), xp);
        Xva = dpfk.apply_minmax(X(va, :), xp);
        ytr = dpfk.apply_minmax(y(tr), yp);
        model = dpfk.train_krvfl(Xtr, ytr, C, gamma, alpha, [], kernelType);
        pred = dpfk.inverse_minmax(dpfk.predict_krvfl(model, Xva), yp);
        errors(r, q) = sqrt(mean((y(va) - pred).^2));
    end
end
end

function best = choose_best(rows, meanErrors)
rankTable = [meanErrors(:), 2.^rows.log2C, ...
    2.^rows.log2Gamma, rows.alpha, (1:height(rows))'];
rankTable = sortrows(rankTable, [1 2 3 4 5]);
idx = rankTable(1, 5);
best.log2C = rows.log2C(idx);
best.log2Gamma = rows.log2Gamma(idx);
best.alpha = rows.alpha(idx);
best.meanRMSE = meanErrors(idx);
end
