function selection = select_puf_params(X, y, globalParams, folds, opts, seed, nestedGlobalParams)
%SELECT_PUF_PARAMS End-to-end selection of c and one common local C.
% globalParams is retained in the public signature for compatibility. During
% validation, global parameters are re-selected within each outer-fold
% training prefix; it is not used as a fixed validation-time configuration.
if nargin < 7 || isempty(nestedGlobalParams)
    nestedGlobalParams = dpfk.select_nested_global_params(X, y, folds, opts);
end

coarseLogC = opts.log2C.coarse;
[coarseTable, coarseRMSE, coarseXB] = evaluate_grid( ...
    X, y, folds, nestedGlobalParams, opts.clusterCandidates, coarseLogC, opts, seed);
[~, coarseBest] = min(mean(coarseRMSE, 2));
fineLogC = unique(coarseTable.log2CLocal(coarseBest) + opts.fineOffsets);
[fineTable, fineRMSE, fineXB] = evaluate_grid( ...
    X, y, folds, nestedGlobalParams, opts.clusterCandidates, fineLogC, opts, seed);

candidateTable = [coarseTable; fineTable];
foldRMSE = [coarseRMSE; fineRMSE];
foldXB = [coarseXB; fineXB];
meanRMSE = mean(foldRMSE, 2);
meanXB = mean(foldXB, 2);
[minimumRMSE, minimumIndex] = min(meanRMSE);
oneSE = std(foldRMSE(minimumIndex, :), 0, 2) / sqrt(numel(folds));
eligible = find(meanRMSE <= minimumRMSE + oneSE + 1e-12);

rankTable = [candidateTable.clusterCount(eligible), ...
    2.^candidateTable.log2CLocal(eligible), eligible];
rankTable = sortrows(rankTable, [1 2 3]);
bestIndex = rankTable(1, 3);

selection.clusterCount = candidateTable.clusterCount(bestIndex);
selection.CLocal = 2 ^ candidateTable.log2CLocal(bestIndex);
selection.meanRMSE = meanRMSE(bestIndex);
selection.meanXB = meanXB(bestIndex);
selection.oneSE = oneSE;
selection.candidates = candidateTable;
selection.foldRMSE = foldRMSE;
selection.foldXB = foldXB;
selection.selectedIndex = bestIndex;
end

function [candidateTable, foldRMSE, foldXB] = evaluate_grid( ...
    X, y, folds, nestedGlobalParams, clusterCandidates, log2CLocal, opts, seed)

[cMesh, cLocalMesh] = ndgrid(clusterCandidates, log2CLocal);
candidateTable = table(cMesh(:), cLocalMesh(:), ...
    'VariableNames', {'clusterCount','log2CLocal'});
foldRMSE = nan(height(candidateTable), numel(folds));
foldXB = nan(height(candidateTable), numel(folds));

for q = 1:numel(folds)
    tr = folds(q).trainIdx;
    va = folds(q).valIdx;
    % Nested selection is computed once per training context and reused here:
    % global KRVFL parameters are selected using only the current outer-fold
    % training prefix, never the validation block.
    foldGlobalParams = nestedGlobalParams{q};
    xp = dpfk.fit_minmax(X(tr, :));
    yp = dpfk.fit_minmax(y(tr));
    Xtr = dpfk.apply_minmax(X(tr, :), xp);
    Xva = dpfk.apply_minmax(X(va, :), xp);
    ytr = dpfk.apply_minmax(y(tr), yp);

    globalModel = dpfk.train_krvfl(Xtr, ytr, foldGlobalParams.C, ...
        foldGlobalParams.gamma, foldGlobalParams.alpha, [], ...
        local_kernel_type(foldGlobalParams));
    globalValidation = dpfk.predict_krvfl(globalModel, Xva);

    oof = dpfk.make_oof_residuals( ...
        X(tr, :), y(tr), foldGlobalParams, opts.innerFolds);
    Xresidual = Xtr(oof.indices, :);
    residualNormalized = oof.residual / yp.range;

    for c = clusterCandidates
        centerSeed = seed + 1000 * q + 10 * c;
        puf = dpfk.optimize_puf_centers( ...
            Xtr, c, opts.fuzzifier, opts.pso, centerSeed);
        Uresidual = dpfk.memberships( ...
            Xresidual, puf.centers, opts.fuzzifier);
        Uvalidation = dpfk.memberships( ...
            Xva, puf.centers, opts.fuzzifier);
        xb = dpfk.xb_index(Xtr, puf.centers, opts.fuzzifier);

        rows = find(candidateTable.clusterCount == c);
        for row = rows'
            CLocal = 2 ^ candidateTable.log2CLocal(row);
            localModels = dpfk.train_local_models( ...
                Xresidual, residualNormalized, Uresidual, CLocal, ...
                foldGlobalParams.gamma, foldGlobalParams.alpha, ...
                local_kernel_type(foldGlobalParams));
            correction = dpfk.predict_local_correction( ...
                localModels, Xva, Uvalidation);
            prediction = dpfk.inverse_minmax( ...
                globalValidation + correction, yp);
            foldRMSE(row, q) = sqrt(mean((y(va) - prediction).^2));
            foldXB(row, q) = xb;
        end
    end
end
end

function kt = local_kernel_type(globalParams)
%LOCAL_KERNEL_TYPE Read the selected kernel family, defaulting to legacy mixed.
if isstruct(globalParams) && isfield(globalParams, 'kernelType') ...
        && ~isempty(globalParams.kernelType)
    kt = globalParams.kernelType;
else
    kt = 'mixed';
end
end
