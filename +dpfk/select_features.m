function selection = select_features(Xexo, yLag, y, folds, useTargetLag)
%SELECT_FEATURES Select exogenous feature count using the one-SE rule.

if nargin < 5
    useTargetLag = true;
end

p = size(Xexo, 2);
qCount = numel(folds);
foldRMSE = nan(qCount, p);

for q = 1:qCount
    tr = folds(q).trainIdx;
    va = folds(q).valIdx;
    [scores, ~, ~] = dpfk.dsfs_scores(Xexo(tr, :), y(tr));
    orderTable = [(-scores(:)), (1:p)'];
    [~, sortRows] = sortrows(orderTable, [1 2]);
    order = sortRows(:)';

    for k = 1:p
        cols = order(1:k);
        Xtr = dpfk.assemble_inputs( ...
            Xexo(tr, cols), yLag(tr), useTargetLag);
        Xva = dpfk.assemble_inputs( ...
            Xexo(va, cols), yLag(va), useTargetLag);
        xParams = dpfk.fit_minmax(Xtr);
        yParams = dpfk.fit_minmax(y(tr));
        XtrN = dpfk.apply_minmax(Xtr, xParams);
        XvaN = dpfk.apply_minmax(Xva, xParams);
        ytrN = dpfk.apply_minmax(y(tr), yParams);
        gamma = dpfk.median_gamma(XtrN);
        model = dpfk.train_krvfl(XtrN, ytrN, 1, gamma, 0.5);
        pred = dpfk.inverse_minmax(dpfk.predict_krvfl(model, XvaN), yParams);
        foldRMSE(q, k) = sqrt(mean((y(va) - pred).^2));
    end
end

meanRMSE = mean(foldRMSE, 1);
[minimumRMSE, minIndex] = min(meanRMSE);
standardError = std(foldRMSE(:, minIndex), 0, 1) / sqrt(qCount);
eligible = find(meanRMSE <= minimumRMSE + standardError + 1e-12);
kStar = eligible(1);

[scores, tauAbs, dcor] = dpfk.dsfs_scores(Xexo, y);
orderTable = [(-scores(:)), (1:p)'];
[~, sortRows] = sortrows(orderTable, [1 2]);
finalOrder = sortRows(:)';

selection.k = kStar;
selection.indices = finalOrder(1:kStar);
selection.order = finalOrder;
selection.scores = scores;
selection.kendall = tauAbs;
selection.dcor = dcor;
selection.foldRMSE = foldRMSE;
selection.meanRMSE = meanRMSE;
selection.minimumIndex = minIndex;
selection.oneSE = standardError;
end
