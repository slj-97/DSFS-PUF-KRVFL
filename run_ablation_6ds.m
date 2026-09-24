function run_ablation_6ds()
%RUN_ABLATION_6DS Run the four-model ablation on six datasets (seeds 42:51).

codeDir = fileparts(mfilename('fullpath'));
addpath(codeDir);
cd(codeDir);

selected = {'SeoulBike', 'BS', 'Stock', 'SteelEnergy', 'PedalMe', 'DFO'};

opts = dpfk.default_options();
opts.saveResults = true;
opts.verbose = false;
opts.seeds = 42:51;

configs = dpfk_dataset_config();
keep = false(1, numel(configs));
for i = 1:numel(configs)
    keep(i) = any(strcmp(configs(i).name, selected));
end
configs = configs(keep);
assert(numel(configs) == numel(selected), ...
    'Expected %d selected datasets, found %d', numel(selected), numel(configs));

outDir = fullfile(codeDir, 'results', 'ablation_6ds_confirm');
if ~isfolder(outDir), mkdir(outDir); end
logPath = fullfile(outDir, 'confirm_log.txt');
logFid = fopen(logPath, 'a');
logCleanup = onCleanup(@() fclose(logFid)); %#ok<NASGU>

fprintf(logFid, '\n===== 6-dataset confirmation started %s =====\n', datestr(now));
fprintf('6-dataset confirmation started. Log: %s\n', logPath);

allResults = struct([]);
for i = 1:numel(configs)
    config = configs(i);
    t0 = tic;
    fprintf('\n[%d/%d] %s ...\n', i, numel(configs), config.name);
    fprintf(logFid, '[%d/%d] %s start %s\n', i, numel(configs), config.name, datestr(now));

    try
        common = dpfk.prepare_dataset_context(config, codeDir, opts);
        fullContext = dpfk.build_variant_context(common, false, opts);
        dsfsContext = dpfk.build_variant_context(common, true, opts);

        models = { 'KRVFL', false, false; 'DSFS-KRVFL', true, false; ...
                   'PUF-KRVFL', false, true; 'DSFS-PUF-KRVFL', true, true };
        metricNames = {'RMSE','MAE','MAPE','R2','WI'};
        summaryHeader = {'Model'};
        for k = 1:numel(metricNames)
            summaryHeader{end+1} = [metricNames{k} '_mean']; %#ok<AGROW>
            summaryHeader{end+1} = [metricNames{k} '_std'];  %#ok<AGROW>
        end
        runHeader = [{'Model','seed'}, metricNames];
        summaryRows = cell(4, numel(summaryHeader));
        runRows = cell(0, numel(runHeader));

        for m = 1:size(models,1)
            modelName = models{m,1};
            if models{m,2}, ctx = dsfsContext; else, ctx = fullContext; end
            usePUF = models{m,3};

            runs = struct([]);
            detRun = [];
            for s = 1:numel(opts.seeds)
                if usePUF
                    r = dpfk.evaluate_puf_context(ctx, opts, opts.seeds(s));
                else
                    if isempty(detRun)
                        detRun = dpfk.evaluate_global_context(ctx);
                    end
                    r = detRun;
                end
                r.seed = opts.seeds(s);
                if s == 1, runs = r; else, runs(s) = r; end

                mvals = zeros(1, numel(metricNames));
                for k = 1:numel(metricNames)
                    mvals(k) = r.metrics.(metricNames{k});
                end
                runRows(end+1,:) = [{modelName}, num2cell(r.seed), num2cell(mvals)]; %#ok<AGROW>
            end

            summaryRow = {modelName};
            for k = 1:numel(metricNames)
                v = arrayfun(@(x) x.metrics.(metricNames{k}), runs);
                summaryRow{end+1} = mean(v, 'omitnan'); %#ok<AGROW>
                summaryRow{end+1} = std(v, 'omitnan');  %#ok<AGROW>
            end
            summaryRows(m,:) = summaryRow;

            fprintf('    %-16s RMSE=%.6g (std %.3g)  R2=%.6g  WI=%.6g\n', ...
                modelName, summaryRow{2}, summaryRow{3}, summaryRow{8}, summaryRow{10});
        end

        summaryTable = cell2table(summaryRows, 'VariableNames', summaryHeader);
        runTable = cell2table(runRows, 'VariableNames', runHeader);
        writetable(summaryTable, fullfile(outDir, sprintf('%s_summary.csv', config.name)));
        writetable(runTable, fullfile(outDir, sprintf('%s_runs.csv', config.name)));

        res = struct();
        res.dataset = config.name;
        res.summary = summaryTable;
        res.runs = runTable;
        res.selectedFeatures = dsfsContext.featureSelection;
        res.globalParams = dsfsContext.globalParams;
        res.nSamples = numel(common.samples.y);
        res.nTrain = numel(common.trainIdx);
        res.nTest = numel(common.testIdx);
        res.elapsed = toc(t0);
        if isempty(allResults), allResults = res; else, allResults(i) = res; end

        fprintf('    k* = %d, nSamples = %d, done in %.1f s\n', ...
            numel(res.selectedFeatures.indices), res.nSamples, res.elapsed);
        fprintf(logFid, '[%d/%d] %s done in %.1f s (k*=%d)\n', ...
            i, numel(configs), config.name, res.elapsed, ...
            numel(res.selectedFeatures.indices));
        save(fullfile(outDir, 'all_results_6ds.mat'), 'allResults');
    catch err
        fprintf('    *** FAILED: %s\n', err.message);
        fprintf(logFid, '[%d/%d] %s FAILED: %s\n', i, numel(configs), config.name, err.message);
    end
end
fprintf(logFid, '===== 6-dataset confirmation finished %s =====\n', datestr(now));
fprintf('\n6-dataset confirmation complete.\n');
end
