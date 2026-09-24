function opts = default_options()
%DEFAULT_OPTIONS Reproducible settings from the approved design.

opts.outerTrainFraction = 0.70;
opts.innerFolds = 3;
opts.useTargetLag = true;
opts.seeds = 42:51;
opts.clusterCandidates = 2:5;
opts.fuzzifier = 2;

opts.log2C.coarse = [-5 -2 1 4 7 10 13 16];
opts.log2Gamma.coarse = [-5 -3 -1 1 3];
opts.alphaGrid = [0 0.25 0.5 0.75 1];
opts.fineOffsets = [-1 -0.5 0 0.5 1];

% Kernel families considered for the global (and local) regression stage.
% 'mixed' preserves the original linear + Matern-3/2 convex combination.
% Single kernels are the focus of the kernel-study experiments.
opts.kernelCandidates = {'mixed'};
opts.allKernels = {'linear','poly2','poly3','rbf','matern32', ...
                   'matern52','laplacian','rq','mixed'};

opts.pso.swarmSize = 30;
opts.pso.maxIterations = 30;
opts.pso.functionTolerance = 1e-6;
opts.pso.maxStallIterations = 20;
opts.pso.inertiaStart = 0.9;
opts.pso.inertiaEnd = 0.4;
opts.pso.cognitive = 1.5;
opts.pso.social = 1.5;

opts.resultsDirectory = fullfile('results','approved_protocol');
opts.ablationSubdirectory = 'ablation';
opts.ablationFileTag = 'ablation';
opts.saveResults = true;
opts.verbose = true;
end
