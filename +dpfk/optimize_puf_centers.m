function result = optimize_puf_centers(X, clusterCount, fuzzifier, pso, seed)
%OPTIMIZE_PUF_CENTERS Fixed-budget internal PSO for PUF centers only.

rng(seed, 'twister');
searchTimer = tic;
[n, d] = size(X);
particleCount = pso.swarmSize;
dimension = clusterCount * d;
lower = repmat(min(X, [], 1), 1, clusterCount);
upper = repmat(max(X, [], 1), 1, clusterCount);
span = upper - lower;

position = lower + rand(particleCount, dimension) .* span;
if n >= clusterCount
    anchor = round(linspace(1, n, clusterCount));
    position(1, :) = reshape(X(anchor, :)', 1, []);
end
velocity = zeros(particleCount, dimension);
personalBest = position;
personalValue = inf(particleCount, 1);

for i = 1:particleCount
    centers = reshape(position(i, :), d, clusterCount)';
    personalValue(i) = dpfk.puf_objective(X, centers, fuzzifier);
end
[globalValue, bestIndex] = min(personalValue);
globalBest = personalBest(bestIndex, :);
history = nan(pso.maxIterations, 1);
stall = 0;

for iteration = 1:pso.maxIterations
    previousBest = globalValue;
    fraction = (iteration - 1) / max(pso.maxIterations - 1, 1);
    inertia = pso.inertiaStart + fraction * ...
        (pso.inertiaEnd - pso.inertiaStart);
    for i = 1:particleCount
        r1 = rand(1, dimension);
        r2 = rand(1, dimension);
        velocity(i, :) = inertia * velocity(i, :) + ...
            pso.cognitive * r1 .* (personalBest(i, :) - position(i, :)) + ...
            pso.social * r2 .* (globalBest - position(i, :));
        position(i, :) = min(max(position(i, :) + velocity(i, :), lower), upper);
        centers = reshape(position(i, :), d, clusterCount)';
        value = dpfk.puf_objective(X, centers, fuzzifier);
        if value < personalValue(i)
            personalValue(i) = value;
            personalBest(i, :) = position(i, :);
        end
        if value < globalValue
            globalValue = value;
            globalBest = position(i, :);
        end
    end
    history(iteration) = globalValue;
    if previousBest - globalValue > pso.functionTolerance
        stall = 0;
    else
        stall = stall + 1;
    end
    if stall >= pso.maxStallIterations
        history = history(1:iteration);
        break;
    end
end

result.centers = reshape(globalBest, d, clusterCount)';
result.objective = globalValue;
result.history = history;
result.seed = seed;
result.iterations = numel(history);
result.functionEvaluations = particleCount * (1 + result.iterations);
result.runtime = toc(searchTimer);
end
