function data = load_dataset(config, baseDirectory)
%LOAD_DATASET Load one configured dataset without changing row order.

path = fullfile(baseDirectory, config.file);
if ~isfile(path)
    error('dpfk:MissingDataset', 'Dataset not found: %s', path);
end

if strcmpi(config.name, 'Wan')
    raw = readmatrix(path, 'FileType', 'text', 'NumHeaderLines', 13, ...
        'Delimiter', ',');
    variableNames = {'Max_temperature','Min_temperature','Dewpoint', ...
        'Precipitation','Sea_level_pressure','Standard_pressure', ...
        'Visibility','Wind_speed','Max_wind_speed','Mean_temperature'};
elseif strcmpi(config.name, 'DFO')
    tableData = readtable(path, 'Delimiter', ';', ...
        'VariableNamingRule', 'preserve');
    raw = table2array(tableData);
    variableNames = cellstr(string(tableData.Properties.VariableNames));
else
    tableData = readtable(path, 'VariableNamingRule', 'preserve');
    variableNames = cellstr(string(tableData.Properties.VariableNames));
    requiredIndices = unique([config.exogenousIndices, config.targetIndex]);
    raw = nan(height(tableData), width(tableData));
    for columnIndex = requiredIndices
        values = tableData{:, columnIndex};
        if ~isnumeric(values) && ~islogical(values)
            error('dpfk:NonNumericVariable', ...
                '%s requires numeric variable %s (column %d).', ...
                config.name, variableNames{columnIndex}, columnIndex);
        end
        raw(:, columnIndex) = double(values);
    end
end

raw = double(raw);
if ~isempty(config.missingSentinel)
    raw(raw == config.missingSentinel) = NaN;
end
if size(raw, 2) < config.targetIndex
    error('dpfk:InvalidDataset', ...
        '%s has %d columns but targetIndex is %d.', ...
        config.name, size(raw, 2), config.targetIndex);
end

data.name = config.name;
data.raw = raw;
data.variableNames = variableNames;
data.targetIndex = config.targetIndex;
data.targetName = config.targetName;
data.exogenousIndices = config.exogenousIndices;
data.exogenousNames = variableNames(config.exogenousIndices);
data.excludedLeakageIndices = config.excludedLeakageIndices;
data.originalRowCount = size(raw, 1);
end
