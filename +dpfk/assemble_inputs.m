function X = assemble_inputs(Xexo, yLag, useTargetLag)
%ASSEMBLE_INPUTS Add the previous target only when the protocol enables it.

if nargin < 3 || useTargetLag
    X = [Xexo, yLag];
else
    X = Xexo;
end
end
