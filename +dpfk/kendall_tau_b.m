function tau = kendall_tau_b(x, y)
%KENDALL_TAU_B Tie-adjusted Kendall rank correlation without toolboxes.

x = x(:);
y = y(:);
n = numel(x);
numerator = 0;
nonTiedX = 0;
nonTiedY = 0;
for i = 1:(n - 1)
    sx = sign(x(i) - x((i + 1):n));
    sy = sign(y(i) - y((i + 1):n));
    numerator = numerator + sum(sx .* sy);
    nonTiedX = nonTiedX + sum(sx ~= 0);
    nonTiedY = nonTiedY + sum(sy ~= 0);
end
denominator = sqrt(nonTiedX * nonTiedY);
if denominator == 0
    tau = 0;
else
    tau = numerator / denominator;
end
end
