function [pval, h_ci, sa, sb, ci_a, ci_b, surrog_mean_diff] = myboot(a, b, varargin)
if length(varargin)>=1
    nloop = varargin{1};
else
    nloop = 2000;
end
if length(varargin)>=2
    alpha = varargin{2};
else
    alpha = 0.05;
end
if length(varargin)>=3
    f = varargin{3};
else
    f = @mean;
    % f = @median;
end


ori_mean_diff = f(a)-f(b);
all_mean = f([a;b]);
null_a = a-f(a)+all_mean;
null_b = b-f(b)+all_mean;
surrog_mean_diff = zeros(1,nloop);
sa = zeros(1,nloop);
sb = zeros(1,nloop);

for i = 1:nloop
    sa(i) = f(a(randi(length(a),length(a),1)));
    sb(i) = f(b(randi(length(b),length(b),1)));
    surrog_mean_diff(i) = f(null_a(randi(length(a),length(a),1)))-f(null_b(randi(length(b),length(b),1)));
end
ci_a = quantile(sa,[alpha, 1-alpha]);
ci_b = quantile(sb,[alpha, 1-alpha]);
h_ci = ci_a(2)<ci_b(1) | ci_a(1)>ci_b(2);
pval = sum(surrog_mean_diff > abs(ori_mean_diff) | surrog_mean_diff <= -abs(ori_mean_diff))/nloop;
end