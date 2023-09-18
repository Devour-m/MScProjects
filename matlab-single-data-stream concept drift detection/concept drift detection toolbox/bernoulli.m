function y = bernoulli(n,p)
%% this function generate bernoulli trails with parameter n and p
%%
x = rand(1,n);
y = (x<=p);

end