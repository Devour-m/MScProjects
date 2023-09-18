function [result , p_val] = HotellingTSquareTest(X1 , X2 , delta , alphaHT)
%
% function [result , p_val] = HotellingTSquareTest(X1 , X2 , delta , alphaHT)
% 
% Hotelling T^2 test to check H0: "E[X1] - E[X2] == delta"
%
% Giacomo Boracchi
% Politecnico di Milano
% giacomo.boracchi@polimi.it
% January 2011


%
% population cardinality
n1 = size(X1 , 2); % number of features within the training set (T_0) / nu
n2 = size(X2 , 2); % number of features after the change in the novel state (t_hat - t_ref)/nu

p = size(X1 , 1);

% when the change has been detected on the first feature (the mean), my population is
%                  X1 = \{[MM1(1 , 1) ; ... ; MM1(p , 1)] , [MM1(1 , 2) ; ... ; MM1(p , 2)] , [MM1(1 , n_1) ; ... ; MM1(p , n_1)] \}
%                  X2 = \{[MM2(1 , 1) ; ... ; MM2(p , 1)] , [MM2(1 , 2) ; ... ; MM2(p , 2)] , [MM2(1 , n_2) ; ... ; MM2(p , n_2)] \}

% poulations are normal under H_0 they have same mean and same variance

% compute population mean before and after the (suspected) change
X_bar1 =  mean(X1 , 2);
X_bar2 =  mean(X2 , 2);

% compute S_pooled
Sigma1 = cov(X1');
Sigma2 = cov(X2');

%
S_pooled = ((n1 - 1) * Sigma1 + (n2 - 1) * Sigma2) / (n1 + n2 - 2);

% compute test Statistic (T^2 Hotelling)
T_stat = (X_bar1 - X_bar2 - delta)' * inv((1/n1 + 1/n2) * S_pooled) * (X_bar1 - X_bar2 - delta);

c2 = ((n1 + n2 - 2) * p) / (n1 + n2 - p - 1) * finv(1 - alphaHT  , p , (n1 + n2 - p - 1));

result = 1 - (T_stat <= c2); % this hold with probability 1 - params.alpha when H_0 holds
p_val = 1 - fcdf(c2 ,  p , (n1 + n2 - p - 1));