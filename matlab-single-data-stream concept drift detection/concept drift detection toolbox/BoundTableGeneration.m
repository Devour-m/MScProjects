function BT = BoundTableGeneration(sig_level,num_time,num_MC)
%% this function is used to generate boundtable 
% given significant level sig_level (vector with four elements),
% number of time steps num_time (scale) and number of Monte Carlo
% simulations num_MC (scale)

% an example of sig_level: sig_level = [1/100000,1/100,99/100,1-1/100000];

%% 
p_hat = linspace(0.001,0.999,999); 
% eta_hat = linspace(0.801,0.999,199);
eta_hat = 0.9;
%%
BT = zeros(4,length(p_hat),length(eta_hat)); % initialize boundtable BT
for i = 1:length(p_hat)
    i
    for j = 1:length(eta_hat)
        % generate empirical distribution of F_hat(R)
        R = zeros(1,num_MC);
        for k = 1:num_MC
            I = bernoulli(num_time,p_hat(i));
            R(k) = (1-eta_hat(j))*sum(I.*eta_hat(j).^[num_time-1:-1:0]);
        end
        
        % generate lower and upper bound
        BT(1,i,j) = quantile(R,sig_level(1)); % lower detect bound
        BT(2,i,j) = quantile(R,sig_level(2)); % lower warning bound
        BT(3,i,j) = quantile(R,sig_level(3)); % upper warning bound
        BT(4,i,j) = quantile(R,sig_level(4)); % upper detect bound
    end
end

end