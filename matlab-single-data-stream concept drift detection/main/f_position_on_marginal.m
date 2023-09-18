function [newDetected] = f_position_on_marginal(qfull, qind, thresh)

% find the position of the newly detected change point on the marginal
% stream 
% qfull: marginal data stream of dimension nFeature * nSample
% qind: infdividual data sample of dimension 1*nFeature

position = find(ismember(qfull', qind,'rows'));

position (position < thresh) = inf;
[newDetected, ~] = min(position(position > thresh));
