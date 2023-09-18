function [up,down]=computeIntesection_MultiDim(intersection_interval_UP,intersection_interval_DWN,interval_UP,interval_DWN)
% intersection_interval_UP,
% intersection_interval_DWN,
% interval_UP,
% interval_DWN
%
% are ALL colum vectors
% 
% the output will two column vectors indicating the upper and lower bound
% of intersection
%
% Giacomo Boracchi
% Politecnico di Milano
% January 2010
% giacomo.boracchi@polimi.it
%

for tt=1:size(intersection_interval_DWN,3)
    
    up(:,:,tt)=min([intersection_interval_UP(:,:,tt),interval_UP(:,:,tt)],[],2);
    down(:,:,tt)=max([intersection_interval_DWN(:,:,tt),interval_DWN(:,:,tt)],[],2);

end

    