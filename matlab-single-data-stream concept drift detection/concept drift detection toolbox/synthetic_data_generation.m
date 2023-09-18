function data = synthetic_data_generation(num_data,CX)
%% this function is used to generate synthetic data dat = [y;y_hat] 
% with confusion probability matrix CX

%%
pd = [CX(1,1),CX(1,2),CX(2,1),CX(2,2)]; % probablity distribution

% random sample data from 1:4 with probability weights pd
sample = datasample(1:4,num_data,'Weights',pd,'Replace',true);

%%
data = zeros(2,num_data);
for i=1:num_data
    if sample(i)==1;
        data(:,i)=[0;0];
    elseif sample(i)==2;
            data(:,i)=[1;0];
    elseif sample(i)==3;
        data(:,i)=[0;1];
    else sample(i)==4;
        data(:,i)=[1;1];
    end
end

end