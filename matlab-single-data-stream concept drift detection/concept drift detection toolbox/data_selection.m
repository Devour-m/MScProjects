function [data_new, change_points_new] = data_selection(data, change_points, num_per_concept)

data_new = [];
change_points_new = [];

num_concept = length(change_points)+1;
for i = 1:num_concept
    if i == 1
    loc{i} = randsample(1:change_points(i),num_per_concept);
    elseif i>1 && i<num_concept
        loc{i} = randsample(change_points(i-1):change_points(i),num_per_concept);
    else
        loc{i} = randsample(change_points(i-1):length(data),num_per_concept);
    end
end

for i = 1:num_concept
    data_new = [data_new; data(loc{i},:)];
end

for i = 1:num_concept-1
change_points_new = [change_points_new, num_per_concept*i];
end

end