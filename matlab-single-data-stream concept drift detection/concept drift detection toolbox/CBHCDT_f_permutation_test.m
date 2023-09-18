function test_value = CBHCDT_f_permutation_test(PermutationData,window_size,P,delta,sig_level, classifier)

%% obtain the groud truth error
% SVMModel = svmtrain(PermutationData(1:window_size,end),PermutationData(1:window_size,1:end-1),...
%    '-t 2 -c 100 -g 1 -q');
  switch classifier 
        case 'SVM_linear'
            Mdl = fitcsvm(PermutationData(1:window_size,1:end-1), PermutationData(1:window_size,end), 'KernelFunction','linear');
        case 'SVM_rbf'
            Mdl = fitcsvm(PermutationData(1:window_size,1:end-1), PermutationData(1:window_size,end), 'KernelFunction','rbf');
        case 'Tree'
            Mdl = fitctree(PermutationData(1:window_size,1:end-1),PermutationData(1:window_size,end));
    end

res = predict(Mdl,PermutationData(window_size+1:end,1:end-1));
error_ord = sum(res ~= PermutationData(window_size+1:end,end));

%[predict_labels,accuracy,dec_values] = svmpredict...
%    (PermutationData(window_size+1:end,end),PermutationData(window_size+1:end,1:end-1),SVMModel,'-q');
% error_ord = 1 - accuracy(1);

%% conducting permutation test
for i=1:P
    if mod(i,200)==0
        fprintf('Permutation test after %2.0f iterations \n',i);
    end
% generate new training and testing dataset
index = randperm(size(PermutationData,1));
TrainData = PermutationData(index(1:window_size),:);
TestData = PermutationData(index(window_size+1:end),:);

% train new classifier and calculate error
% SVMModel = svmtrain(TrainData(:,end),TrainData(:,1:end-1),...
%     '-t 2 -c 100 -g 1 -q');
switch classifier
    case 'SVM_linear'
        Mdl = fitcsvm(TrainData(:,1:end-1), TrainData(:,end), 'KernelFunction','linear');
    case 'SVM_rbf'
        Mdl = fitcsvm(TrainData(:,1:end-1), TrainData(:,end), 'KernelFunction','rbf');
    case 'Tree'
        Mdl = fitctree(TrainData(:,1:end-1),TrainData(:,end));
end

res = predict(Mdl,TestData(:,1:end-1));
error_permutation(i) = sum(res ~= TestData(:,end));

%[predict_labels,accuracy,dec_values] = svmpredict...
%    (TestData(:,end),TestData(:,1:end-1),SVMModel,'-q');
% error_permutation(i) = 1 - accuracy(1);
end

%% determine true concept or not
sumE = 0;
for i=1:P
    sumE = sumE + (error_ord<=error_permutation(i));
end

if ((1+sumE)/(1+P)) <= sig_level
    test_value = 1;
else
    test_value = 0;
end

end