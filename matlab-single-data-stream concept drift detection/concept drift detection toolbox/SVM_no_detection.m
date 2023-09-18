function data = SVM_no_detection(TrainingData,TestingData,mode)

if strcmpi(mode,'synthetic') == 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif strcmpi(mode,'real') == 1
    %% Training SVM model
    SVMModel = svmtrain(TrainingData(:,end),TrainingData(:,1:end-1),'-t 0 -c 100 -g 1 -q');
    
    % initialize features, true labels and predicted labels
    X = TestingData(:,1:end-1);
    y = TestingData(:,end);
    y_hat = zeros(size(y));
    data = [y,y_hat]';
    
%% online processing
for i=1:size(data,2)
       data(2,i) = svmpredict(rand(1),X(i,:),SVMModel,'-q');
end    
    
end

end