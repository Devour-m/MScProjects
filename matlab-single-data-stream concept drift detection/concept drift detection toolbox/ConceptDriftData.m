% CCONCEPTDRIFTDATA: Generate Data Streams with Concept Drift
%   [xTrain,cTrain,xTest,cTest] = conceptDriftData(type,T,N)
% 
%   This m-file generates data containing concept drift and
%   classes can be under-sampled to create an imbalanced 
%   learning scenario. 
%   
%   INPUT
%     TYPE: Dataset desired
%       'abrupt': . 
%       'abrupt2': 
%       'incremental': 
%       'recurrence': 
%       'gradual': 
%       'opposite': 
%     d: Number of feature dimensions (positive odd number)
%     N: Number of instances to generate in each concept
%     n: Noise scale
%   OUTPUT
%     data = [X,y];
%     X: matrix containing the feature vectors in
%       column format for training/testing.
%     y: array containing the instance labels in
%       corresponding to X. 
%
%
function data = ConceptDriftData(type,d,N,n)
  % switch on the type of experiment
  switch type
    case 'abrupt'
        threshold = [-2,0,2,-1,1,3];
        c = length(threshold);
        y_threshold = [];
        for i = 1:c
            y_threshold = [y_threshold; threshold(i)*ones(N,1)];
        end 
        
        X = randn(N*c,d);
        for i = 1:(d-1)/2
           sum = X(:,i).*X(:,i+1);
        end
        sum = sum + n*X(:,end);
        y = (sum>y_threshold);
        X = X(:,1:end-1);
        data = [X,y];

    case 'incremental'
        threshold = [-3,-2,-1,0,1,2];
        c = length(threshold);
        y_threshold = [];
        for i = 1:c
            y_threshold = [y_threshold; threshold(i)*ones(N,1)];
        end 
        
        X = randn(N*c,d);
        for i = 1:(d-1)/2
           sum = X(:,i).*X(:,i+1);
        end
        sum = sum + n*X(:,end);
        y = (sum>y_threshold);
        X = X(:,1:end-1);
        data = [X,y];

    case 'recurrence'
        threshold = [-1,1,-1,1,-1,1];
        c = length(threshold);
        y_threshold = [];
        for i = 1:c
            y_threshold = [y_threshold; threshold(i)*ones(N,1)];
        end 
        
        X = randn(N*c,d);
        for i = 1:(d-1)/2
           sum = X(:,i).*X(:,i+1);
        end
        sum = sum + n*X(:,end);
        y = (sum>y_threshold);
        X = X(:,1:end-1);
        data = [X,y];

    case 'gradual'
        threshold = [-2,0,2,-1,1,3];
        c = length(threshold);
        y_threshold = [];
        for i = 1:c
            y_threshold = [y_threshold; threshold(i)*ones(N,1)];
        end 
        
        X = randn(N*c,d);
        for i = 1:(d-1)/2
           sum = X(:,i).*X(:,i+1);
        end
        sum = sum + n*X(:,end);
        y = (sum>y_threshold);
        
        data = [X(:,1:end-1),y];
        data_new = data;
        
        overlap_N = round(N/5);
        
        for i = 1:c-1
            rowrank = randperm(2*overlap_N+1)';
            data_new(i*N-overlap_N:i*N+overlap_N,:) = data(rowrank+(i*N-overlap_N-1),:);
        end
        
        X = data_new(:,1:end-1);
        y = data_new(:,end);
        data = [X,y];

    case 'opposite'
        threshold = [0,0,1,1,2,2];
        c = length(threshold);
        X = randn(N*c,d);
        y = zeros(N*c,1);
        
        for i = 1:c
            y_threshold = threshold(i)*ones(N,1);
            for j = 1:(d-1)/2
                sum = X((i-1)*N+1:(i-1)*N+N,j).*X((i-1)*N+1:(i-1)*N+N,j+1);
            end
            sum = sum + n*X((i-1)*N+1:(i-1)*N+N,end);
            
            if mod(i,2)==0
                y((i-1)*N+1:(i-1)*N+N) = (sum<y_threshold);
            else
                y((i-1)*N+1:(i-1)*N+N) = (sum>y_threshold);
            end
            
        end
        
        X = X(:,1:end-1);
        data = [X,y];
        
    otherwise
      error('ERROR::conceptDriftData.m: Unknow Dataset Selected');
  end
end