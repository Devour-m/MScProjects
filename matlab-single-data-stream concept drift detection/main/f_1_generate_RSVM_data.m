function [RSVMdat, normphi] = f_1_generate_RSVM_data (ker, ds, l, dstrain, ltrain, ntrain, numF, thresh)
%% Oct 2018 Shuyi Zhang
% ker      kernel 'gaussian' or 'linear'
% dstrain  the data set of which the training data set (used to train the
%          normal vectors) is part of it
% ds       the full data set without label dimension that we want the projection to be on nFeature *nSample
% l        full label vector



train_matrix = dstrain(:,1:ntrain)';
train_label = ltrain(1:ntrain)';




% check if all labels are in the form of -1 or 1
if any(train_label) == 1
    train_label(train_label ==0) = -1;
end



RSVMdat = zeros(size(ds',1),1);
alpha = zeros(ntrain,1);
normphi = zeros(size(train_matrix));
nx = zeros(size(train_matrix));
w = zeros(size(train_matrix,2),numF);
j = 1;

switch ker
    case 'linear'
        % multi-dimensional projection
        
        while (j < numF+1)
            % train SVM on training data set
            %{
        svmStruct= svmtrain(train_matrix,num2str(train_label),'kernel_function','linear');
        % extract and calculate the projection vector (normal vector)
        w(:,j) = svmStruct.Alpha' * svmStruct.SupportVectors;
        bias = svmStruct.Bias;
            %}
            % train SVM on training data set
            svmStruct = fitcsvm(train_matrix,num2str(train_label), 'KernelFunction','linear');
            % svmStruct= svmtrain(train_matrix,num2str(train_label),'kernel_function','linear');
            % extract and calculate the projection vector (normal vector)
            SVs = train_matrix(svmStruct.IsSupportVector,:);
            w(:,j) = svmStruct.Alpha' * SVs;
            bias = svmStruct.Bias;
            
            % normalize w
            w(:,j) = w(:,j)/sqrt(sum(w(:,j).^2));
            
            % recursive SVM update
            for dn = 1:ntrain
                nx(dn,:) = train_matrix(dn,:) - sum(train_matrix(dn,:) .* w(:,j)')* w(:,j)';
            end
            
            normphi (:,j) = sum(nx'.^2);
            train_matrix = nx;
            if max(normphi(:,j))>= thresh
                j = j+1;
            else
                j =  numF+1;
            end
        end
        
        RSVMdat = ds' * w;
        
    case 'gaussian'
        % 1-D projection only
        % 2-D or more requires quadratic programming tool instead of just
        % fitcsvm
        
        % choose sigma
        sigma = 1;
        while (j < numF+1)
            % train SVM on training data set
            
            svmStruct = fitcsvm(train_matrix,num2str(train_label), 'KernelFunction','RBF','KernelScale', 1/sqrt(sigma), 'Standardize', true);
            svInd = find(svmStruct.IsSupportVector);
            
            for i = 1:length(svmStruct.Alpha)
                alpha(svInd(i)) = svmStruct.Alpha(i);
            end
            
            %{
         % train SVM on training data set
        svmStruct= svmtrain(train_matrix,num2str(train_label),'kernel_function','RBF');
        alpha = svmStruct.Alpha
            %}
            % extract alphas and calculate projected data as
            
            for dn = 1:size(ds,2)
                RSVMdat(dn,:) = sum(alpha .* exp(-sum((train_matrix(1:ntrain, :) - repmat(ds(:,dn),1,ntrain)') .^ 2,2) / (2 * sigma * sigma)));
            end
            
            normphi (:,j) = norm(RSVMdat(:,j),2);
            train_matrix = RSVMdat;
            if max(normphi(:,j))>= thresh
                j = j+1;
            else
                break
            end
        end
        
        
        
end

RSVMdat = [RSVMdat l']';


end
