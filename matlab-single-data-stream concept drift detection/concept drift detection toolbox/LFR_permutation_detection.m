function [detect_time,data,tpr,tnr,ppv,npv] = LFR_permutation_detection(TrainingData,TestingData,eta,...
    num_least_relearn,mode,table_num,distance_threshold,permutation_window)
% load boundtable BT
if table_num == 1
    [num,txt,raw]= xlsread('BT.xlsx');
elseif table_num == 2
    [num,txt,raw]= xlsread('BT2.xlsx');
elseif table_num == 3
    [num,txt,raw]= xlsread('BT3.xlsx');
elseif table_num == 4
    [num,txt,raw]= xlsread('BT4.xlsx');
end
BT = num(:,2:1000);

if strcmpi(mode,'synthetic') == 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif strcmpi(mode,'real') == 1
    %% Training SVM model
    SVMModel = svmtrain(TrainingData(:,end),TrainingData(:,1:end-1),'-t 2 -c 100 -g 1 -q');
    
    % parameter setting
    iswarning = 0; % boolean variable determine the state of warning
    detect_time = []; % vector to record concept drift time index
    pre_detect_time = 1;
    pre_permutation_time = 1;
    num_wait_steps = 0;
    
    % initialization
    P_ini = 0.5;
    R_ini = 0.5;
    CX = [1,1,;1,1];
    
    % initialize features, true labels and predicted labels
    X = TestingData(:,1:end-1);
    y = TestingData(:,end);
    y_hat = zeros(size(y));
    data = [y,y_hat]';
    
    % define structure of four rates
    tpr = struct('R',zeros(1,length(data)),'P_hat',zeros(1,length(data)),'detect_lb',zeros(1,length(data)),...
        'warn_lb',zeros(1,length(data)),'warn_ub',zeros(1,length(data)),'detect_ub',zeros(1,length(data)));
    tnr = struct('R',zeros(1,length(data)),'P_hat',zeros(1,length(data)),'detect_lb',zeros(1,length(data)),...
        'warn_lb',zeros(1,length(data)),'warn_ub',zeros(1,length(data)),'detect_ub',zeros(1,length(data)));
    ppv = struct('R',zeros(1,length(data)),'P_hat',zeros(1,length(data)),'detect_lb',zeros(1,length(data)),...
        'warn_lb',zeros(1,length(data)),'warn_ub',zeros(1,length(data)),'detect_ub',zeros(1,length(data)));
    npv = struct('R',zeros(1,length(data)),'P_hat',zeros(1,length(data)),'detect_lb',zeros(1,length(data)),...
        'warn_lb',zeros(1,length(data)),'warn_ub',zeros(1,length(data)),'detect_ub',zeros(1,length(data)));
    
    %% initialization for first time index
    data(2,1) = svmpredict(rand(1),X(1,:),SVMModel,'-q');
    if data(1,1)==0 && data(2,1)==0
        tpr.R(1) = R_ini;
        tnr.R(1) = eta(2)*R_ini+(1-eta(2));
        ppv.R(1) = R_ini;
        npv.R(1) = eta(4)*R_ini+(1-eta(4));
        CX(1,1) = CX(1,1)+1;
        tpr.P_hat(1) = P_ini;
        tnr.P_hat(1) = CX(1,1)/(CX(1,1)+CX(2,1));
        ppv.P_hat(1) = P_ini;
        npv.P_hat(1) = CX(1,1)/(CX(1,1)+CX(1,2));
    elseif data(1,1)==1 && data(2,1)==0
        tpr.R(1) = eta(1)*R_ini;
        tnr.R(1) = R_ini;
        ppv.R(1) = R_ini;
        npv.R(1) = eta(4)*R_ini;  
        CX(1,2) = CX(1,2)+1;
        tpr.P_hat(1) = CX(2,2)/(CX(1,2)+CX(2,2));
        tnr.P_hat(1) = P_ini;
        ppv.P_hat(1) = P_ini;
        npv.P_hat(1) = CX(1,1)/(CX(1,1)+CX(1,2));
    elseif data(1,1)==0 && data(2,1)==1
        tpr.R(1) = R_ini;
        tnr.R(1) = eta(2)*R_ini;
        ppv.R(1) = eta(3)*R_ini;
        npv.R(1) = R_ini;
        CX(2,1) = CX(2,1)+1;
        tpr.P_hat(1) = P_ini;
        tnr.P_hat(1) = CX(1,1)/(CX(1,1)+CX(2,1));
        ppv.P_hat(1) = CX(2,2)/(CX(2,1)+CX(2,2));
        npv.P_hat(1) = P_ini;
    elseif data(1,1)==1 && data(2,1)==1
        tpr.R(1) = eta(1)*R_ini+(1-eta(1));
        tnr.R(1) = R_ini;
        ppv.R(1) = eta(3)*R_ini+(1-eta(3));
        npv.R(1) = R_ini;
        CX(2,2) = CX(2,2)+1;
        tpr.P_hat(1) = CX(2,2)/(CX(1,2)+CX(2,2));
        tnr.P_hat(1) = P_ini;
        ppv.P_hat(1) = CX(2,2)/(CX(2,1)+CX(2,2));
        npv.P_hat(1) = P_ini;
    end
    
    %% update warning bound and detection bound
    % update bound for tpr
    [detect_lb,warn_lb,warn_ub,detect_ub] = bound_retrieval(BT,tpr.P_hat(1));
    tpr.detect_lb(1) = detect_lb;
    tpr.warn_lb(1) = warn_lb;
    tpr.warn_ub(1) = warn_ub;
    tpr.detect_ub(1) = detect_ub;
    % update bound for tnr
    [detect_lb,warn_lb,warn_ub,detect_ub] = bound_retrieval(BT,tnr.P_hat(1));
    tnr.detect_lb(1) = detect_lb;
    tnr.warn_lb(1) = warn_lb;
    tnr.warn_ub(1) = warn_ub;
    tnr.detect_ub(1) = detect_ub;
    % update bound for ppv
    [detect_lb,warn_lb,warn_ub,detect_ub] = bound_retrieval(BT,ppv.P_hat(1));
    ppv.detect_lb(1) = detect_lb;
    ppv.warn_lb(1) = warn_lb;
    ppv.warn_ub(1) = warn_ub;
    ppv.detect_ub(1) = detect_ub;
    % update bound for npv
    [detect_lb,warn_lb,warn_ub,detect_ub] = bound_retrieval(BT,npv.P_hat(1));
    npv.detect_lb(1) = detect_lb;
    npv.warn_lb(1) = warn_lb;
    npv.warn_ub(1) = warn_ub;
    npv.detect_ub(1) = detect_ub;
    
%% online processing
for i=2:size(data,2)
    if num_wait_steps ==0
       data(2,i) = svmpredict(rand(1),X(i,:),SVMModel,'-q');
    %% update tpr, tnr, ppv, npv
    if data(1,i)==0 && data(2,i)==0
        tpr.R(i) = tpr.R(i-1);
        tnr.R(i) = eta(2)*tnr.R(i-1)+(1-eta(2));
        ppv.R(i) = ppv.R(i-1);
        npv.R(i) = eta(4)*npv.R(i-1)+(1-eta(4));
        CX(1,1) = CX(1,1)+1;
        tpr.P_hat(i) = tpr.P_hat(i-1);
        tnr.P_hat(i) = CX(1,1)/(CX(1,1)+CX(2,1));
        ppv.P_hat(i) = ppv.P_hat(i-1);
        npv.P_hat(i) = CX(1,1)/(CX(1,1)+CX(1,2));
    elseif data(1,i)==1 && data(2,i)==0
        tpr.R(i) = eta(1)*tpr.R(i-1);
        tnr.R(i) = tnr.R(i-1);
        ppv.R(i) = ppv.R(i-1);
        npv.R(i) = eta(4)*npv.R(i-1);  
        CX(1,2) = CX(1,2)+1;
        tpr.P_hat(i) = CX(2,2)/(CX(1,2)+CX(2,2));
        tnr.P_hat(i) = tnr.P_hat(i-1);
        ppv.P_hat(i) = ppv.P_hat(i-1);
        npv.P_hat(i) = CX(1,1)/(CX(1,1)+CX(1,2));
    elseif data(1,i)==0 && data(2,i)==1
        tpr.R(i) = tpr.R(i-1);
        tnr.R(i) = eta(2)*tnr.R(i-1);
        ppv.R(i) = eta(3)*ppv.R(i-1);
        npv.R(i) = npv.R(i-1);
        CX(2,1) = CX(2,1)+1;
        tpr.P_hat(i) = tpr.P_hat(i-1);
        tnr.P_hat(i) = CX(1,1)/(CX(1,1)+CX(2,1));
        ppv.P_hat(i) = CX(2,2)/(CX(2,1)+CX(2,2));
        npv.P_hat(i) = npv.P_hat(i-1);
    elseif data(1,i)==1 && data(2,i)==1
        tpr.R(i) = eta(1)*tpr.R(i-1)+(1-eta(1));
        tnr.R(i) = tnr.R(i-1);
        ppv.R(i) = eta(3)*ppv.R(i-1)+(1-eta(3));
        npv.R(i) = npv.R(i-1);
        CX(2,2) = CX(2,2)+1;
        tpr.P_hat(i) = CX(2,2)/(CX(1,2)+CX(2,2));
        tnr.P_hat(i) = tnr.P_hat(i-1);
        ppv.P_hat(i) = CX(2,2)/(CX(2,1)+CX(2,2));
        npv.P_hat(i) = npv.P_hat(i-1);
    end
    
    %% update warning bound and detection bound
    % update bound for tpr
    [detect_lb,warn_lb,warn_ub,detect_ub] = bound_retrieval(BT,tpr.P_hat(i));
    tpr.detect_lb(i) = detect_lb;
    tpr.warn_lb(i) = warn_lb;
    tpr.warn_ub(i) = warn_ub;
    tpr.detect_ub(i) = detect_ub;
    % update bound for tnr
    [detect_lb,warn_lb,warn_ub,detect_ub] = bound_retrieval(BT,tnr.P_hat(i));
    tnr.detect_lb(i) = detect_lb;
    tnr.warn_lb(i) = warn_lb;
    tnr.warn_ub(i) = warn_ub;
    tnr.detect_ub(i) = detect_ub;
    % update bound for ppv
    [detect_lb,warn_lb,warn_ub,detect_ub] = bound_retrieval(BT,ppv.P_hat(i));
    ppv.detect_lb(i) = detect_lb;
    ppv.warn_lb(i) = warn_lb;
    ppv.warn_ub(i) = warn_ub;
    ppv.detect_ub(i) = detect_ub;
    % update bound for npv
    [detect_lb,warn_lb,warn_ub,detect_ub] = bound_retrieval(BT,npv.P_hat(i));
    npv.detect_lb(i) = detect_lb;
    npv.warn_lb(i) = warn_lb;
    npv.warn_ub(i) = warn_ub;
    npv.detect_ub(i) = detect_ub;
    
    %% determine warnining or alarming
    % check if warning starts
    warn_tpr = (tpr.R(i)<tpr.warn_lb(i))||(tpr.R(i)>tpr.warn_ub(i));
    warn_tnr = (tnr.R(i)<tnr.warn_lb(i))||(tnr.R(i)>tnr.warn_ub(i));
    warn_ppv = (ppv.R(i)<ppv.warn_lb(i))||(ppv.R(i)>ppv.warn_ub(i));
    warn_npv = (npv.R(i)<npv.warn_lb(i))||(npv.R(i)>npv.warn_ub(i));
    if (warn_tpr+warn_tnr+warn_ppv+warn_npv>0) && (iswarning==0)
        warn_time = i;
        iswarning = 1;
    elseif (warn_tpr+warn_tnr+warn_ppv+warn_npv==0) && (iswarning==1)
        iswarning = 0;
    end
    
    % check if potential detection found
    detect_tpr = ((tpr.R(i)<tpr.detect_lb(i))||(tpr.R(i)>tpr.detect_ub(i)))...
        && (tpr.R(i)<0.99) && (tpr.R(i)>0.01);
    detect_tnr = ((tnr.R(i)<tnr.detect_lb(i))||(tnr.R(i)>tnr.detect_ub(i)))...
        && (tnr.R(i)<0.99) && (tnr.R(i)>0.01);
    detect_ppv = ((ppv.R(i)<ppv.detect_lb(i))||(ppv.R(i)>ppv.detect_ub(i)))...
        && (ppv.R(i)<0.99) && (ppv.R(i)>0.01);
    detect_npv = ((npv.R(i)<npv.detect_lb(i))||(npv.R(i)>npv.detect_ub(i)))...
        && (npv.R(i)<0.99) && (npv.R(i)>0.01);
    if (detect_tpr+detect_tnr+detect_ppv+detect_npv>0) && (i-pre_detect_time>distance_threshold) ...
        && (i >= permutation_window) && (i <= size(data,2)-permutation_window)
        
        % confirm potential alarm with permutation test
        if i-pre_permutation_time >= distance_threshold
        fprintf('Permutation test for potential point %2.0f \n',i);
        PermutationData = TestingData(i-permutation_window+1:i+permutation_window,:);
        test_value = permutation_test(PermutationData,permutation_window,1000,0,0.2);
        fprintf('Permutation test for potential point %2.0f is %1.0f \n',i,test_value);
        end
        
        pre_permutation_time = i;
        
        if test_value == 1
        detect_time = [detect_time i];
        pre_detect_time = i;
        iswarning = 0;
        
        % relearn the classifier
        if (i-warn_time) >= num_least_relearn
            SVMModel = svmtrain(y(warn_time:i),X(warn_time:i,:),'-t 2 -c 100 -g 1 -q');
        % reset linear four rates
        tpr.R(i)=R_ini; 
        tpr.P_hat(i)=P_ini; 
        tpr.detect_lb(i)=0;
        tpr.detect_ub(i)=1;
        tpr.warn_lb(i)=0;
        tpr.warn_ub(i)=1;
        
        tnr.R(i)=R_ini; 
        tnr.P_hat(i)=P_ini; 
        tnr.detect_lb(i)=0;
        tnr.detect_ub(i)=1;
        tnr.warn_lb(i)=0;
        tnr.warn_ub(i)=1;
        
        ppv.R(i)=R_ini; 
        ppv.P_hat(i)=P_ini; 
        ppv.detect_lb(i)=0;
        ppv.detect_ub(i)=1;
        ppv.warn_lb(i)=0;
        ppv.warn_ub(i)=1;
        
        npv.R(i)=R_ini; 
        npv.P_hat(i)=P_ini; 
        npv.detect_lb(i)=0;
        npv.detect_ub(i)=1;
        npv.warn_lb(i)=0;
        npv.warn_ub(i)=1;
        
        CX = [1,1;1,1];
        
        elseif (i-warn_time) < num_least_relearn && (warn_time + num_least_relearn) <= size(data,2)
            SVMModel = svmtrain(y(warn_time:(warn_time+num_least_relearn)),...
                X(warn_time:(warn_time+num_least_relearn),:),...
                '-t 2 -c 100 -g 1 -q');
            num_wait_steps = warn_time + num_least_relearn - i;
        elseif (i-warn_time) < num_least_relearn && (warn_time + num_least_relearn) > size(data,2)
            SVMModel = svmtrain(y(warn_time:end),X((warn_time:end),:),...
                '-t 2 -c 100 -g 1 -q');
            num_wait_steps = size(data,2) - i;
        end
        end
          
    end
    
    else
        num_wait_steps = num_wait_steps - 1;
        % reset linear four rates
        tpr.R(i)=R_ini; 
        tpr.P_hat(i)=P_ini; 
        tpr.detect_lb(i)=0;
        tpr.detect_ub(i)=1;
        tpr.warn_lb(i)=0;
        tpr.warn_ub(i)=1;
        
        tnr.R(i)=R_ini; 
        tnr.P_hat(i)=P_ini; 
        tnr.detect_lb(i)=0;
        tnr.detect_ub(i)=1;
        tnr.warn_lb(i)=0;
        tnr.warn_ub(i)=1;
        
        ppv.R(i)=R_ini; 
        ppv.P_hat(i)=P_ini; 
        ppv.detect_lb(i)=0;
        ppv.detect_ub(i)=1;
        ppv.warn_lb(i)=0;
        ppv.warn_ub(i)=1;
        
        npv.R(i)=R_ini; 
        npv.P_hat(i)=P_ini; 
        npv.detect_lb(i)=0;
        npv.detect_ub(i)=1;
        npv.warn_lb(i)=0;
        npv.warn_ub(i)=1;
        
        CX = [1,1;1,1];
        
        if num_wait_steps == 0
            pre_detect_time = i;
        end
        
    end
end    
    
end

end