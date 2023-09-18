function [detect_time,data,error_rate] = DDM_detection(TrainingData,TestingData,warn_coeff,...
    detect_coeff,num_least_relearn,mode,distance_threshold)
%%
if strcmpi(mode,'synthetic') == 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%    
elseif strcmpi(mode,'real') == 1
    % initialization
    detect_time = [];
    pre_detect_time = 1;
    iswarning=0; % boolean variable determin is warning or not
    num_wait_steps = 0;
    
    error_rate = struct('p',zeros(1,length(TestingData)),'s',zeros(1,length(TestingData)),...
        'p_min',zeros(1,length(TestingData)),'s_min',zeros(1,length(TestingData)),...
        'warn_ub',zeros(1,length(TestingData)),'detect_ub',zeros(1,length(TestingData)));
    
    % Training SVM or DT model
    SVMModel = svmtrain(TrainingData(:,end),TrainingData(:,1:end-1),...
        '-t 2 -c 100 -g 1 -q');
%     DTModel = classregtree(TrainingData(:,1:end-1),TrainingData(:,end));

    p_ini = 0;
    s_ini = 0;
    p_min_ini = 1;
    s_min_ini = 0;
    
    % initialize features, true labels and predicted labels
    X = TestingData(:,1:end-1);
    y = TestingData(:,end);
    y_hat = zeros(size(y));
    data = [y,y_hat]';
    
    % initialization for first time index
    data(2,1) = svmpredict(rand(1),X(1,:),SVMModel,'-q');
%     data(2,1) = eval(DTModel,X(1,:));
    if data(1,1) == data(2,1)
        error_rate.p(1) = 0;
        error_rate.s(1) = 0;
        error_rate.p_min(1) = 1;
        error_rate.s_min(1) = 0;
    elseif data(1,1) ~= data(2,1)
        error_rate.p(1) = 1;
        error_rate.s(1) = 0;
        error_rate.p_min(1) = 1;
        error_rate.s_min(1) = 0;
    end
    
    % concept drift detection
    for i=2:size(data,2)
        if num_wait_steps ==0
            data(2,i) = svmpredict(rand(1),X(i,:),SVMModel,'-q');
%             data(2,i) = eval(DTModel,X(i,:));
            %% update error_rate
            if data(1,i) == data(2,i)
                error_rate.p(i) = (error_rate.p(i-1)*(i-1-pre_detect_time))/(i-pre_detect_time);
            elseif data(1,i) ~= data(2,i)
                error_rate.p(i) = (error_rate.p(i-1)*(i-1-pre_detect_time)+1)/(i-pre_detect_time);
            end
            
            error_rate.s(i) = sqrt((error_rate.p(i)*(1-error_rate.p(i)))/(i-pre_detect_time));
     
            if ((error_rate.p(i)+error_rate.s(i))<(error_rate.p_min(i-1)+error_rate.s_min(i-1))) &&...
                    ((i-pre_detect_time)>distance_threshold)
                error_rate.p_min(i) = error_rate.p(i);
                error_rate.s_min(i) = error_rate.s(i);
            else
                error_rate.p_min(i) = error_rate.p_min(i-1);
                error_rate.s_min(i) = error_rate.s_min(i-1);
            end
            
            %% update warning bound and detection bound
            error_rate.detect_ub(i) = error_rate.p_min(i) + detect_coeff*error_rate.s_min(i);
            error_rate.warn_ub(i) = error_rate.p_min(i) + warn_coeff*error_rate.s_min(i);
            
            %% determine warnining or alarming
            % check if warning starts
            if ((error_rate.p(i)+error_rate.s(i))>=error_rate.warn_ub(i)) && (iswarning==0)...
                    && ((i-pre_detect_time)>distance_threshold)
                warn_time = i;
                iswarning = 1;

            elseif ((error_rate.p(i)+error_rate.s(i))<error_rate.warn_ub(i)) && (iswarning==1)
                iswarning = 0;
            end
            
            % check if detection found
            if ((error_rate.p(i)+error_rate.s(i))>=error_rate.detect_ub(i)) && ...
                    ((i-pre_detect_time)>distance_threshold)
                detect_time = [detect_time i];
                pre_detect_time = i;
                iswarning = 0;
                error_rate.p(i) = p_ini;
                error_rate.s(i) = s_ini;
                error_rate.p_min(i) = p_min_ini;
                error_rate.s_min(i) = s_min_ini;
                
                if (i-warn_time) >= num_least_relearn
                    SVMModel = svmtrain(y(warn_time:i),X(warn_time:i,:),...
                        '-t 2 -c 100 -g 1 -q');
%                     DTModel = classregtree(X(warn_time:i,:),y(warn_time:i));
                elseif (i-warn_time) < num_least_relearn && (warn_time + num_least_relearn) <= size(data,2)
                    SVMModel = svmtrain(y(warn_time:(warn_time+num_least_relearn)),...
                        X(warn_time:(warn_time+num_least_relearn),:),...
                '-t 2 -c 100 -g 1 -q');
%                     DTModel = classregtree(X(warn_time:(warn_time+num_least_relearn),:),...
%                 y(warn_time:(warn_time+num_least_relearn)));
                    num_wait_steps = warn_time + num_least_relearn - i;
                elseif (i-warn_time) < num_least_relearn && (warn_time + num_least_relearn) > size(data,2)
                    SVMModel = svmtrain(y(warn_time:end),X((warn_time:end),:),...
                        '-t 2 -c 100 -g 1 -q');
%                     DTModel = classregtree(X((warn_time:end),:),y(warn_time:end));
                    num_wait_steps = size(data,2) - i;
                end
            end
            
        else
            num_wait_steps = num_wait_steps - 1;
            error_rate.p(i) = p_ini;
            error_rate.s(i) = s_ini;
            error_rate.p_min(i) = p_min_ini;
            error_rate.s_min(i) = s_min_ini;
             
            if num_wait_steps == 0
                pre_detect_time = i;
            end
        end
    end    
else
    disp(['Unknown mode: ' mode]);      
end