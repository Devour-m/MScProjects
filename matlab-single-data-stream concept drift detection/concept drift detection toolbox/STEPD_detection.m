function [detect_time,data,error_rate] = STEPD_detection(TrainingData,TestingData,warn_coeff,...
    detect_coeff,num_least_relearn,mode,distance_threshold,window_size)
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
    
    error_rate = struct('P_value',zeros(1,length(TestingData)));
    
    % Training SVM or DT model
    SVMModel = svmtrain(TrainingData(:,end),TrainingData(:,1:end-1),...
        '-t 2 -c 100 -g 1 -q');
%     DTModel = classregtree(TrainingData(:,1:end-1),TrainingData(:,end));

    P_value_ini = 1; % initialize P_value
    num_correct = 0;
    
    % initialize features, true labels and predicted labels
    X = TestingData(:,1:end-1);
    y = TestingData(:,end);
    y_hat = zeros(size(y));
    data = [y,y_hat]';
    
    % initialization for first time index
    data(2,1) = svmpredict(rand(1),X(1,:),SVMModel,'-q');
%     data(2,1) = eval(DTModel,X(1,:));
    if data(1,1) == data(2,1)
        error_rate.P_value(1) = 1;
        num_correct = 1;
    elseif data(1,1) ~= data(2,1)
        error_rate.P_value(1) = 1;
        num_correct = 0;
    end
    
    % concept drift detection
    for i=2:size(data,2)
        if num_wait_steps ==0
            data(2,i) = svmpredict(rand(1),X(i,:),SVMModel,'-q');
%             data(2,i) = eval(DTModel,X(i,:));
            %% update error_rate
            if data(1,i) == data(2,i)
                num_correct = num_correct + 1;
            elseif data(1,i) ~= data(2,i)
                num_correct = num_correct;           
            end
            
            if (i-pre_detect_time)>=window_size
                num_error_window = xor(data(1,i-window_size+1:i),data(2,i-window_size+1:i));
                num_correct_window = window_size - sum(num_error_window);
            % statistic calculation (using chi-square test with Yates's correction)
            error_rate.P_value(i) = prop_test([(num_correct-num_correct_window) num_correct_window],...
                [(i-pre_detect_time-window_size+1) window_size], true);
            else
                error_rate.P_value(i) = 1;
            end
            
            
            %% determine warnining or alarming
            % check if warning starts
            if (error_rate.P_value(i)<warn_coeff) && (iswarning==0)...
                    && ((i-pre_detect_time)>distance_threshold)
                warn_time = i;
                iswarning = 1;

            elseif (error_rate.P_value(i)>=warn_coeff) && (iswarning==1)
                iswarning = 0;
            end
            
            % check if detection found
            if (error_rate.P_value(i)<detect_coeff) && ...
                    ((i-pre_detect_time)>distance_threshold)
                detect_time = [detect_time i];
                pre_detect_time = i;
                iswarning = 0;
                error_rate.P_value(i) = P_value_ini;
                num_correct = 0;
                
                if (i-warn_time) >= num_least_relearn
                    SVMModel = svmtrain(y(warn_time:i),X(warn_time:i,:),...
                        '-t 2 -c 100 -g 1 -q');
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
            
        else
            num_wait_steps = num_wait_steps - 1;
            error_rate.P_value(i) = P_value_ini;
            num_correct = 0;
             
            if num_wait_steps == 0
                pre_detect_time = i;
            end
        end
    end    
else
    disp(['Unknown mode: ' mode]);      
end