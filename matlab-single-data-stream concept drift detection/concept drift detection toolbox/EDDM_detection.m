function [detect_time,data,distance_rate] = EDDM_detection(TrainingData,TestingData,warn_coeff,...
    detect_coeff,num_least_relearn,mode,min_num_errors,distance_threshold)
%%
if strcmpi(mode,'synthetic') == 1
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%    
elseif strcmpi(mode,'real') == 1
    % initialization
    detect_time = [];
    pre_detect_time = 0;
    iswarning=0; % boolean variable determin is warning or not
    num_wait_steps = 0;
    data = TestingData;
    distance_rate = struct('p',zeros(1,length(data)),'s',zeros(1,length(data)),'v',zeros(1,length(data)),...
        'p_max',zeros(1,length(data)),'s_max',zeros(1,length(data)));
    
    % Training SVM or DT model
    SVMModel = svmtrain(TrainingData(:,end),TrainingData(:,1:end-1),'-t 2 -c 100 -g 1 -q');
    
    p_ini = 0;
    s_ini = 0;
    v_ini = 0;
    p_max_ini = 0;
    s_max_ini = 0;

    % initialize features, true labels and predicted labels
    X = TestingData(:,1:end-1);
    y = TestingData(:,end);
    y_hat = zeros(size(y));
    data = [y,y_hat]';    
    
    % initialization for first time index
    [predicted_label] = svmpredict(rand(1),X(1,:),SVMModel,'-q');
    data(2,1) = predicted_label;
    if data(1,1) == data(2,1)
        distance_rate.p(1) = 0;
        distance_rate.s(1) = 0;
        distance_rate.v(1) = 0;
        distance_rate.p_max(1) = 0;
        distance_rate.s_max(1) = 0;
        num_errors = 0;
        pre_error_time = 0;
        
    elseif data(1,1) ~= data(2,1)
        distance_rate.p(1) = 1;
        distance_rate.s(1) = 0;
        distance_rate.v(1) = 0;
        distance_rate.p_max(1) = 1;
        distance_rate.s_max(1) = 0;
        num_errors = 1;
        pre_error_time = 1;        
    end
    
    % concept drift detection
    %%
    for i=2:size(data,2)
        if num_wait_steps ==0
            [predicted_label] = svmpredict(rand(1),X(i,:),SVMModel,'-q');
            data(2,i) = predicted_label;
            %% update error_rate
            if data(1,i) == data(2,i)
                distance_rate.p(i) = distance_rate.p(i-1);
                distance_rate.v(i) = distance_rate.v(i-1);                
                distance_rate.s(i) = distance_rate.s(i-1);
            elseif data(1,i) ~= data(2,i)
                new_distance = (i-pre_detect_time) - (pre_error_time-pre_detect_time);
                num_errors = num_errors + 1;
                % estimate new mean
                distance_rate.p(i) = distance_rate.p(i-1) + ...
                    (new_distance-distance_rate.p(i-1))/num_errors;
                % estimate new variance
                distance_rate.v(i) = distance_rate.v(i-1) + ...
                    (new_distance-distance_rate.p(i))*(new_distance-distance_rate.p(i-1));
                distance_rate.s(i) = sqrt(distance_rate.v(i)/num_errors);
                pre_error_time = i;
            end
            
            if (distance_rate.p(i)+2*distance_rate.s(i))>(distance_rate.p_max(i-1)+2*distance_rate.s_max(i-1)) ...
                    && ((i-pre_detect_time)>distance_threshold)
                distance_rate.p_max(i) = distance_rate.p(i);
                distance_rate.s_max(i) = distance_rate.s(i);
            else
                distance_rate.p_max(i) = distance_rate.p_max(i-1);
                distance_rate.s_max(i) = distance_rate.s_max(i-1);
            end
            
            % calculate current test level
            test_level = (distance_rate.p(i)+2*distance_rate.s(i))/(distance_rate.p_max(i)+2*distance_rate.s_max(i));
            
            %% determine warnining or alarming
            if num_errors >= min_num_errors
            % check if warning starts
            if (test_level<warn_coeff) && (iswarning==0) && ((i-pre_detect_time)>distance_threshold)
                warn_time = i;
                iswarning = 1;

            elseif (test_level>=warn_coeff) && (iswarning==1)
                iswarning = 0;
            end
            
            % check if detection found
%             if (test_level<detect_coeff) && (iswarning == 1) && ((i-detect_time(end))>100)
            if (test_level<detect_coeff) && ((i-pre_detect_time)>distance_threshold)
                detect_time = [detect_time i];
                pre_detect_time = i;
                iswarning = 0;
                distance_rate.p(i) = p_ini;
                distance_rate.s(i) = s_ini;
                distance_rate.v(i) = v_ini;
                distance_rate.p_max(i) = p_max_ini;
                distance_rate.s_max(i) = s_max_ini;
                num_errors = 0;
                pre_error_time = pre_detect_time;                

                if (i-warn_time) >= num_least_relearn
                    SVMModel = svmtrain(y(warn_time:i),X(warn_time:i,:),...
                        '-t 2 -c 100 -g 1 -q');                    
                elseif (i-warn_time) < num_least_relearn && (warn_time + num_least_relearn) <= size(data,2)
                    options.MaxIter = 100000;
                    SVMModel = svmtrain(y(warn_time:(warn_time+num_least_relearn)),...
                        X(warn_time:(warn_time+num_least_relearn),:),...
                '-t 2 -c 100 -g 1 -q');                    
                    num_wait_steps = warn_time + num_least_relearn - i;
                elseif (i-warn_time) < num_least_relearn && (warn_time + num_least_relearn) > size(data,2)
                    options.MaxIter = 100000;
                    SVMModel = svmtrain(y(warn_time:end),X((warn_time:end),:),...
                        '-t 2 -c 100 -g 1 -q');
                    num_wait_steps = size(data,2) - i;                    
                end
            end
            end
            
        else
            num_wait_steps = num_wait_steps - 1;
            distance_rate.p(i) = p_ini;
            distance_rate.s(i) = s_ini;
            distance_rate.v(i) = v_ini;
            distance_rate.p_max(i) = p_max_ini;
            distance_rate.s_max(i) = s_max_ini;
%             num_errors = 0;
%             pre_error_time = i;                 
            if num_wait_steps == 0
                pre_detect_time = i;
                num_errors = 0;
                pre_error_time = pre_detect_time;
            end       
        end
    end
    
else
    disp(['Unknown mode: ' mode]);      
end