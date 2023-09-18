function show_results(detect_time,test_L,num_least_relearn)
%% number of elements in classification results
num = numel(detect_time);

figure,
for i=1:num-1
    % # of methods
    subplot(num-1,1,i);
    xlim([0 test_L]);
    for j=1:length(detect_time{i}.values)
        % # of runs
        % Actual detection
        line([detect_time{i}.values(j) detect_time{i}.values(j)],ylim,'Color','r'); hold on
    end

    for j=1:length(detect_time{num}.values)
        % Ground truth e.g., 900, 1900, 2900, ...
    line([detect_time{num}.values(j)-num_least_relearn detect_time{num}.values(j)-num_least_relearn],...
        ylim,'Color','k');
    end
    title(['concept drift detection results with ',detect_time{i}.method]);
    hold off
end