import numpy as np
from numpy import unique
import copy

from timeit import default_timer as timer
from skmultiflow.utils import constants


def test(evaluator,reset_model, need_adapt,detected_drift_points,retrain_num,reset):    
    adapt_X = None
    adapt_y = None
    
    if need_adapt:
        drift_point_cnt = 0
        adapt_start_point = detected_drift_points[drift_point_cnt]
        adapt_end_point = adapt_start_point + retrain_num -1
    
    evaluator._start_time = timer()
    evaluator._end_time = timer()
    print('Prequential Evaluation')
    print('Evaluating {} target(s).'.format(evaluator.stream.n_targets))

    actual_max_samples = evaluator.stream.n_remaining_samples()
    if actual_max_samples == -1 or actual_max_samples > evaluator.max_samples:
        actual_max_samples = evaluator.max_samples

    first_run = True
    if evaluator.pretrain_size > 0:
        print('Pre-training on {} sample(s).'.format(evaluator.pretrain_size))

        X, y = evaluator.stream.next_sample(evaluator.pretrain_size)
        
        if need_adapt:
            adapt_X = np.zeros((0,X.shape[1]))
            adapt_y = np.zeros(0)

        for i in range(evaluator.n_models):
            if evaluator._task_type == constants.CLASSIFICATION:
                # Training time computation
                evaluator.running_time_measurements[i].compute_training_time_begin()
                evaluator.model[i].partial_fit(X=X, y=y, classes=evaluator.stream.target_values)
                evaluator.running_time_measurements[i].compute_training_time_end()
            elif evaluator._task_type == constants.MULTI_TARGET_CLASSIFICATION:
                evaluator.running_time_measurements[i].compute_training_time_begin()
                evaluator.model[i].partial_fit(X=X, y=y, classes=unique(evaluator.stream.target_values))
                evaluator.running_time_measurements[i].compute_training_time_end()
            else:
                evaluator.running_time_measurements[i].compute_training_time_begin()
                evaluator.model[i].partial_fit(X=X, y=y)
                evaluator.running_time_measurements[i].compute_training_time_end()
            evaluator.running_time_measurements[i].update_time_measurements(evaluator.pretrain_size)
        evaluator.global_sample_count += evaluator.pretrain_size
        first_run = False

    update_count = 0
    print('Evaluating...')
    while ((evaluator.global_sample_count < actual_max_samples) & (evaluator._end_time - evaluator._start_time < evaluator.max_time)
            & (evaluator.stream.has_more_samples())):
        try:
            X, y = evaluator.stream.next_sample(evaluator.batch_size)

            if X is not None and y is not None:
                
                # Test
                prediction = [[] for _ in range(evaluator.n_models)]
                for i in range(evaluator.n_models):
                    try:
                        # Testing time
                        evaluator.running_time_measurements[i].compute_testing_time_begin()
                        prediction[i].extend(evaluator.model[i].predict(X))
                        evaluator.running_time_measurements[i].compute_testing_time_end()
                    except TypeError:
                        raise TypeError("Unexpected prediction value from {}"
                                        .format(type(evaluator.model[i]).__name__))
                evaluator.global_sample_count += evaluator.batch_size

                for j in range(evaluator.n_models):
                    for i in range(len(prediction[0])):
                        evaluator.mean_eval_measurements[j].add_result(y[i], prediction[j][i])
                        evaluator.current_eval_measurements[j].add_result(y[i], prediction[j][i])
                evaluator._check_progress(actual_max_samples)

                # Active adapt
                if (need_adapt) and (adapt_start_point<= evaluator.global_sample_count <= adapt_end_point): 
                    if reset and (evaluator.global_sample_count == adapt_start_point):                                  
                        evaluator._init_evaluation(model=reset_model, stream=evaluator.stream, model_names=evaluator.model_names)
                    
                    # TODO: evaluator.batch_size must be 1
                    assert evaluator.batch_size == 1
                    adapt_X = np.concatenate((adapt_X,X),axis=0)
                    adapt_y = np.concatenate((adapt_y,y),axis=0)
                    
                    if first_run:
                        for i in range(evaluator.n_models):
                            if evaluator._task_type != constants.REGRESSION and \
                               evaluator._task_type != constants.MULTI_TARGET_REGRESSION:
                                # Accounts for the moment of training beginning
                                evaluator.running_time_measurements[i].compute_training_time_begin()
                                evaluator.model[i].partial_fit(X, y, evaluator.stream.target_values)
                                # Accounts the ending of training
                                evaluator.running_time_measurements[i].compute_training_time_end()
                            else:
                                evaluator.running_time_measurements[i].compute_training_time_begin()
                                evaluator.model[i].partial_fit(X, y)
                                evaluator.running_time_measurements[i].compute_training_time_end()

                            # Update total running time
                            evaluator.running_time_measurements[i].update_time_measurements(evaluator.batch_size)
                        first_run = False
                    else:
                        for i in range(evaluator.n_models):
                            evaluator.running_time_measurements[i].compute_training_time_begin()
                            evaluator.model[i].partial_fit(X, y)
                            evaluator.running_time_measurements[i].compute_training_time_end()
                            evaluator.running_time_measurements[i].update_time_measurements(evaluator.batch_size)
                            
                    if evaluator.global_sample_count == adapt_end_point:                        
                        for i in range(evaluator.n_models):
                            evaluator.running_time_measurements[i].compute_training_time_begin()
                            evaluator.model[i].partial_fit(adapt_X, adapt_y)
                            evaluator.running_time_measurements[i].compute_training_time_end()
                            evaluator.running_time_measurements[i].update_time_measurements(evaluator.batch_size)
                        
                        # preparing for the next concept drift
                        adapt_X = np.zeros((0,X.shape[1]))
                        adapt_y = np.zeros(0)
                        
                        drift_point_cnt = drift_point_cnt + 1
                        if drift_point_cnt < detected_drift_points.shape[0]:
                            adapt_start_point = detected_drift_points[drift_point_cnt]
                            adapt_end_point = adapt_start_point + retrain_num
                        else:
                            adapt_start_point = np.inf
                            adapt_end_point = np.inf
                        
                        

                if ((evaluator.global_sample_count % evaluator.n_wait) == 0 or
                        (evaluator.global_sample_count >= actual_max_samples) or
                        (evaluator.global_sample_count / evaluator.n_wait > update_count + 1)):
                    if prediction is not None:
                        evaluator._update_metrics()
                    update_count += 1

            evaluator._end_time = timer()
        except BaseException as exc:
            print(exc)
            if exc is KeyboardInterrupt:
                evaluator._update_metrics()
            break

    # Flush file buffer, in case it contains data
    evaluator._flush_file_buffer()

    if len(set(evaluator.metrics).difference({constants.DATA_POINTS})) > 0:
        evaluator.evaluation_summary()
    else:
        print('Done')

    if evaluator.restart_stream:
        evaluator.stream.restart()

    return evaluator.model


def evaluate(evaluator, stream, model, model_names, need_adapt=False, detected_drift_points=np.array([]), retrain_num = 200, reset=True):
    """ Evaluates a model or set of models on samples from a stream.
    Parameters
    ----------
    stream: Stream
        The stream from which to draw the samples.
    model: skmultiflow.core.BaseStreamModel or sklearn.base.BaseEstimator or list
        The model or list of models to evaluate.
    model_names: list, optional (Default=None)
        A list with the names of the models.
    Returns
    -------
    StreamModel or list
        The trained model(s).
    """
    
    original_model = copy.deepcopy(model)
        
    evaluator._init_evaluation(model=model, stream=stream, model_names=model_names)

    if evaluator._check_configuration():
        evaluator._reset_globals()
        # Initialize metrics and outputs (plots, log files, ...)
        evaluator._init_metrics()
        evaluator._init_plot()
        evaluator._init_file()
        
        evaluator.model = test(evaluator,original_model,need_adapt,detected_drift_points,retrain_num,reset)

        if evaluator.show_plot:
            evaluator.visualizer.hold()

        return evaluator.model
    
def fading_accuracy(cur_accuracy, fading_factor):
    stream_length = cur_accuracy.shape[0]
    cur_error = 1 - cur_accuracy
    
    fraction = 0
    fading_error = np.zeros(stream_length)
    for i, cur_err in enumerate(cur_error):
        fraction = fading_factor * fraction + 1
        if i==0:
            fading_error[i] = cur_err
        else:
            fading_error[i] = (fading_error[i-1]*((fraction-1)/fading_factor)*fading_factor + cur_err) / fraction
    
    return 1 - fading_error
 
 
def get_evaluation_mean_performance(evaluator):
    mean_measurements = evaluator.get_mean_measurements()
    model_num = len(mean_measurements)
    
    mean_performance = np.zeros((model_num, 4))
    for i, mean_measurement in enumerate(mean_measurements):
        mean_performance[i,0] = mean_measurement.accuracy_score()
        mean_performance[i,1] = mean_measurement.kappa_score()
        mean_performance[i,2] = mean_measurement.recall_score()
        mean_performance[i,3] = mean_measurement.f1_score()
    return mean_performance