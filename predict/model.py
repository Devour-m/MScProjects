from skmultiflow.bayes import NaiveBayes
from skmultiflow.trees import HoeffdingTreeClassifier, HoeffdingAdaptiveTreeClassifier
from skmultiflow.lazy import KNNClassifier
from skmultiflow.meta import AdaptiveRandomForestClassifier

def init_models(model_names):
    models = list()
    for model_name in model_names:
        if model_name == 'NB':
            model = NaiveBayes()
        elif model_name == 'HT':
            model = HoeffdingTreeClassifier()
        elif model_name == 'HAT':
            model = HoeffdingAdaptiveTreeClassifier()
        elif model_name == 'KNN':
            model = KNNClassifier()
        elif model_name == 'ARF':
            model = AdaptiveRandomForestClassifier()
        else:
            model = None
        models.append(model)
    return models

def print_model_names(model_names):
    model_name_all = ''
    for model_name in model_names:
        model_name_all = model_name_all + model_name + '_'
    return model_name_all[:-1]