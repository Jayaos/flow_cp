import numpy as np
from .model import LOOBootstrapPredictor, LSTMPredictor
from .data import *
from fcp.utils.utils import save_data


def run_base_predictor(base_predictor_config, dataset_config, device):

    dataset = dict()

    if dataset_config.dataset_name == "wind":
        total_location_num = 30
    elif dataset_config.dataset_name == "solar":
        total_location_num = 9
    elif dataset_config.dataset_name == "traffic":
        total_location_num = 15
    else:
        total_location_num = None

    np.random.seed(dataset_config.seed)

    for i in range(dataset_config.repeat_num):

        print("run {}".format(i+1))

        if total_location_num:
            location_selected = list(np.random.choice(total_location_num, dataset_config.location_num, replace=False))
        else:
            location_selected = None
                    
        if dataset_config.dataset_name == "solar":
            print("load solar dataset...")
            X, Y = load_solar_dataset(dataset_config.dataset_dir, 
                                      dataset_config.rolling_window, 
                                      dataset_config.standardize, 
                                      location_selected)

        elif dataset_config.dataset_name == "wind":
            print("load wind dataset...")
            X, Y = load_wind_dataset(dataset_config.dataset_dir, 
                                    dataset_config.rolling_window, 
                                    dataset_config.standardize, 
                                    location_selected)

        elif dataset_config.dataset_name == "traffic":
            print("load traffic dataset...")
            X, Y = load_traffic_dataset(dataset_config.dataset_dir, 
                                        dataset_config.rolling_window, 
                                        dataset_config.standardize, 
                                        location_selected)
            
        else:
            raise ValueError("dataset out of range")
        
        if base_predictor_config.base_predictor in ["loo-bootstrap"]:
            predictor = LOOBootstrapPredictor(base_predictor_config.num_estimators, base_predictor_config.stride)
            predictor.fit(X, Y, base_predictor_config.train_test_ratio)
            dataset[i] = predictor.generate_residual_dataset(base_predictor_config.valid_test_ratio)

        elif base_predictor_config.base_predictor in ["LSTM", "lstm"]:
            predictor = LSTMPredictor(base_predictor_config.dim_model, base_predictor_config.context_window)
            predictor.fit(X, Y, base_predictor_config.predictor_data_ratio, base_predictor_config.train_valid_ratio,
                          base_predictor_config.max_epoch, base_predictor_config.batch_size, 
                          base_predictor_config.learning_rate, base_predictor_config.early_stop, device)
            dataset[i] = predictor.generate_residual_dataset(base_predictor_config.train_valid_test_ratio)
        else:
            raise NotImplementedError("wrong predictor argument")

        if location_selected:
            dataset[i]["location_selected"] = location_selected

    os.makedirs(base_predictor_config.save_dir, exist_ok=True)

    dataset_file_name = "/{}_{}_repeat{}_dataset.pkl".format(dataset_config.dataset_name, 
                                                             base_predictor_config.base_predictor, 
                                                             dataset_config.repeat_num)
    save_data(base_predictor_config.save_dir + dataset_file_name, dataset)

    config_file_name = "/{}_{}_dataset_config.pkl".format(dataset_config.dataset_name, 
                                                          base_predictor_config.base_predictor)
    save_data(base_predictor_config.save_dir + config_file_name, dataset_config)