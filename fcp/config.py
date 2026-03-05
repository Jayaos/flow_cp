

class FCPConfig():
    
    def __init__(
            self,
            dataset_dir,
            dataset_config_dir,
            saving_dir,
            vf_layer_type,
            hidden_dims,
            activation,
            initial_gaussian_distribution_cov_scale,
            null_condition_prob,
            batch_size,
            learning_rate,
            max_epoch,
            additional_training_epoch,
            past_resid_as_feature,
            early_stop,
            ):
        
        self.dataset_dir = dataset_dir
        self.dataset_config_dir = dataset_config_dir
        self.saving_dir = saving_dir
        self.vf_layer_type = vf_layer_type
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.initial_gaussian_distribution_cov_scale = initial_gaussian_distribution_cov_scale
        self.null_condition_prob = null_condition_prob
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.additional_training_epoch = additional_training_epoch
        self.past_resid_as_feature = past_resid_as_feature
        self.early_stop = early_stop


class FCPEvaluationConfig():

    def __init__(self, 
                 dataset_dir,
                 dataset_config_dir,
                 model_dir,
                 guidance_scale,
                 target_coverage,
                 sampling_num,
                 batch_processing,
                 atol,
                 rtol,
                 ):
        
        self.dataset_dir = dataset_dir
        self.dataset_config_dir = dataset_config_dir
        self.model_dir = model_dir
        self.guidance_scale = guidance_scale
        self.target_coverage = target_coverage
        self.sampling_num = sampling_num
        self.batch_processing = batch_processing
        self.atol = atol 
        self.rtol = rtol


class IdentityEncoderConfig():

    def __init__(self, context_window):
        self.context_window = context_window
        

class TransformerEncoderConfig():

    def __init__(self, model_dim, num_head, dim_ff, num_layer, dropout, context_window):
        self.model_dim = model_dim
        self.num_head = num_head
        self.dim_ff = dim_ff
        self.num_layer = num_layer
        self.dropout = dropout
        self.context_window = context_window

    
class LOOBootstrapPredictorConfig():

    def __init__(self, save_dir, num_estimators, stride, train_test_ratio, valid_test_ratio):
        
        self.base_predictor = "loo-bootstrap"
        self.save_dir = save_dir
        self.num_estimators = num_estimators
        self.stride = stride
        self.train_test_ratio = train_test_ratio
        self.valid_test_ratio = valid_test_ratio


class LSTMPredictorConfig():

    def __init__(self, save_dir, dim_model, context_window, predictor_data_ratio, train_valid_ratio, train_valid_test_ratio, 
                 max_epoch, batch_size, 
                 learning_rate, early_stop):
        
        self.base_predictor = "LSTM"
        self.save_dir = save_dir
        self.dim_model = dim_model
        self.context_window = context_window
        self.predictor_data_ratio = predictor_data_ratio
        self.train_valid_ratio = train_valid_ratio
        self.train_valid_test_ratio = train_valid_test_ratio
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stop = early_stop


class RealDatasetConfig():

    def __init__(self, dataset_name, dataset_dir, rolling_window, standardize, location_num, repeat_num, seed):
        
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.rolling_window = rolling_window
        self.standardize = standardize
        self.location_num = location_num
        self.repeat_num = repeat_num
        self.seed = seed


class SimulationDatasetConfig():

    def __init__(self, dataset_name, dim_outcome, lag_order, len_sequence, standardize, repeat_num, seed):
        
        self.dataset_name = dataset_name
        self.dim_outcome = dim_outcome
        self.lag_order = lag_order
        self.len_sequence = len_sequence
        self.standardize = standardize
        self.repeat_num = repeat_num
        self.seed = seed