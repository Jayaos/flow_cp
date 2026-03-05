import torch
import os
import numpy as np
from tqdm import tqdm
from fcp.config import *
from fcp.model import CFGFlow, initialize_encoder, initialize_vector_field
from fcp.conditionalode import CFGFlowODE, CombinedODE
from fcp.utils import set_initial_gaussian_distribution
from fcp.path import set_affine_probability_path
from fcp.utils.data import build_dataloader
from fcp.utils.utils import *


def run_fcp(fcp_config: FCPConfig, encoder_config, device):

    dataset = load_data(fcp_config.dataset_dir)
    dataset_config = load_data(fcp_config.dataset_config_dir)

    for i, dataset_fold in dataset.items():

        train_dataloader, valid_dataloader, _ = build_dataloader(dataset_fold,
                                                                encoder_config.context_window,
                                                                fcp_config.batch_size,
                                                                fcp_config.past_resid_as_feature)
        
        outcome_dim = dataset_fold["pred_train"].shape[-1]
        if fcp_config.past_resid_as_feature:
            feature_dim = dataset_fold["x_train"].shape[-1] + outcome_dim
        else:
            feature_dim = dataset_fold["x_train"].shape[-1]

        # initialize the flow
        ts_encoder = initialize_encoder(encoder_config, feature_dim, outcome_dim)
        if isinstance(encoder_config, TransformerEncoderConfig):
            vector_field = initialize_vector_field(outcome_dim, 
                                                encoder_config.model_dim, 
                                                1, 
                                                fcp_config.hidden_dims, 
                                                fcp_config.vf_layer_type, 
                                                fcp_config.activation)
        elif isinstance(encoder_config, IdentityEncoderConfig):
            vector_field = initialize_vector_field(outcome_dim, 
                                                feature_dim, 
                                                1, 
                                                fcp_config.hidden_dims, 
                                                fcp_config.vf_layer_type, 
                                                fcp_config.activation)
        initial_dist = set_initial_gaussian_distribution(fcp_config.initial_gaussian_distribution_cov_scale, 
                                                         outcome_dim)
        prob_path = set_affine_probability_path()

        cfg_flow = CFGFlow(ts_encoder, vector_field, initial_dist, prob_path)
        
        print("training flow...")
        cfg_flow, fold_train_result = train_cfg_flow(cfg_flow, train_dataloader, valid_dataloader, 
                                                     fcp_config.null_condition_prob,
                                                     fcp_config.max_epoch, fcp_config.additional_training_epoch,
                                                     fcp_config.learning_rate, fcp_config.early_stop, device=device)
        
        print("saving the results...")
        os.makedirs(fcp_config.saving_dir + "/{}".format(i), exist_ok=True)
        fold_saving_dir = fcp_config.saving_dir + "/{}/".format(i)

        torch.save(cfg_flow.state_dict(), 
                   fold_saving_dir + "fcp_{}_{}d.pt".format(dataset_config.dataset_name, 
                                                            outcome_dim))
        save_data(fold_saving_dir + "fcp_{}_{}d_config.pkl".format(dataset_config.dataset_name,
                                                                   outcome_dim), fcp_config)
        save_data(fold_saving_dir + "fcp_{}_{}d_encoder_config.pkl".format(dataset_config.dataset_name,
                                                                   outcome_dim), encoder_config)
        save_data(fold_saving_dir + "fcp_{}_{}d_train_result.pkl".format(dataset_config.dataset_name, 
                                                                         outcome_dim),
                                                                         fold_train_result)
        

def evaluate_fcp(fcp_evaluation_config: FCPEvaluationConfig, device):

    dataset = load_data(fcp_evaluation_config.dataset_dir)
    dataset_config = load_data(fcp_evaluation_config.dataset_config_dir)

    for i, dataset_fold in dataset.items():

        outcome_dim = dataset_fold["pred_train"].shape[-1]

        try:

            fcp_config = load_data(fcp_evaluation_config.model_dir + "/{}/fcp_{}_{}d_config.pkl".format(i, 
                                                                                dataset_config.dataset_name,
                                                                                outcome_dim))
            
            encoder_config = load_data(fcp_evaluation_config.model_dir + "/{}/fcp_{}_{}d_encoder_config.pkl".format(i, 
                                                                                            dataset_config.dataset_name,
                                                                                            outcome_dim))

            fold_saved_model_dir = fcp_evaluation_config.model_dir + "/{}/fcp_{}_{}d.pt".format(i, 
                                                                        dataset_config.dataset_name,
                                                                        outcome_dim)
            
            train_dataloader, valid_dataloader, test_dataloader = build_dataloader(dataset_fold,
                                                                    encoder_config.context_window,
                                                                    fcp_config.batch_size,
                                                                    fcp_config.past_resid_as_feature)
        except:
            continue
        
        if fcp_config.past_resid_as_feature:
            feature_dim = dataset_fold["x_train"].shape[-1] + outcome_dim
        else:
            feature_dim = dataset_fold["x_train"].shape[-1]
        
        # initialize the flow
        ts_encoder = initialize_encoder(encoder_config, feature_dim, outcome_dim)
        if isinstance(encoder_config, TransformerEncoderConfig):
            # transformer encoder
            vector_field = initialize_vector_field(outcome_dim, 
                                                encoder_config.model_dim, 
                                                1, 
                                                fcp_config.hidden_dims, 
                                                fcp_config.vf_layer_type, 
                                                fcp_config.activation)
        elif isinstance(encoder_config, IdentityEncoderConfig):
            # no encoder
            vector_field = initialize_vector_field(outcome_dim, 
                                                feature_dim, 
                                                1, 
                                                fcp_config.hidden_dims, 
                                                fcp_config.vf_layer_type, 
                                                fcp_config.activation)
        initial_dist = set_initial_gaussian_distribution(fcp_config.initial_gaussian_distribution_cov_scale, 
                                                         outcome_dim)
        prob_path = set_affine_probability_path()

        # load the saved flow 
        cfg_flow = CFGFlow(ts_encoder, vector_field, initial_dist, prob_path)
        cfg_flow.load_state_dict(torch.load(fold_saved_model_dir))

        print("computing empirical coverage by solving reverse flow ODE...")
        cfg_flow_ode = CFGFlowODE(cfg_flow, 
                                  atol=fcp_evaluation_config.atol, 
                                  rtol=fcp_evaluation_config.rtol)
        empirical_coverage, _ = compute_empirical_coverage(cfg_flow, 
                                                        cfg_flow_ode, 
                                                        test_dataloader, 
                                                        fcp_evaluation_config.guidance_scale, 
                                                        fcp_evaluation_config.target_coverage, 
                                                        initial_dist, device)
        print("empirical coverage: {}".format(empirical_coverage))

        print("estimating the region size...")
        combined_ode = CombinedODE(cfg_flow, fcp_evaluation_config.atol, fcp_evaluation_config.rtol)
        est_region_size_list, det_jacobian_mean_list, base_region_size = estimate_region_size(cfg_flow, 
                                                                            combined_ode,  
                                                                            test_dataloader,
                                                                            fcp_evaluation_config.guidance_scale,
                                                                            fcp_evaluation_config.target_coverage,
                                                                            initial_dist,
                                                                            fcp_evaluation_config.sampling_num, 
                                                                            fcp_evaluation_config.batch_processing,
                                                                            device=device)
        print("base region size: {}".format(base_region_size))
        print("estimated avg region size: {}".format(np.mean(est_region_size_list)))
        
        evaluation_result = {"empirical_coverage" : empirical_coverage, 
                             "est_region_size_list" : est_region_size_list,
                             "base_region_size" : base_region_size,
                             "det_jacobian_mean_list" : det_jacobian_mean_list}
        
        save_data(fcp_evaluation_config.model_dir + 
                  "/{}/fcp_{}_{}d_{}w_evaluation_result.pkl".format(i, dataset_config.dataset_name,
                                                                outcome_dim, fcp_evaluation_config.guidance_scale),
                                                                evaluation_result)
        

def evaluate_coverage_fcp(fcp_evaluation_config: FCPEvaluationConfig, device):
    """
    computes the avg of empirical coverages for quick sanity check experiment
    """

    dataset = load_data(fcp_evaluation_config.dataset_dir)
    dataset_config = load_data(fcp_evaluation_config.dataset_config_dir)
    coverage_list = []

    for i, dataset_fold in dataset.items():

        outcome_dim = dataset_fold["pred_train"].shape[-1]

        fcp_config = load_data(fcp_evaluation_config.model_dir + "/{}/fcp_{}_{}d_config.pkl".format(i, 
                                                                              dataset_config.dataset_name,
                                                                              outcome_dim))
        
        encoder_config = load_data(fcp_evaluation_config.model_dir + "/{}/fcp_{}_{}d_encoder_config.pkl".format(i, 
                                                                                          dataset_config.dataset_name,
                                                                                          outcome_dim))

        fold_saved_model_dir = fcp_evaluation_config.model_dir + "/{}/fcp_{}_{}d.pt".format(i, 
                                                                      dataset_config.dataset_name,
                                                                      outcome_dim)
        
        train_dataloader, valid_dataloader, test_dataloader = build_dataloader(dataset_fold,
                                                                 encoder_config.context_window,
                                                                 fcp_config.batch_size,
                                                                 fcp_config.past_resid_as_feature)
        
        if fcp_config.past_resid_as_feature:
            feature_dim = dataset_fold["x_train"].shape[-1] + outcome_dim
        else:
            feature_dim = dataset_fold["x_train"].shape[-1]
        
        # initialize the flow
        ts_encoder = initialize_encoder(encoder_config, feature_dim, outcome_dim)
        if isinstance(encoder_config, TransformerEncoderConfig):
            vector_field = initialize_vector_field(outcome_dim, 
                                                encoder_config.model_dim, 
                                                1, 
                                                fcp_config.hidden_dims, 
                                                fcp_config.vf_layer_type, 
                                                fcp_config.activation)
        elif isinstance(encoder_config, IdentityEncoderConfig):
            vector_field = initialize_vector_field(outcome_dim, 
                                                feature_dim, 
                                                1, 
                                                fcp_config.hidden_dims, 
                                                fcp_config.vf_layer_type, 
                                                fcp_config.activation)
        initial_dist = set_initial_gaussian_distribution(fcp_config.initial_gaussian_distribution_cov_scale, 
                                                         outcome_dim)
        prob_path = set_affine_probability_path()

        # load the saved flow 
        cfg_flow = CFGFlow(ts_encoder, vector_field, initial_dist, prob_path)
        cfg_flow.load_state_dict(torch.load(fold_saved_model_dir))

        print("computing empirical coverage by solving reverse flow ODE...")
        cfg_flow_ode = CFGFlowODE(cfg_flow, 
                                  atol=fcp_evaluation_config.atol, 
                                  rtol=fcp_evaluation_config.rtol)
        empirical_coverage, coverages = compute_empirical_coverage(cfg_flow, 
                                                        cfg_flow_ode, 
                                                        test_dataloader, 
                                                        fcp_evaluation_config.guidance_scale, 
                                                        fcp_evaluation_config.target_coverage, 
                                                        initial_dist, device)
        coverage_list.append(empirical_coverage)
        print("empirical coverage: {}".format(empirical_coverage))
        coverage_evaluation_result = {"empirical_coverage" : empirical_coverage,
                                         "coverages" : coverages} 
        save_data(fcp_evaluation_config.model_dir + 
                    "/{}/fcp_{}_{}d_{}w_coverage_evaluation_result.pkl".format(i, dataset_config.dataset_name,
                                                                outcome_dim, fcp_evaluation_config.guidance_scale),
                                                                coverage_evaluation_result)
    
    print("dataset: {}".format(dataset_config.dataset_name))
    print("dim: {}".format(outcome_dim))
    print("vector field architecture: {}".format(fcp_config.hidden_dims))
    print("init cov scale: {}".format(fcp_config.initial_gaussian_distribution_cov_scale))
    print("guidance_scale: {}".format(fcp_evaluation_config.guidance_scale))
    print("empirical coverage: {}".format(np.mean(coverage_list)))



def train_cfg_flow(flow, train_dataloader, valid_dataloader, null_condition_prob,
                   max_epoch, additional_training_epoch, 
                   learning_rate, early_stop, device="cpu"):

    flow.to(device)
    Optimizer = torch.optim.Adam(flow.parameters(), lr=learning_rate)

    training_loss_record = []
    validation_loss_record = []
    best_loss = np.inf
    best_epoch = 0

    for e in range(max_epoch):

        if early_stop:
            if (e+1-best_epoch) >= early_stop:
                break

        batch_loss_sum = 0.
        batch_num = len(train_dataloader)
        
        flow.train()
        for x_batch, y_batch in tqdm(train_dataloader):

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            Optimizer.zero_grad()
            attn_mask = torch.nn.Transformer.generate_square_subsequent_mask(x_batch.size(1)) # causal mask
            attn_mask = attn_mask.to(device)
            # src_key_padding_mask is None since we fix input sequence with length of past_window 
            # TODO: update to have src_key_padding_mask is not None. We can utilize when seq_len < past_window
            output = flow(x_batch, 
                          src_mask=attn_mask,
                          src_key_padding_mask=None, 
                          y=y_batch, 
                          null_condition_prob=null_condition_prob,
                          device=device)
            loss = output["loss"]
            loss.backward()
            Optimizer.step()
            batch_loss_sum += loss.item()

        epoch_loss = batch_loss_sum/batch_num
        training_loss_record.append(epoch_loss)
        print("training loss at epoch {}: {}".format(e+1, epoch_loss))

        flow.eval()
        batch_loss_sum = 0.
        batch_num = len(valid_dataloader)
        for x_batch, y_batch in tqdm(valid_dataloader):

            with torch.no_grad():

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                attn_mask = torch.nn.Transformer.generate_square_subsequent_mask(x_batch.size(1)) # causal mask
                attn_mask = attn_mask.to(device)
                output = flow(x_batch, 
                              src_mask=attn_mask, 
                              src_key_padding_mask=None, 
                              y=y_batch,
                              null_condition_prob=null_condition_prob,
                              device=device)
                loss = output["loss"]
                batch_loss_sum += loss.item()

        epoch_loss = batch_loss_sum/batch_num
        validation_loss_record.append(epoch_loss)
        print("validation loss at epoch {}: {}".format(e+1, epoch_loss))

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = e+1
            best_model = flow.state_dict()
    
    print("best model at epoch {}".format(best_epoch))
    flow.load_state_dict(best_model)

    # additional training with validation dataset
    if additional_training_epoch > 0:
        print("start additional training on validation dataset...")
        for e in range(additional_training_epoch):

            batch_loss_sum = 0.
            batch_num = len(valid_dataloader)

            flow.train()
            for x_batch, y_batch in tqdm(valid_dataloader):

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                Optimizer.zero_grad()
                attn_mask = torch.nn.Transformer.generate_square_subsequent_mask(x_batch.size(1)) # causal mask
                attn_mask = attn_mask.to(device)
                output = flow(x_batch, 
                              src_mask=attn_mask, 
                              src_key_padding_mask=None, 
                              y=y_batch,
                              null_condition_prob=null_condition_prob,
                              device=device)
                loss = output["loss"]
                loss.backward()
                Optimizer.step()
                batch_loss_sum += loss.item()

            epoch_loss = batch_loss_sum/batch_num
            print("training loss of additional training at epoch {}: {}".format(e+1, epoch_loss))

    training_result = {"training_loss_hist" : training_loss_record, 
                       "validation_loss_hist" : validation_loss_record}

    return flow, training_result
