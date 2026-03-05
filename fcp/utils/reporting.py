import numpy as np
from .utils import load_data


def summarize_fcp_results(results_dir, num_folds, data, outcome_dim, guidance_scale):

    cov_list = []
    avg_size_list = []

    for i in range(num_folds):
        try:
            file_dir = results_dir + "{}/fcp_{}_{}d_{}w_evaluation_result.pkl".format(i, data, outcome_dim, guidance_scale)
        
            fold_result = load_data(file_dir)
            cov_list.append(fold_result["empirical_coverage"])
            avg_size_list.append(np.mean(fold_result["est_region_size_list"]))
        except:
            print("no results at {}fold".format(i))
            pass

    print("avg coverage: {}".format(np.mean(cov_list)))
    print("std coverage: {}".format(np.std(cov_list)))        
    print("avg size: {}".format(np.mean(avg_size_list)))
    print("std size: {}".format(np.std(avg_size_list)))