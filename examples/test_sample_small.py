import numpy as np
from AutoScore import AutoScore_binary, utils


# np.random.seed(0)
data = utils.load_sample_data('small')
data = data.rename(columns={'Mortality_inpatient': 'label'})
utils.compute_descriptive_table(data)
utils.check_data(data)

data = utils.convert_categorical_vars(data)
AutoScore_binary.compute_uni_variable_table(data)
AutoScore_binary.compute_multi_variable_table(data)
train_set, validation_set, test_set = utils.split_data(data, (0.7, 0, 0.3), cross_validation=True, strat_by_label=False)

rank = AutoScore_binary.AutoScore_rank(train_set, validation_set, method='rf')
# rank = AutoScore_binary.AutoScore_rank(train_set, validation_set, method='auc')

AUC_df = AutoScore_binary.AutoScore_parsimony(train_set, validation_set, rank, n_min=1, n_max=rank.shape[0],
                                              cross_validation=True, quantiles=(0, 0.25, 0.5, 0.75, 1))
# AUC_df = AutoScore_binary.AutoScore_parsimony(train_set, validation_set, rank, n_min=1, n_max=rank.shape[0],
#                                               cross_validation=True, categorize='kmeans', max_cluster=5)

final_variables = ['Age', 'Lab_B', 'Lab_H', 'Vital_E', 'Lab_K']
cut_vec = AutoScore_binary.AutoScore_weighting(train_set, validation_set, final_variables, max_score=100,
                                               categorize="quantile", quantiles=(0, 0.25, 0.5, 0.75, 1))
print('Initial cut Vector:\n', cut_vec)

# Fine-tune cut_vec
cut_vec['Age'] = np.array([35, 50, 80])
cut_vec['Lab_H'] = np.array([1, 2, 3])
cut_vec['Lab_B'] = np.array([8, 12, 18])
cut_vec['Vital_E'] = np.array([15, 22])
cut_vec['Lab_K'] = np.array([45, 60])
print('Fine-tuned cut vector:\n', cut_vec)

score_table = AutoScore_binary.AutoScore_fine_tuning(train_set, validation_set, final_variables, cut_vec, max_score=100)

pred_score = AutoScore_binary.AutoScore_testing(test_set, final_variables, cut_vec, score_table)
# pred_score.to_csv('pred_score.csv')
print(pred_score.head())
