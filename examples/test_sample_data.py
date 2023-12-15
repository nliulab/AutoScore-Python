import numpy as np
from AutoScore import AutoScore_binary, utils


# np.random.seed(0)
data = utils.load_sample_data('binary')
data = data.rename(columns={'Mortality_inpatient': 'label'})
utils.compute_descriptive_table(data)
utils.check_data(data)

data = utils.convert_categorical_vars(data)
AutoScore_binary.compute_uni_variable_table(data)
AutoScore_binary.compute_multi_variable_table(data)
train_set, validation_set, test_set = utils.split_data(data, (0.7, 0.1, 0.2), cross_validation=False, strat_by_label=False)

rank = AutoScore_binary.AutoScore_rank(train_set, validation_set, method='rf')
# rank = AutoScore_binary.AutoScore_rank(train_set, validation_set, method='auc')

AUC_df = AutoScore_binary.AutoScore_parsimony(train_set, validation_set, rank, n_min=1, n_max=rank.shape[0])
# AUC_df = AutoScore_binary.AutoScore_parsimony(train_set, validation_set, rank, n_max=rank.shape[0],
#                                               categorize='kmeans', max_cluster=5)

final_variables = ['Age', 'Lab_H', 'Lab_K', 'Lab_B', 'Vital_E', 'Vital_A']
cut_vec = AutoScore_binary.AutoScore_weighting(train_set, validation_set, final_variables, max_score=100,
                                               categorize="quantile", quantiles=(0, 0.05, 0.2, 0.8, 0.95, 1))
print('Initial cut vector:\n', cut_vec)

# Fine-tune cut_vec
cut_vec['Age'] = np.array([50, 75, 90])
cut_vec['Lab_H'] = np.array([0.2, 1, 3, 4])
cut_vec['Lab_K'] = np.array([10, 40])
cut_vec['Lab_B'] = np.array([10, 17])
cut_vec['Vital_A'] = np.array([70, 98])
print('Fine-tuned cut vector:\n', cut_vec)

score_table = AutoScore_binary.AutoScore_fine_tuning(train_set, validation_set, final_variables, cut_vec, max_score=100)

pred_score = AutoScore_binary.AutoScore_testing(test_set, final_variables, cut_vec, score_table)
# pred_score.to_csv('pred_score.csv')
print(pred_score.head())

AutoScore_binary.conversion_table(pred_score, by='risk', values=(0.01, 0.05, 0.1, 0.2, 0.5))
# AutoScore_binary.conversion_table(pred_score=pred_score, by="score", values=(20, 40, 60, 75))
utils.plot_predicted_risk(pred_score, max_score=100)
