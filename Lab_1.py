import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
df = df.drop(columns=['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'time', 'DEATH_EVENT'])

n_bins = 20
fig, axs = plt.subplots(2, 3)
fig.tight_layout()

axs[0, 0].hist(df['age'].values, bins=n_bins)
axs[0, 0].set_title('age')

axs[0, 1].hist(df['creatinine_phosphokinase'].values, bins=n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')

axs[0, 2].hist(df['ejection_fraction'].values, bins=n_bins)
axs[0, 2].set_title('ejection_fraction')

axs[1, 0].hist(df['platelets'].values, bins=n_bins)
axs[1, 0].set_title('platelets')

axs[1, 1].hist(df['serum_creatinine'].values, bins=n_bins)
axs[1, 1].set_title('serum_creatinine')

axs[1, 2].hist(df['serum_sodium'].values, bins=n_bins)
axs[1, 2].set_title('serum_sodium')

plt.show()

def print_range_average(column_name: str) -> str:
    column_range = min(df[column_name].values), max(df[column_name].values)
    column_average_value = df[column_name].mode()[0]

    return f'Для {column_name} - {column_range}, max наблюдений: ~ {column_average_value}'


for i in df.columns:
    print(print_range_average(i))
    
def print_range_average(column_name: str) -> str:
    column_range = min(df[column_name].values), max(df[column_name].values)
    column_average_value = df[column_name].mode()[0]

    return f'Для {column_name} - {column_range}, max наблюдений: ~ {column_average_value}'


for i in df.columns:
    print(print_range_average(i))
 
dictionary_get = {'age': 0, 'creatinine_phosphokinase': 1, 'ejection_fraction': 2, 'platelets': 3,
                  'serum_creatinine': 4, 'serum_sodium': 5}


def mathematical_expectation(column_name: str) -> None:
    first_df_mathematical_expectation = np.mean(df[column_name])
    second_df_mathematical_expectation = np.mean(data_scaled[:, dictionary_get[column_name]])

    first_df_standard_deviation = np.std(df[column_name])
    second_df_standard_deviation = np.std(data_scaled[:, dictionary_get[column_name]])
    print(
        f'Мат. ожидание "{column_name}" до стандартизации: {first_df_mathematical_expectation}, после: {second_df_mathematical_expectation}')
    print(
        f'СКО "{column_name}" до стандартизации: {first_df_standard_deviation}, после: {second_df_standard_deviation}')
    print()
  
for i in df.columns:
    mathematical_expectation(i)
    
print(f'mean_:\n{scaler.mean_}')
print(f'var_:\n{scaler.var_}')

scaler_all = preprocessing.StandardScaler().fit(data)
data_scaled_all = scaler_all.transform(data)

print(f'mean_:\n{scaler_all.mean_}')
print(f'var_:\n{scaler_all.var_}')

min_max_scaler = preprocessing.MinMaxScaler().fit(data)
data_min_max_scaled = min_max_scaler.transform(data)
     
fig, axs = plt.subplots(2, 3)
fig.tight_layout()

axs[0, 0].hist(data_min_max_scaled[:, 0], bins=n_bins)
axs[0, 0].set_title('age')

axs[0, 1].hist(data_min_max_scaled[:, 1], bins=n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')

axs[0, 2].hist(data_min_max_scaled[:, 2], bins=n_bins)
axs[0, 2].set_title('ejection_fraction')

axs[1, 0].hist(data_min_max_scaled[:, 3], bins=n_bins)
axs[1, 0].set_title('platelets')

axs[1, 1].hist(data_min_max_scaled[:, 4], bins=n_bins)
axs[1, 1].set_title('serum_creatinine')

axs[1, 2].hist(data_min_max_scaled[:, 5], bins=n_bins)
axs[1, 2].set_title('serum_sodium')

plt.show()

print(f'min: {min_max_scaler.data_min_}')
print(f'max: {min_max_scaler.data_max_}')

max_abs_scaler = preprocessing.MaxAbsScaler().fit(data)
data_max_abs_scaled = max_abs_scaler.transform(data)

fig, axs = plt.subplots(2, 3)
fig.tight_layout()

axs[0, 0].hist(data_max_abs_scaled[:, 0], bins=n_bins)
axs[0, 0].set_title('age')

axs[0, 1].hist(data_max_abs_scaled[:, 1], bins=n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')

axs[0, 2].hist(data_max_abs_scaled[:, 2], bins=n_bins)
axs[0, 2].set_title('ejection_fraction')

axs[1, 0].hist(data_max_abs_scaled[:, 3], bins=n_bins)
axs[1, 0].set_title('platelets')

axs[1, 1].hist(data_max_abs_scaled[:, 4], bins=n_bins)
axs[1, 1].set_title('serum_creatinine')

axs[1, 2].hist(data_max_abs_scaled[:, 5], bins=n_bins)
axs[1, 2].set_title('serum_sodium')

plt.show()

robust_scaler = preprocessing.RobustScaler().fit(data)
robust_scaled = max_abs_scaler.transform(data)

fig, axs = plt.subplots(2, 3)
fig.tight_layout()

axs[0, 0].hist(robust_scaled[:, 0], bins=n_bins)
axs[0, 0].set_title('age')

axs[0, 1].hist(robust_scaled[:, 1], bins=n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')

axs[0, 2].hist(robust_scaled[:, 2], bins=n_bins)
axs[0, 2].set_title('ejection_fraction')

axs[1, 0].hist(robust_scaled[:, 3], bins=n_bins)
axs[1, 0].set_title('platelets')

axs[1, 1].hist(robust_scaled[:, 4], bins=n_bins)
axs[1, 1].set_title('serum_creatinine')

axs[1, 2].hist(robust_scaled[:, 5], bins=n_bins)
axs[1, 2].set_title('serum_sodium')

plt.show()

my_scaler = preprocessing.MinMaxScaler(feature_range=(-5, 10)).fit(data)
my_scaled = my_scaler.transform(data)

fig, axs = plt.subplots(2, 3)
fig.tight_layout()

axs[0, 0].hist(my_scaled[:, 0], bins=n_bins)
axs[0, 0].set_title('age')

axs[0, 1].hist(my_scaled[:, 1], bins=n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')

axs[0, 2].hist(my_scaled[:, 2], bins=n_bins)
axs[0, 2].set_title('ejection_fraction')

axs[1, 0].hist(my_scaled[:, 3], bins=n_bins)
axs[1, 0].set_title('platelets')

axs[1, 1].hist(my_scaled[:, 4], bins=n_bins)
axs[1, 1].set_title('serum_creatinine')

axs[1, 2].hist(my_scaled[:, 5], bins=n_bins)
axs[1, 2].set_title('serum_sodium')

plt.show()

quantile_transformer = preprocessing.QuantileTransformer(n_quantiles=100, random_state=0).fit(data)
data_quantile_scaled = quantile_transformer.transform(data)
     
fig, axs = plt.subplots(2, 3)
fig.tight_layout()

axs[0, 0].hist(data_quantile_scaled[:, 0], bins=n_bins)
axs[0, 0].set_title('age')

axs[0, 1].hist(data_quantile_scaled[:, 1], bins=n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')

axs[0, 2].hist(data_quantile_scaled[:, 2], bins=n_bins)
axs[0, 2].set_title('ejection_fraction')

axs[1, 0].hist(data_quantile_scaled[:, 3], bins=n_bins)
axs[1, 0].set_title('platelets')

axs[1, 1].hist(data_quantile_scaled[:, 4], bins=n_bins)
axs[1, 1].set_title('serum_creatinine')

axs[1, 2].hist(data_quantile_scaled[:, 5], bins=n_bins)
axs[1, 2].set_title('serum_sodium')

plt.show()

quantile_transformer_output = preprocessing.QuantileTransformer(n_quantiles=100, output_distribution='normal').fit(data)
data_quantile_scaled_output = quantile_transformer_output.transform(data)
     
fig, axs = plt.subplots(2, 3)
fig.tight_layout()

axs[0, 0].hist(data_quantile_scaled_output[:, 0], bins=n_bins)
axs[0, 0].set_title('age')

axs[0, 1].hist(data_quantile_scaled_output[:, 1], bins=n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')

axs[0, 2].hist(data_quantile_scaled_output[:, 2], bins=n_bins)
axs[0, 2].set_title('ejection_fraction')

axs[1, 0].hist(data_quantile_scaled_output[:, 3], bins=n_bins)
axs[1, 0].set_title('platelets')

axs[1, 1].hist(data_quantile_scaled_output[:, 4], bins=n_bins)
axs[1, 1].set_title('serum_creatinine')

axs[1, 2].hist(data_quantile_scaled_output[:, 5], bins=n_bins)
axs[1, 2].set_title('serum_sodium')

plt.show()

transformer_power_scale = preprocessing.PowerTransformer().fit(data)
scaled_power_scaled = transformer_power_scale.transform(data)
     

fig, axs = plt.subplots(2, 3)
fig.tight_layout()

axs[0, 0].hist(scaled_power_scaled[:, 0], bins=n_bins)
axs[0, 0].set_title('age')

axs[0, 1].hist(scaled_power_scaled[:, 1], bins=n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')

axs[0, 2].hist(scaled_power_scaled[:, 2], bins=n_bins)
axs[0, 2].set_title('ejection_fraction')

axs[1, 0].hist(scaled_power_scaled[:, 3], bins=n_bins)
axs[1, 0].set_title('platelets')

axs[1, 1].hist(scaled_power_scaled[:, 4], bins=n_bins)
axs[1, 1].set_title('serum_creatinine')

axs[1, 2].hist(scaled_power_scaled[:, 5], bins=n_bins)
axs[1, 2].set_title('serum_sodium')

plt.show()

discretizer_scale = preprocessing.KBinsDiscretizer(n_bins=[3, 4, 3, 10, 2, 4], encode='ordinal', strategy='quantile')
discretizer_scaled = discretizer_scale.fit_transform(data)
     

fig, axs = plt.subplots(2, 3)
fig.tight_layout()

axs[0, 0].hist(discretizer_scaled[:, 0], bins=n_bins)
axs[0, 0].set_title('age')

axs[0, 1].hist(discretizer_scaled[:, 1], bins=n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')

axs[0, 2].hist(discretizer_scaled[:, 2], bins=n_bins)
axs[0, 2].set_title('ejection_fraction')

axs[1, 0].hist(discretizer_scaled[:, 3], bins=n_bins)
axs[1, 0].set_title('platelets')

axs[1, 1].hist(discretizer_scaled[:, 4], bins=n_bins)
axs[1, 1].set_title('serum_creatinine')

axs[1, 2].hist(discretizer_scaled[:, 5], bins=n_bins)
axs[1, 2].set_title('serum_sodium')

plt.show()
