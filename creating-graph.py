import pandas as pd
import itertools
import csv

'''
df = pd.read_csv('/Users/Desktop/SDoH/DIAGNOSES_ICD.csv')
selected_columns = df[['SUBJECT_ID', 'ICD9_CODE']]
print(selected_columns.head())
selected_columns.to_csv('/Users/Desktop/SDoH-2/disease_subject_id.csv', index=False)
'''


'''
df = pd.read_csv('/Users/Desktop/SDoH/PRESCRIPTIONS.csv')
selected_columns = df[['SUBJECT_ID', 'DRUG']]
print(selected_columns.head())
selected_columns.to_csv('/Users/Desktop/SDoH-2/drugs_subject_id.csv', index=False)
'''

'''
df = pd.read_csv('/Users/Desktop/SDoH/SBDH_file_Subject_id.csv')
mapping = {
    'sdoh_community_present': {1: 'True'},
    'sdoh_community_absent': {1: 'True'},
    'sdoh_education': {1: 'True'},
    'sdoh_economics': {1: 'True', 2: 'False'},
    'sdoh_environment': {1: 'True', 2: 'False'},
    'behavior_alcohol': {1: 'Present', 2: 'Past', 3: 'Never', 4: 'Unsure'},
    'behavior_tobacco': {1: 'Present', 2: 'Past', 3: 'Never', 4: 'Unsure'},
    'behavior_drug': {1: 'Present', 2: 'Past', 3: 'Never', 4: 'Unsure'}
}

for column, replace_map in mapping.items():
    df[column] = df[column].replace(replace_map)

df.to_csv('/Users/Desktop/SDoH-2/updated_SBDH_file.csv', index=False)
'''

'''
disease_df = pd.read_csv('/Users/Desktop/SDoH-2/disease_subject_id.csv')
drugs_df = pd.read_csv('/Users/Desktop/SDoH-2/drugs_subject_id.csv')
sbdh_df = pd.read_csv('/Users/Desktop/SDoH-2/updated_SBDH_file.csv')
agg_funcs_drug_disease = lambda x: ';'.join(x.astype(str))


grouped_disease_df = disease_df.groupby('SUBJECT_ID')['ICD9_CODE'].agg(agg_funcs_drug_disease).reset_index()
grouped_drugs_df = drugs_df.groupby('SUBJECT_ID')['DRUG'].agg(agg_funcs_drug_disease).reset_index()
agg_funcs_sdoh = lambda x: ';'.join(set(x.astype(str)))  
grouped_sbdh_df = sbdh_df.groupby('SUBJECT_ID').agg(agg_funcs_sdoh).reset_index()


merge1 = pd.merge(grouped_disease_df, grouped_drugs_df, on='SUBJECT_ID', how='inner')
final_merge = pd.merge(merge1, grouped_sbdh_df, on='SUBJECT_ID', how='inner')
print(final_merge.head())
final_merge.to_csv('/Users/Desktop/SDoH-2/final_merged_data.csv', index=False)
'''

'''
df = pd.read_csv('/Users/Desktop/SDoH-2/final_merged_data.csv')
with open('/Users/Desktop/SDoH-2/drug_disease_sdoh_combinations.csv', mode='w', newline='') as file:
    writer = csv.writer(file)

    writer.writerow(['SUBJECT_ID', 'Combination'])
    for index, row in df.iterrows():
        drugs = row['DRUG'].split(';')
        diseases = row['ICD9_CODE'].split(';')
        sdoh_factors = []
        if row['sdoh_community_present'] != '0':
            sdoh_factors.append(f"community_present: {row['sdoh_community_present']}")
        if row['sdoh_community_absent'] != '0':
            sdoh_factors.append(f"community_absent: {row['sdoh_community_absent']}")
        if row['sdoh_education'] != '0':
            sdoh_factors.append(f"education: {row['sdoh_education']}")
        if row['sdoh_economics'] != '0':
            sdoh_factors.append(f"economics: {row['sdoh_economics']}")
        if row['sdoh_environment'] != '0':
            sdoh_factors.append(f"environment: {row['sdoh_environment']}")
        if row['behavior_alcohol'] != '0':
            sdoh_factors.append(f"alcohol: {row['behavior_alcohol']}")
        if row['behavior_tobacco'] != '0':
            sdoh_factors.append(f"tobacco: {row['behavior_tobacco']}")
        if row['behavior_drug'] != '0':
            sdoh_factors.append(f"drug: {row['behavior_drug']}")

        for combination in itertools.product(drugs, diseases, sdoh_factors):
            writer.writerow([row['SUBJECT_ID'], ', '.join(combination)])
'''




