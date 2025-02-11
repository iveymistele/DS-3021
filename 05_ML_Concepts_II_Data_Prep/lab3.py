import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# Dataset 1: College Completion Data
# Brainstorming Q: How does academic recognition/achievement vary across private and public colleges/universities?
# Independent Business Metric: awards_per_value (number of awards per 100 undergrad students)

# difference between metric and target var: ex on job metric would be job or not TV could be if enrollment went up (intermediate variable)
# look for larger problem
#
#  data preparation for whole dataset (typically you don't drop columns)

# 1 : correct variable types
college_data = pd.read_csv("/workspaces/DS-3021/data/cc_institution_details.csv")
# print(college_data.head())

# from kaggle dictionary, variables to be changed to categorical: 
# level , control 
cols = ["level","control"]
college_data[cols]= college_data[cols].astype('category')

# check
# print(college_data["level"].dtype) 
# print(college_data["control"].dtype)

# collapse factor levels as needed 
# cat_cols = college_data.select_dtypes(include=['category']).columns.tolist()
# print(cat_cols)  # output level, control
# only two categorical variables and each is only two options
# so not needed

# one-hot encoding
category_list = list(college_data.select_dtypes('category'))
college_data1 = pd.get_dummies(college_data, columns = category_list)

# normalize continuous variables
# analyze numeric columns 
# potential_continuous_vars = college_data1.select_dtypes(include=['int64', 'float64'])
# print(potential_continuous_vars)
# 51 columns 
# from looking at dataset dictionary, 
# continuous variables appear to be only variables with type float
continuous_vars = ["long_x", "lat_y", "awards_per_value", "awards_per_state_value","awards_per_natl_value","exp_award_value","exp_award_state_value","exp_award_natl_value","exp_award_percentile","ft_pct","fte_value","fte_percentile"]

scaler = MinMaxScaler()
normalized_vars = scaler.fit_transform(college_data1[continuous_vars])
college_data1[continuous_vars] = normalized_vars
# print(college_data1[continuous_vars])

# drop unneeded variables 

columns_to_drop = [
    'chronname', 'city', 'state', 'site', 'nicknames',  # Location and name
    'basic', 'hbcu', 'flagship', 'similar',  
    'vsa_grad_elsewhere_after6_first', 'vsa_enroll_after6_first', 'vsa_enroll_elsewhere_after6_first', 
    'vsa_grad_after4_transfer', 'vsa_grad_elsewhere_after4_transfer', 'vsa_enroll_after4_transfer', 
    'vsa_enroll_elsewhere_after4_transfer', 'vsa_grad_after6_transfer', 'vsa_grad_elsewhere_after6_transfer', 
    'vsa_enroll_after6_transfer', 'vsa_enroll_elsewhere_after6_transfer',  
    'cohort_size',  
    'state_sector_ct', 'carnegie_ct'  
]
college_data1 = college_data1.drop(columns=columns_to_drop)
# print(college_data1.columns)

# create target variable with awards_per_value
# i am making a threshold of low achievement/high achievement 

# add new column classifying by target var
threshold = college_data1['awards_per_value'].median()
college_data1['target'] = (college_data1['awards_per_value'] > threshold).astype(int)
# print(college_data1['target'].value_counts())

# target variable prevalence
target_prevalence = college_data1['target'].value_counts(normalize=True)
# print(target_prevalence)
# 0 0.501 and 1 0.499 shows that they are pretty equal 

# train,tune,test 
Train, Remaining = train_test_split(college_data1,  train_size = 55, stratify = college_data1['target'])

Tune, Test = train_test_split(Remaining,  train_size = .5, stratify= Remaining['target'])

# combine into a single function
def college_data_pipeline(college_data):
    # Convert to categorical types
    cols = ["level", "control"]
    college_data[cols] = college_data[cols].astype('category')

    # One-hot encoding
    category_list = list(college_data.select_dtypes('category'))
    college_data1 = pd.get_dummies(college_data, columns=category_list)

    # Normalize continuous variables
    continuous_vars = ["long_x", "lat_y", "awards_per_value", "awards_per_state_value", "awards_per_natl_value",
                       "exp_award_value", "exp_award_state_value", "exp_award_natl_value", "exp_award_percentile",
                       "ft_pct", "fte_value", "fte_percentile"]
    scaler = MinMaxScaler()
    normalized_vars = scaler.fit_transform(college_data1[continuous_vars])
    college_data1[continuous_vars] = normalized_vars

    # Drop columns
    columns_to_drop = [
        'chronname', 'city', 'state', 'site', 'nicknames',  # Location and name
        'basic', 'hbcu', 'flagship', 'similar',  
        'vsa_grad_elsewhere_after6_first', 'vsa_enroll_after6_first', 'vsa_enroll_elsewhere_after6_first', 
        'vsa_grad_after4_transfer', 'vsa_grad_elsewhere_after4_transfer', 'vsa_enroll_after4_transfer', 
        'vsa_enroll_elsewhere_after4_transfer', 'vsa_grad_after6_transfer', 'vsa_grad_elsewhere_after6_transfer', 
        'vsa_enroll_after6_transfer', 'vsa_enroll_elsewhere_after6_transfer',  
        'cohort_size',  
        'state_sector_ct', 'carnegie_ct'
    ]
    college_data1 = college_data1.drop(columns=columns_to_drop)

    # target var
    threshold = college_data1['awards_per_value'].median()
    college_data1['target'] = (college_data1['awards_per_value'] > threshold).astype(int)

    # prevalence
    target_prevalence = college_data1['target'].value_counts(normalize=True)
    print(target_prevalence)

    # Train tune test
    Train, Remaining = train_test_split(college_data1, train_size=0.55, stratify=college_data1['target'])
    Tune, Test = train_test_split(Remaining, train_size=0.5, stratify=Remaining['target'])

    return Train, Tune, Test, college_data1




# Step 3: Concerns
# The dataset has a wide variety of useful columns that can help address the problem. It has many different metrics on awards,
# which can help speak to the prestige of the institution. Additionally, the prevalence of the target variable is very
# close to even which means the target variable is represented fairly evenly making
# it a good training set. These features can all reveal patterns in relation to the target variable/overall question. 
# The bias to be concerned about is potential confounding variables such as institution size, since private institutions are often smaller than public institutions 
# which could impact the awards statistics. Additionally, location could impact the awards granted based on state funding/policy. 
# There is also more data on private institutions which could impact results. 

# ----------------------------
# Dataset 2: Job Placement Data
# Brainstorming Q: How does gender affect job salary of college graduates?
# Independent business metric: Gender 

placement_data = pd.read_csv("/workspaces/DS-3021/05_ML_Concepts_II_Data_Prep/Placement_data.csv")
# print(placement_data.columns)

# data pipeline

# print(placement_data.dtypes)
# notes: 
# convert to categoric: gender, ssc_b, hsc_b, workex, specialisation, status
# hsc_s change to commerce, science, others (for all not science) - categoric
# same as above for degree_t
# continuous: ssc_p, hsc_p, degree_p, etest_p,mba_p 
# leave as is: salary

# 1: correct variable types
# excluding those which we will drop 
cat_cols = ["gender",   "workex", "specialisation"]
placement_data[cat_cols]=placement_data[cat_cols].astype('category')
#cereal.mfr = (cereal.mfr.apply(lambda x: x if x in top else "Other")).astype('category')

# commented out since ended up dropping hsc_s and degree_t
# top = ['Commerce','Science']
# placement_data['hsc_s'] = (placement_data['hsc_s'].apply(lambda x: x if x in top else "Other")).astype('category')
# print(placement_data['hsc_s'])

# top2 = ['Comm&Mgmt', 'Sci&Tech']
#placement_data['degree_t'] = (placement_data['degree_t'].apply(lambda x: x if x in top2 else "Other")).astype('category')

# o-h-e
placement_data = pd.get_dummies(placement_data, columns = cat_cols)


cont_vars = ["ssc_p", "hsc_p", "degree_p", "etest_p","mba_p"]
norm_vars = scaler.fit_transform(placement_data[cont_vars])
placement_data[cont_vars]= norm_vars
# print(placement_data.columns)

# drop unneeded variables 
drops = [ 'sl_no',   'ssc_b', 'hsc_b',  'hsc_s', 'degree_t', 'status']
placement_data = placement_data.drop(columns = drops)

salary_threshold = placement_data['salary'].median()
#  target variable: 1 for high salary, 0 for low salary
placement_data['targetV'] = (placement_data['salary'] > salary_threshold).astype(int)
prevalence2 = placement_data['targetV'].value_counts(normalize=True)
# print(prevalence2)
# note its not as even as first dataset (bad)

# train-test split 
Train, Test = train_test_split(placement_data, train_size=0.8, stratify=placement_data['targetV'])
Tune, Test = train_test_split(Test, train_size=0.5, stratify=Test['targetV'])

def placement_data_pipeline(placement_data):
    cat_cols = ["gender", "workex", "specialisation"]
    placement_data[cat_cols] = placement_data[cat_cols].astype('category')

    # One-hot encoding
    placement_data = pd.get_dummies(placement_data, columns=cat_cols)

    # Normalize continuous variables
    cont_vars = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]
    scaler = MinMaxScaler()
    norm_vars = scaler.fit_transform(placement_data[cont_vars])
    placement_data[cont_vars] = norm_vars

    # Drop vars
    drops = ['sl_no', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'status']
    placement_data = placement_data.drop(columns=drops)

    # target var
    salary_threshold = placement_data['salary'].median()
    placement_data['targetV'] = (placement_data['salary'] > salary_threshold).astype(int)

    # Prevalence 
    prevalence2 = placement_data['targetV'].value_counts(normalize=True)
    print(prevalence2)
    # Train, Tune, test partitions
    Train, Test = train_test_split(placement_data, train_size=0.8, stratify=placement_data['targetV'])
    Tune, Test = train_test_split(Test, train_size=0.5, stratify=Test['targetV'])

    return Train, Tune, Test, placement_data



# Step 3: Concerns
# I think this data can somewhat address the problem but it may
# not be the best dataset to do so. The prevalence of the target
# variable was not very close to even, which can introduce bias in the
# model. Additionally, the spread of genders (shown below) uneven:

#gender_counts = placement_data[['gender_F', 'gender_M']].sum()
# print(gender_counts)
# gender_F = 76, gender_M = 139
# This shows that there are almost twice as many males 
# in the dataset, which may produce results that are unrepresentative 
# of other datasets. So this may not be the best 
# data to train with. 
