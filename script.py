import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt




class Utilities:
    @staticmethod
    def fill_missing_values_by_median(df, col):
        m = df[col].dropna().median()
        #df[col + 'IsNull'] = pd.isnull(df[col]).astype(int)
        df[col].loc[df[col].isnull()] = m

    @staticmethod
    def scaling(df):
        # scaling - normalize
        for col in ['Occupation_Satisfaction'
                    ,'Yearly_IncomeK'
                    ,'Overall_happiness_score'
                    ,'Garden_sqr_meter_per_person_in_residancy_area'
                    ,'Yearly_IncomeK'
                    ,'Avg_Residancy_Altitude'
                    ,'Yearly_ExpensesK'
                    ,'%Time_invested_in_work'
                    ,'%_satisfaction_financial_policy'
                    ,'Last_school_grades'
                    ,'Financial_balance_score_(0-1)'
                    ,'%Of_Household_Income']:
            Utilities.normalize(df, col)

        #scaling - Standardize
        for col in ['Avg_monthly_expense_when_under_age_21'
                    ,'AVG_lottary_expanses'
                    ,'Avg_monthly_expense_on_pets_or_plants'
                    ,'Avg_environmental_importance'
                    ,'Avg_government_satisfaction'
                    ,'Avg_Satisfaction_with_previous_vote'
                    ,'Avg_monthly_household_cost'
                    ,'Phone_minutes_10_years'
                    ,'Avg_size_per_room'
                    ,'Weighted_education_rank'
                    ,'Avg_monthly_income_all_years'
                    ,'Number_of_differnt_parties_voted_for'
                    ,'Political_interest_Total_Score'
                    ,'Number_of_valued_Kneset_members'
                    ,'Avg_education_importance'
                    ,'Num_of_kids_born_last_10_years']:
            Utilities.standardize(df, col)

    @staticmethod
    def normalize(df, col):
        min_max_scalar = preprocessing.MinMaxScaler(feature_range=(-1,1))
        df[col] = min_max_scalar.fit_transform(df[col].values.reshape(-1, 1))

    @staticmethod
    def standardize(df, col):
        df[col] = (df[col] - df[col].mean()) / df[col].std(ddof=0)


    @staticmethod
    def fill_missing_values_by_category(df, col):
        most_frequent_value = df[col].mode()[0]
        df[col] = df[col].fillna(most_frequent_value)

    @staticmethod
    def fix_known_outliers(df):

        df['Avg_monthly_expense_on_pets_or_plants'] = df['Avg_monthly_expense_on_pets_or_plants'].abs()
        df['Avg_monthly_expense_when_under_age_21'] = df['Avg_monthly_expense_when_under_age_21'].abs()
        df['AVG_lottary_expanses'] = df['AVG_lottary_expanses'].abs()

    @staticmethod
    def fill_empty_values_by_known_attribute_dependencies(df):
        df['Avg_monthly_expense_when_under_age_21'].values[df['Avg_monthly_expense_when_under_age_21'].isnull()] = \
        3 * df['Avg_monthly_expense_on_pets_or_plants'].values[[df['Avg_monthly_expense_when_under_age_21'].isnull()]]

        df['Avg_monthly_expense_on_pets_or_plants'].values[df['Avg_monthly_expense_on_pets_or_plants'].isnull()] = \
        df['Avg_monthly_expense_when_under_age_21'].values[[df['Avg_monthly_expense_on_pets_or_plants'].isnull()]] / 3

        df['Phone_minutes_10_years'].values[df['Phone_minutes_10_years'].isnull()] = \
        df['Avg_environmental_importance'].values[[df['Phone_minutes_10_years'].isnull()]] ** 2

        df['Avg_environmental_importance'].values[df['Avg_environmental_importance'].isnull()] = \
        df['Phone_minutes_10_years'].apply(np.sqrt).values[[df['Avg_environmental_importance'].isnull()]]

        mapping = {"Education": 100,"Environment": 90,"Healthcare": 80,"Social": 70,"Financial": 60,"Other": 50,"Foreign_Affairs": 40,"Military": 30}
        for val in mapping.keys():
            indexes = (df['Last_school_grades'].isnull()) & (df['Most_Important_Issue'] == val)
            df['Last_school_grades'].values[indexes] = \
            df["Most_Important_Issue"].replace(mapping, inplace=False).values[indexes]

        mapping = {v: k for k, v in mapping.items()}
        for val in mapping.keys():
            indexes = (df['Most_Important_Issue'].isnull()) & (df['Last_school_grades'] == val)
            df['Most_Important_Issue'].values[indexes] = \
            df["Last_school_grades"].replace(mapping, inplace=False).values[indexes]

        df['Avg_monthly_income_all_years'].values[df['Avg_monthly_income_all_years'].isnull()] = \
        df['Avg_monthly_expense_on_pets_or_plants'].values[[df['Avg_monthly_income_all_years'].isnull()]] ** 2

        df['Avg_monthly_expense_on_pets_or_plants'].values[df['Avg_monthly_expense_on_pets_or_plants'].isnull()] = \
        df['Avg_monthly_income_all_years'].apply(np.sqrt).values[[df['Avg_monthly_expense_on_pets_or_plants'].isnull()]]

        df['Avg_monthly_income_all_years'].values[df['Avg_monthly_income_all_years'].isnull()] = \
        9 * df['Avg_monthly_expense_when_under_age_21'].values[[df['Avg_monthly_income_all_years'].isnull()]] ** 2

        df['Avg_monthly_expense_when_under_age_21'].values[df['Avg_monthly_expense_when_under_age_21'].isnull()] = \
        df['Avg_monthly_income_all_years'].apply(np.sqrt).values[[df['Avg_monthly_expense_when_under_age_21'].isnull()]] / 3

        df['Avg_government_satisfaction'].values[df['Avg_government_satisfaction'].isnull()] = \
        df['Avg_size_per_room'].values[[df['Avg_government_satisfaction'].isnull()]] ** 2

        df['Avg_size_per_room'].values[df['Avg_size_per_room'].isnull()] = \
        df['Avg_government_satisfaction'].apply(np.sqrt).values[[df['Avg_size_per_room'].isnull()]]

    @staticmethod
    def prepare_data_set(df):
        # Fix outliers
        Utilities.fix_known_outliers(df)

        #fill Missing value by known attribute relations
        Utilities.fill_empty_values_by_known_attribute_dependencies(df)

        # Fill in missing values
        for col in df.columns:
            if df[col].isnull().sum().sum() > 0:
                if df[col].dtype.name != 'object':
                    Utilities.fill_missing_values_by_median(df, col)
                else:
                    Utilities.fill_missing_values_by_category(df, col)

        # Cast as category
        for col in ['Most_Important_Issue', 'Main_transportation', 'Occupation']:
            df[col] = df[col].astype('category')

        # Cast yes/maybe/no as int
        for col in ['Looking_at_poles_results', 'Will_vote_only_large_party', 'Married', 'Financial_agenda_matters']:
            df[col] = df[col].map({'Yes': 1, "Maybe": 0, 'No': -1}).astype(int)

        # Cast as integer category - there is a meaning for the number
        df['Age_group'] = df['Age_group'].map( {'Below_30':1, '30-45':2, '45_and_up': 3}).astype(int)
        df['Voting_Time'] = df['Voting_Time'].map( {'By_16:00':1, 'After_16:00':2}).astype(int)
        df['Gender'] = df['Gender'].map( {'Male':1, 'Female':2}).astype(int)

        Utilities.scaling(df)

        return pd.get_dummies(df)


# Load origin file
df = pd.read_csv('ElectionsData.csv')



df['Vote'] = df['Vote'].astype('category')
X, X_test, y, y_test = train_test_split(df.loc[:, df.columns != 'Vote'], df['Vote'], test_size=0.1)
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.15)

# store the raw sets
X_train['Vote'] = y_train
X_train.to_csv('train.csv', index=False, encoding='utf-8')
X_validation['Vote'] = y_validation
X_validation.to_csv('validation.csv', index=False, encoding='utf-8')
X_test['Vote'] = y_test
X_test.to_csv('test.csv', index=False, encoding='utf-8')

# delete the label again
X_train = X_train.drop(columns=['Vote'])
X_validation = X_validation.drop(columns=['Vote'])
X_test = X_test.drop(columns=['Vote'])

all = Utilities.prepare_data_set(df.loc[:, df.columns != 'Vote'])
all['Vote'] = df['Vote']
all.to_csv('all.csv', index=False, encoding='utf-8')

#Prepare data
X_train = Utilities.prepare_data_set(X_train.copy())
X_validation = Utilities.prepare_data_set(X_validation.copy())
X_test = Utilities.prepare_data_set(X_test.copy())


necessaries_features = ['Avg_environmental_importance', 'Avg_government_satisfaction', 'Avg_education_importance',
                        'Most_Important_Issue_Social', 'Most_Important_Issue_Other','Most_Important_Issue_Military',
                        'Most_Important_Issue_Healthcare','Most_Important_Issue_Foreign_Affairs',
                        'Most_Important_Issue_Financial',
                        'Most_Important_Issue_Environment', 'Most_Important_Issue_Education',
                        'Avg_monthly_expense_on_pets_or_plants', 'Avg_Residancy_Altitude',
                        'Yearly_ExpensesK', 'Weighted_education_rank', 'Number_of_valued_Kneset_members']

X_train = X_train[necessaries_features]
X_train['Vote'] = y_train
X_validation = X_validation[necessaries_features]
X_validation['Vote'] = y_validation
X_test = X_test[necessaries_features]
X_test['Vote'] = y_test

# store the prepared sets
X_train.to_csv('prepared_train.csv', index=False, encoding='utf-8')
X_validation.to_csv('prepared_validation.csv', index=False, encoding='utf-8')
X_test.to_csv('prepared_test.csv', index=False, encoding='utf-8')

print('end')