import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
	# Read cohort from /input
	df_cohort = pd.read_csv('/input/0/cohort_data.csv')

	# Split to train and test sets
	df_train, df_test = train_test_split(df_cohort, test_size=0.33)

	# Write cohorts to /output
	df_train.to_csv('/output/0/cohort_data.csv', index=False)
	df_test.to_csv('/output/1/cohort_data.csv', index=False)

