import pandas as pd
import os

# Define paths for each dataset
file_paths = {
    "person": "input/0/dataset.csv",
    "visit_occurrence": "input/1/dataset.csv",
    "condition_occurrence": "input/2/dataset.csv",
    "procedure_occurrence": "input/3/dataset.csv",
    "drug_exposure": "input/4/dataset.csv"
}

# Load each CSV file into a dictionary of DataFrames
tables = {name: pd.read_csv(path) for name, path in file_paths.items()}

# Define quality check functions
def check_missing_values(df):
    """Check for missing values in each column of the DataFrame."""
    missing_data = df.isnull().sum()
    return missing_data[missing_data > 0]

def check_unique_key(df, key_column):
    """Check if the primary key column has unique values."""
    duplicates = df[df.duplicated(subset=[key_column])]
    return duplicates

def check_foreign_key(df, foreign_key, ref_df, ref_column):
    """Check if foreign keys in a table match the primary keys in the referenced DataFrame."""
    missing_foreign_keys = df[~df[foreign_key].isin(ref_df[ref_column])]
    return missing_foreign_keys

def check_date_order(df, start_date, end_date):
    """Check if start dates are before or equal to end dates in the DataFrame."""
    date_issues = df[df[start_date] > df[end_date]]
    return date_issues

# Perform data quality checks on each table
quality_checks = {}

# person table checks
person_df = tables["person"]
quality_checks['person'] = {
    "missing_values": check_missing_values(person_df),
    "unique_key": check_unique_key(person_df, "person_id"),
}

# visit_occurrence table checks
visit_occurrence_df = tables["visit_occurrence"]
quality_checks['visit_occurrence'] = {
    "missing_values": check_missing_values(visit_occurrence_df),
    "unique_key": check_unique_key(visit_occurrence_df, "visit_occurrence_id"),
    "foreign_key_person": check_foreign_key(visit_occurrence_df, "person_id", person_df, "person_id"),
    "date_order": check_date_order(visit_occurrence_df, "visit_start_date", "visit_end_date"),
}

# condition_occurrence table checks
condition_occurrence_df = tables["condition_occurrence"]
quality_checks['condition_occurrence'] = {
    "missing_values": check_missing_values(condition_occurrence_df),
    "unique_key": check_unique_key(condition_occurrence_df, "condition_occurrence_id"),
    "foreign_key_person": check_foreign_key(condition_occurrence_df, "person_id", person_df, "person_id"),
    "foreign_key_visit": check_foreign_key(condition_occurrence_df, "visit_occurrence_id", visit_occurrence_df, "visit_occurrence_id"),
    "date_order": check_date_order(condition_occurrence_df, "condition_start_date", "condition_end_date"),
}

# procedure_occurrence table checks
procedure_occurrence_df = tables["procedure_occurrence"]
quality_checks['procedure_occurrence'] = {
    "missing_values": check_missing_values(procedure_occurrence_df),
    "unique_key": check_unique_key(procedure_occurrence_df, "procedure_occurrence_id"),
    "foreign_key_person": check_foreign_key(procedure_occurrence_df, "person_id", person_df, "person_id"),
    "foreign_key_visit": check_foreign_key(procedure_occurrence_df, "visit_occurrence_id", visit_occurrence_df, "visit_occurrence_id"),
    "date_order": check_date_order(procedure_occurrence_df, "procedure_date", "procedure_date"),  # For same-day check
}

# drug_exposure table checks
drug_exposure_df = tables["drug_exposure"]
quality_checks['drug_exposure'] = {
    "missing_values": check_missing_values(drug_exposure_df),
    "unique_key": check_unique_key(drug_exposure_df, "drug_exposure_id"),
    "foreign_key_person": check_foreign_key(drug_exposure_df, "person_id", person_df, "person_id"),
    "foreign_key_visit": check_foreign_key(drug_exposure_df, "visit_occurrence_id", visit_occurrence_df, "visit_occurrence_id"),
    "date_order": check_date_order(drug_exposure_df, "drug_exposure_start_date", "drug_exposure_end_date"),
}

# Print the quality check results
for table, checks in quality_checks.items():
    print(f"\nData Quality Checks for {table} table:")
    for check, result in checks.items():
        print(f"\n{check.capitalize()}:")
        if result.empty:
            print("No issues found.")
        else:
            print(result)
