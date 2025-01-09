import pandas as pd

if __name__ == "__main__":
    first = pd.read_csv("/input/0/cohort_data.csv")
    second = pd.read_csv("/input/1/cohort_data.csv")
    third = pd.read_csv("/input/2/cohort_data.csv")
    merged = pd.concat([first, second, third], ignore_index=True, axis=0)
    merged.to_csv("/output/0/cohort_data.csv", index=False)
