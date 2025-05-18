import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# Mapping for categorical features
LOW_QUANTILE = 0.10
HIGH_QUANTILE = 0.90
IRQ_COEFF = 3

def data_preprocessing(ds):
    # Remove unnecessary columns
    important_df =  ds[['Age', 'Breed1', 'Breed2',
                        'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize', 'Vaccinated',
                        'FurLength', 'Dewormed', 'Sterilized', 'Health',
                        'Quantity', 'Fee', 'State', 'AdoptionSpeed']]
    
    # Categorical features
    important_df = important_df.drop(columns=['Color3', 'Fee', 'State'])

    # Handling Nans
    missing = list()
    for x in important_df.columns:
        if important_df[x].isnull().sum() != 0:
            missing.append(x)

    # Replacing anomalies
    def replace_anomalies_with_minmax_values(df, column):
        sorted_values = df[column].sort_values()
        Q1 = sorted_values.quantile(LOW_QUANTILE)
        Q3 = sorted_values.quantile(HIGH_QUANTILE)
        IQR = Q3 - Q1
        lower_bound = Q1 - IRQ_COEFF * IQR
        upper_bound = Q3 + IRQ_COEFF * IQR

        min_real_value = sorted_values[sorted_values >= lower_bound].min()
        max_real_value = sorted_values[sorted_values <= upper_bound].max()

        df.loc[df[column] < lower_bound, column] = min_real_value
        df.loc[df[column] > upper_bound, column] = max_real_value

    numerical_columns = ['Age', 'Breed1', 'Breed2',
                        'Gender', 'Color1', 'Color2', 'MaturitySize', 'Vaccinated',
                        'FurLength', 'Dewormed', 'Sterilized', 'Health',
                        'Quantity', 'AdoptionSpeed']

    for col in numerical_columns:
        replace_anomalies_with_minmax_values(important_df, col)

    return important_df


def split_data(input_filename, train_filename, val_filename):
    df = data_preprocessing(pd.read_csv(input_filename))

    mean_adoption_speed = df['AdoptionSpeed'].mean()
    print(f"mean_adoption_speed: {mean_adoption_speed:.4f}")

    df['AdoptionSpeedBinary'] = (df['AdoptionSpeed'] > mean_adoption_speed).astype(int)

    X = df.drop(columns=['AdoptionSpeedBinary'])
    y = df['AdoptionSpeedBinary']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    train_df = X_train.copy()
    train_df['AdoptionSpeedBinary'] = y_train

    val_df = X_val.copy()
    val_df['AdoptionSpeedBinary'] = y_val

    train_df.to_csv(train_filename, index=False)
    val_df.to_csv(val_filename, index=False)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python data_split.py <input_file> <train_file> <val_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    train_file = sys.argv[2]
    val_file = sys.argv[3]
    split_data(input_file, train_file, val_file)
