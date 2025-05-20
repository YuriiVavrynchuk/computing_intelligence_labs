import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
import joblib
import sys


def train(input_train_file, output_model_file):
    pipeline = Pipeline([
        ('model', GaussianNB(var_smoothing=1e-12))
    ])

    df = pd.read_csv(input_train_file)

    mean_adoption_speed = df['AdoptionSpeed'].mean()
    print(f"mean_adoption_speed: {mean_adoption_speed:.4f}")

    df['AdoptionSpeedBinary'] = (df['AdoptionSpeed'] > mean_adoption_speed).astype(int)

    X_train = df.drop(columns=['AdoptionSpeedBinary']).drop(columns=['AdoptionSpeed'])
    y_train = df['AdoptionSpeedBinary']
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, output_model_file)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train.py <input_train_file> <output_model_file>")
        sys.exit(1)

    input_train_file = sys.argv[1]
    output_model_file = sys.argv[2]
    train(input_train_file, output_model_file)
