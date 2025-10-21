# scripts/featurize.py
import argparse
import pandas as pd

def build_parser():
    p = argparse.ArgumentParser(description="Feature engineering for Titanic")
    p.add_argument("--input", required=True, help="Path to cleaned CSV")
    p.add_argument("--output", required=True, help="Path to save features CSV")
    return p

def family_size(number):
    if number == 1:
        return "Alone"
    elif 1 < number < 5:
        return "Small"
    else:
        return "Large"

def main():
    args = build_parser().parse_args()
    
    df = pd.read_csv(args.input)
    
    # Fill missing Age by median grouped by Sex & Pclass
    df['Age'] = df.groupby(['Sex','Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
    
    # Titles
    df['Title'] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    df['Title'] = df['Title'].replace(['Lady', 'the Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle','Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme','Mrs')
    
    # Family size
    df['Family_size'] = df['SibSp'] + df['Parch'] + 1
    df['Family_size'] = df['Family_size'].apply(family_size)
    # New feature: IsAlone
    df['IsAlone'] = df['Family_size'].apply(lambda x: 1 if x == 'Alone' else 0)

    
    
    # Drop original columns no longer needed
    df.drop(columns=['Name','Parch','SibSp','Ticket'], inplace=True)
    
    # Convert Age to int
    df['Age'] = df['Age'].astype('int64')
    
    df.to_csv(args.output, index=False)
    print(f"Features saved to {args.output}")

if __name__ == "__main__":
    main()
