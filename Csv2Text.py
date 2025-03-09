import pandas as pd
import os

# specifying the .csv dataset path for Nigerian Fraud emails
spam_datasets_path = os.path.join(os.cwd(), "raw_datasets")

df_Nigerian_Fraud = pd.read_csv(os.path.join(spam_datasets_path, "Nigerian_Fraud.csv"))  ##loading the dataset
df_Nigerian_Fraud = df_Nigerian_Fraud.fillna("") ## imputing the NaN values with ""

## combining all the columns into a single email column
df_Nigerian_Fraud["sender"] = "Sender: " + df_Nigerian_Fraud["sender"]
df_Nigerian_Fraud["receiver"] = "Receiver: " + df_Nigerian_Fraud["receiver"]
df_Nigerian_Fraud["date"] = "Date: " + df_Nigerian_Fraud["date"]
df_Nigerian_Fraud["subject"] = "Subject: " + df_Nigerian_Fraud["subject"]
df_Nigerian_Fraud["body"] = "Body: " + "\n" + "\n" +  df_Nigerian_Fraud["body"]
df_Nigerian_Fraud["email"] = df_Nigerian_Fraud["sender"] + "\n" + df_Nigerian_Fraud["receiver"] + "\n" + df_Nigerian_Fraud["date"] + "\n" + df_Nigerian_Fraud["subject"]+ "\n" + df_Nigerian_Fraud["body"]
df_Nigerian_Fraud["email"] = df_Nigerian_Fraud.apply(lambda row: row["email"] + " URL "*int(row["urls"]), axis = 1)

save_path = os.path.join(os.cwd(), "dataset")
ham_path = os.path.join(save_path, "ham")
spam_path = os.path.join(save_path, "spam")

## function to save the emails from pandas dataframe to .txt file
def write_email_to_file(df, ham_path, spam_path):
    """
    Email should be in column named "email"
    label should be in column named "label"
    label: 1 for spam and 0 for ham
    """
    for i, text in enumerate(df.loc[df["label"] == 0, "email"]):
        with open(os.path.join(ham_path, f"nigerian_fraud_ham_{i}.txt"), 'w') as f:
            f.write(text)

    for i, text in enumerate(df.loc[df["label"] == 1, "email"]):
        with open(os.path.join(spam_path, f"nigerian_fraud_spam_{i}.txt"), 'w') as f:
            f.write(text)

write_email_to_file(df_Nigerian_Fraud, ham_path, spam_path)

