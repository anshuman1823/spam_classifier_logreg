############################## Importing dependencies ##########################

## This cwd path should be the path to the folder in which the .py file containing code is present
## All other supporting functions are also present in the same directory

import os
from Email2Text import email_to_text
import pandas as pd

#**********************************************************************************

base_path = os.getcwd()
dataset_path = os.path.join(base_path, "dataset")
ham_path = os.path.join(dataset_path, "ham")
spam_path = os.path.join(dataset_path, "spam")
ham_files = [os.path.join(ham_path,i) for i in os.listdir(ham_path)]
spam_files = [os.path.join(spam_path,i) for i in os.listdir(spam_path)]
df = pd.DataFrame(columns = ["email", "label"])

data = [] # Initialize an empty list to store data temporarily

# Process ham emails
for file in ham_files:
    with open(file, mode="rb") as f:
        content = f.read().decode("utf-8", errors="ignore")
        data.append({"email": content, "label": 0})

# Process spam emails
for file in spam_files:
    with open(file, mode="rb") as f:
        content = f.read().decode("utf-8", errors="ignore")
        data.append({"email": content, "label": 1})

# Create a DataFrame from the accumulated data
df = pd.DataFrame(data)

print(f"total number of emails loaded: {len(df)}")
print(f"spam emails loaded: {len(spam_files)}")
print(f"ham emails loaded: {len(ham_files)}")


df["content"] = df["email"].apply(lambda x: email_to_text(x)) ## applying all the pre-processing functions to process raw emails
df.to_csv(os.path.join(os.getcwd(),"processed_emails.csv"), columns = ["content", "label"])  ## exporting pre-processed emails to load later