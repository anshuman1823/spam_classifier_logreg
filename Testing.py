############################## Importing dependencies ##########################

## The cwd path should be the path to the folder in which the .py file containing code is present
## All other supporting functions are also present in the same directory

import os
from Email2Text import email_to_text
import pickle

import pandas as pd

#**********************************************************************************

def testing():

    ## loading the logreg model from saved pickle file
    if not os.path.exists(os.path.join(os.getcwd(), "logreg_c.pkl")):
        raise ValueError("Logistic regression pickle file not found")
    else:
        with open(os.path.join(os.getcwd(), "logreg_c.pkl"), "rb") as file:
            logreg_c = pickle.load(file=file)
    
    ## loading the fitted cv from saved pickle file
    if not os.path.exists(os.path.join(os.getcwd(), "cv.pkl")):
        raise ValueError("CV pickle file not found")
    else:
        with open(os.path.join(os.getcwd(), "cv.pkl"), "rb") as file:
            cv = pickle.load(file=file)
    
    ## Testing by loading data files in the test folder
    test_path = os.path.join(os.getcwd(), "test")
    test_files = [os.path.join(test_path, i) for i in os.listdir(test_path) if i.endswith(".txt")]

    ## Remove preds.csv file from created any previous test run
    try:
        os.remove(os.path.join(test_path, "preds.csv"))
    except Exception as e:
        pass
    
    data = []
    # Process the emails
    for file in test_files:
        with open(file, mode="rb") as f:
            content = f.read().decode("utf-8", errors="ignore")
            data.append({"email": content})

    df_test = pd.DataFrame(data)
    print(f"No. of emails in test folder: {len(df_test)}")

    ## Testing Transformation steps
    ## extracting info from the emails
    df_test["content"] = df_test["email"].apply(lambda x: email_to_text(x))
    X_test = df_test["content"]
    X_test = cv.transform(X_test)

    preds = logreg_c.predict(X_test)  # making predictions

    # saving predictions in preds.csv in the test folder
    pred_list = [{"filename": tf.split(os.sep)[-1], "pred": i} for i,tf in zip(preds.ravel(), test_files)]
    df_preds = pd.DataFrame(pred_list)
    df_preds.to_csv(os.path.join(test_path, "preds.csv"), index = False)

print("Testing.py file execution started")
testing()