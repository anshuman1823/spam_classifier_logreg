Instructions for running and testing the model:

1. The testing process has only two steps:
    1. Place the emails to be tested in the `test` folder inside the working directory.
    2. Run the [`Testing.py`] file.
2. All the source code is written in Python and give in .py file.
3. Make sure that `nltk` library is installed in the environment in which you are running the code.
4. While running any source code file, the working directory should be the folder in which all source code files are present. So, change your working directory if that is not the case.
5. Do not change the name or location of any sub-file or sub-folder.
6. Due to size restriction of moodle (150 MB), the files present in the `dataset` and `raw_datasets` folders have not been added to the final zip file being uploaded on moodle. The google drive links for these two files are as follows:
    
    `dataset` : https://drive.google.com/file/d/1xYYXzL8A5bt2Rj4m7GGQZWzQCm974pP5/view?usp=sharing
    
    `raw_datasets`: https://drive.google.com/open?id=1xZMeuSzCixqih9IqOPuuNODfqRzVkUXk&usp=drive_fs
    
    Due to the absence of ham and spam .txt files in `dataset` folder, the [`ProcessingTrainingData.py`] file (used for the pre-processing of email .txt files from scratch) won’t execute. If the [`ProcessingTrainingData.py`]is to be executed to carry out pre-processing, kindly download the spam and ham email .txt files from the `dataset` link mentioned above and place the emails in the `spam` and `ham` folders respectively before running the [`ProcessingTrainingData.py`]file.
