############################## Importing dependencies ##########################

## The cwd path should be the path to the folder in which the .py file containing code is present
## All other supporting functions are also present in the same directory

import nltk
import re
import string

#**********************************************************************************


# function to remove escape characters and punctuations
def remove_escape_characters_and_punctuations(text):
    # Replace escape characters with a space
    cleaned_text = text.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
    
    # Replace punctuation with spaces
    cleaned_text = cleaned_text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    
    # Remove any extra spaces that may have been added
    cleaned_text = ' '.join(cleaned_text.split())
    
    return cleaned_text


# html parser
def html_parser(html):
    # Step 1: Remove content within <head> tags
    text = re.sub('<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)
    
    # Step 2: Replace <a> tags with "HYPERLINK"
    text = re.sub('<a\s.*?>', ' HYPERLINK ', text, flags=re.M | re.S | re.I)
    
    # Step 3: Remove all remaining HTML tags
    text = re.sub('<.*?>', '', text, flags=re.M | re.S)
    
    # Step 4: Remove extra newlines
    text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)
    
    # Step 5: Replace common HTML entities manually
    html_entities = {
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&#39;': "'",
        '&nbsp;': ' '
    }
    for entity, char in html_entities.items():
        text = text.replace(entity, char)
    
    return remove_escape_characters_and_punctuations(text)

## Url detector and swapper
def url_swapper(text):
    if not isinstance(text, str):
        return ""
    
    # Define a regex pattern to identify URLs
    url_pattern = re.compile(
        r'((http|https|ftp):\/\/)?'          # Protocol (optional)
        r'(\w+(\-\w+)*\.)+'                 # Subdomain(s)
        r'([a-z]{2,6})'                     # Domain
        r'(:\d+)?'                          # Port (optional)
        r'(\/[\w\-\.\~\?=&%\+;]*)*',        # Path (optional)
        flags=re.IGNORECASE
    )
    
    # Substitute all matched URLs with "URL"
    text = url_pattern.sub(" URL ", text)
    
    return text

# stemmer function
def stemmer(text):
    ps = nltk.PorterStemmer()
    text_list = text.split(" ")
    out_list = [ps.stem(w) for w in text_list]
    return " ".join(out_list)

## combining all the pre-processing functions
def email_to_text(email):
    email = html_parser(email)   ## parsing html content
    email = url_swapper(email)   ## replacing urls with URL
    email = stemmer(email)       ## carrying out stemming
    return email