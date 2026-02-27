#!/usr/bin/env python
# coding: utf-8

# In[2]:


import nltk
# Download the necessary resources
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4') # Often required for WordNet on some systems


# In[3]:


import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# 1. Download missing resources
nltk.download('punkt_tab')
nltk.download('wordnet')

# 2. Setup
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def process_feedback(text):
    tokens = word_tokenize(text.lower())
    stems = [stemmer.stem(w) for w in tokens]
    lemmas = [lemmatizer.lemmatize(w) for w in tokens]
    return stems, lemmas

# 3. Example Usage
sample = "The professors are teaching very helpful lessons."
stems, lemmas = process_feedback(sample)

print(f"Original:   {sample}")
print(f"Stemmed:    {stems}")
print(f"Lemmatized: {lemmas}")


# In[4]:


import random
import pandas as pd
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

# Download requirements
nltk.download('punkt')
nltk.download('wordnet')

# 1. Setup Data Generation
subjects = ["Python", "Machine Learning", "Data Science", "Web Dev", "Statistics"]
adjectives = ["amazing", "confusing", "helpful", "outdated", "practical", "difficult"]
nouns = ["lectures", "assignments", "quizzes", "grading", "instructor", "content"]

def generate_feedback(n=50):
    feedback_list = []
    for _ in range(n):
        s = f"The {random.choice(subjects)} {random.choice(nouns)} were {random.choice(adjectives)}."
        feedback_list.append(s)
    return list(set(feedback_list)) # Remove duplicates

# 2. Initialize Tools
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# 3. Process Data
raw_data = generate_feedback(60) # Generate extra to ensure 50 unique
final_data = []

for comment in raw_data[:50]:
    tokens = word_tokenize(comment.lower())
    stemmed = " ".join([stemmer.stem(w) for w in tokens])
    lemmatized = " ".join([lemmatizer.lemmatize(w) for w in tokens])
    
    final_data.append({
        "Original": comment,
        "Stemmed": stemmed,
        "Lemmatized": lemmatized
    })

# 4. Save to CSV for your report
df = pd.DataFrame(final_data)
df.to_csv("student_feedback_results.csv", index=False)
print("Processing complete. File saved as 'student_feedback_results.csv'.")
print(df.head())


# In[5]:


get_ipython().system('pip install pandas')


# In[1]:


import random
import pandas as pd
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

# Download requirements
nltk.download('punkt')
nltk.download('wordnet')

# 1. Setup Data Generation
subjects = ["Python", "Machine Learning", "Data Science", "Web Dev", "Statistics"]
adjectives = ["amazing", "confusing", "helpful", "outdated", "practical", "difficult"]
nouns = ["lectures", "assignments", "quizzes", "grading", "instructor", "content"]

def generate_feedback(n=50):
    feedback_list = []
    for _ in range(n):
        s = f"The {random.choice(subjects)} {random.choice(nouns)} were {random.choice(adjectives)}."
        feedback_list.append(s)
    return list(set(feedback_list)) # Remove duplicates

# 2. Initialize Tools
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# 3. Process Data
raw_data = generate_feedback(60) # Generate extra to ensure 50 unique
final_data = []

for comment in raw_data[:50]:
    tokens = word_tokenize(comment.lower())
    stemmed = " ".join([stemmer.stem(w) for w in tokens])
    lemmatized = " ".join([lemmatizer.lemmatize(w) for w in tokens])
    
    final_data.append({
        "Original": comment,
        "Stemmed": stemmed,
        "Lemmatized": lemmatized
    })

# 4. Save to CSV for your report
df = pd.DataFrame(final_data)
df.to_csv("student_feedback_results.csv", index=False)
print("Processing complete. File saved as 'student_feedback_results.csv'.")
print(df.head())


# In[2]:


import sys
get_ipython().system('{sys.executable} -m pip install pandas')


# In[5]:


import random
import pandas as pd
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# 1. Setup & Downloads
nltk.download('punkt')
nltk.download('wordnet')

# Force pandas to show ALL rows and full column width
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

# 2. Data Generation Logic
subjects = ["Python", "Machine Learning", "Data Science", "Web Dev", "Statistics", "AI", "NLP", "SQL"]
adjectives = ["amazing", "confusing", "helpful", "outdated", "practical", "difficult", "engaging", "vague"]
nouns = ["lectures", "assignments", "quizzes", "grading", "instructor", "content", "projects", "videos"]

def generate_unique_feedback(target_count=50):
    feedback_set = set()
    while len(feedback_set) < target_count:
        comment = f"The {random.choice(subjects)} {random.choice(nouns)} were {random.choice(adjectives)}."
        feedback_set.add(comment)
    return list(feedback_set)

# 3. Processing
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

raw_comments = generate_unique_feedback(50)
results = []

for comment in raw_comments:
    tokens = word_tokenize(comment.lower())
    # Apply Stemming
    stems = " ".join([stemmer.stem(w) for w in tokens])
    # Apply Lemmatization
    lemmas = " ".join([lemmatizer.lemmatize(w) for w in tokens])
    
    results.append({
        "Original Comment": comment,
        "Stemmed Version": stems,
        "Lemmatized Version": lemmas
    })

# 4. Display Everything
df = pd.DataFrame(results)
print(df)

# Save to CSV for your records
df.to_csv("student_feedback_full_50.csv", index=False)


# In[ ]:




