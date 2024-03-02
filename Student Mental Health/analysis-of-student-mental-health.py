import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown]
# **Importing of Libraries**

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:02.463388Z","iopub.execute_input":"2024-03-02T03:12:02.463908Z","iopub.status.idle":"2024-03-02T03:12:04.153161Z","shell.execute_reply.started":"2024-03-02T03:12:02.463877Z","shell.execute_reply":"2024-03-02T03:12:04.151335Z"}}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# **Loading of the Dataset**

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.160131Z","iopub.execute_input":"2024-03-02T03:12:04.161725Z","iopub.status.idle":"2024-03-02T03:12:04.186640Z","shell.execute_reply.started":"2024-03-02T03:12:04.161671Z","shell.execute_reply":"2024-03-02T03:12:04.185593Z"}}
data = pd.read_csv('/kaggle/input/student-mental-health/Student Mental health.csv')


# %% [markdown]
# **Checking the information of the dataset**

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.188095Z","iopub.execute_input":"2024-03-02T03:12:04.189099Z","iopub.status.idle":"2024-03-02T03:12:04.214219Z","shell.execute_reply.started":"2024-03-02T03:12:04.189060Z","shell.execute_reply":"2024-03-02T03:12:04.213319Z"}}
data.head(3)

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.216018Z","iopub.execute_input":"2024-03-02T03:12:04.216762Z","iopub.status.idle":"2024-03-02T03:12:04.236021Z","shell.execute_reply.started":"2024-03-02T03:12:04.216724Z","shell.execute_reply":"2024-03-02T03:12:04.234550Z"}}
data.tail(3)

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.237713Z","iopub.execute_input":"2024-03-02T03:12:04.238422Z","iopub.status.idle":"2024-03-02T03:12:04.275393Z","shell.execute_reply.started":"2024-03-02T03:12:04.238388Z","shell.execute_reply":"2024-03-02T03:12:04.273761Z"}}

data.info()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.277107Z","iopub.execute_input":"2024-03-02T03:12:04.277768Z","iopub.status.idle":"2024-03-02T03:12:04.296635Z","shell.execute_reply.started":"2024-03-02T03:12:04.277733Z","shell.execute_reply":"2024-03-02T03:12:04.295473Z"}}
data.isna().sum()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.297856Z","iopub.execute_input":"2024-03-02T03:12:04.298098Z","iopub.status.idle":"2024-03-02T03:12:04.316409Z","shell.execute_reply.started":"2024-03-02T03:12:04.298076Z","shell.execute_reply":"2024-03-02T03:12:04.314861Z"}}
data.shape

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.320481Z","iopub.execute_input":"2024-03-02T03:12:04.320822Z","iopub.status.idle":"2024-03-02T03:12:04.338332Z","shell.execute_reply.started":"2024-03-02T03:12:04.320794Z","shell.execute_reply":"2024-03-02T03:12:04.336900Z"}}
data.describe()

# %% [markdown]
# From the descriptive table above, the statistics shows that the Mean Age of the students in the dataset is around 20.53 years. The age values are dispersed by approximately 2.49628 years. Minimum and Maximum Ages is 18 and 24 respectively. In addition, 25% of the students are 18 years old or younger. 50% of the students are 19 years old while, 75% of the individuals are 23 years old. 

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.339694Z","iopub.execute_input":"2024-03-02T03:12:04.340169Z","iopub.status.idle":"2024-03-02T03:12:04.359602Z","shell.execute_reply.started":"2024-03-02T03:12:04.340139Z","shell.execute_reply":"2024-03-02T03:12:04.357308Z"}}
data.duplicated().sum()

# %% [markdown]
# **Copy of data is made. This is to sure that any changes or modifications done during processing or analysis will not affect the original dataset, hence dataset is duplicated. This helps to preserve the accuracy of your data.**

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.361498Z","iopub.execute_input":"2024-03-02T03:12:04.362677Z","iopub.status.idle":"2024-03-02T03:12:04.377429Z","shell.execute_reply.started":"2024-03-02T03:12:04.362530Z","shell.execute_reply":"2024-03-02T03:12:04.376115Z"}}
cdata = data.copy()

# %% [markdown]
# **Data Cleaning and Tranformation**

# %% [markdown]
# * Renaming of column labels for easy identification and manipulation.
# * Handling missing values
# * Cast type variables in there correct format
# * Correctly formatting  columns and remove whitespace
# * Renaming of variables for readability

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.378839Z","iopub.execute_input":"2024-03-02T03:12:04.379587Z","iopub.status.idle":"2024-03-02T03:12:04.393919Z","shell.execute_reply.started":"2024-03-02T03:12:04.379519Z","shell.execute_reply":"2024-03-02T03:12:04.392945Z"}}
cdata.columns

# %% [markdown]
# Renaming of column labels for easy identification and manipulation

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.395048Z","iopub.execute_input":"2024-03-02T03:12:04.397313Z","iopub.status.idle":"2024-03-02T03:12:04.410108Z","shell.execute_reply.started":"2024-03-02T03:12:04.397206Z","shell.execute_reply":"2024-03-02T03:12:04.408521Z"}}
cdata.columns = ['Timestamp', 'Gender', 'Age', 'Course',
       'Year of Study', 'CGPA', 'Marital status',
       'Depression', 'Anxiety',
       'Panic attack',
       'Treatment']

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.411900Z","iopub.execute_input":"2024-03-02T03:12:04.412814Z","iopub.status.idle":"2024-03-02T03:12:04.445222Z","shell.execute_reply.started":"2024-03-02T03:12:04.412778Z","shell.execute_reply":"2024-03-02T03:12:04.444272Z"}}
cdata

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.446521Z","iopub.execute_input":"2024-03-02T03:12:04.447427Z","iopub.status.idle":"2024-03-02T03:12:04.456433Z","shell.execute_reply.started":"2024-03-02T03:12:04.447394Z","shell.execute_reply":"2024-03-02T03:12:04.455052Z"}}
cdata['Year of Study'].unique()

# %% [markdown]
# There are some discrepancies with the year of study resulting in similarity in the unique value. Replace method will be applied to correct this discrepancies.

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.458513Z","iopub.execute_input":"2024-03-02T03:12:04.459605Z","iopub.status.idle":"2024-03-02T03:12:04.469484Z","shell.execute_reply.started":"2024-03-02T03:12:04.459552Z","shell.execute_reply":"2024-03-02T03:12:04.468329Z"}}
cdata['Year of Study'].replace({'year 1': 'year 1', 'year 2': 'year 2', 'Year 1': 'year 1', 'year 3': 'year 3', 'year 4': 'year 4', 'Year 2': 'year 2',
       'Year 3': 'year 3'}, inplace = True)

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.471348Z","iopub.execute_input":"2024-03-02T03:12:04.472319Z","iopub.status.idle":"2024-03-02T03:12:04.486909Z","shell.execute_reply.started":"2024-03-02T03:12:04.472229Z","shell.execute_reply":"2024-03-02T03:12:04.485855Z"}}
cdata['Year of Study'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.488416Z","iopub.execute_input":"2024-03-02T03:12:04.489575Z","iopub.status.idle":"2024-03-02T03:12:04.505269Z","shell.execute_reply.started":"2024-03-02T03:12:04.489535Z","shell.execute_reply":"2024-03-02T03:12:04.504230Z"}}
cdata['Course'].unique()

# %% [markdown]
# There are some discrepancies with the course. That is samen course is having different name. Replace method will be use to correct this discrepancies.

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.506749Z","iopub.execute_input":"2024-03-02T03:12:04.507864Z","iopub.status.idle":"2024-03-02T03:12:04.517586Z","shell.execute_reply.started":"2024-03-02T03:12:04.507827Z","shell.execute_reply":"2024-03-02T03:12:04.516427Z"}}
cdata['Course'].replace({'engin': 'Engineering' , 'Engine':'Engineering' , 'Islamic education':'Islamic Education' , 'Pendidikan islam':'Pendidikan Islam' , 'BIT':'IT', 'psychology':'Psychology', 'koe': 'KOE','Koe': 'KOE', 'Kirkhs': 'Irkhs', 'KIRKHS': 'Irkhs', 'BENL': 'Benl', 'Fiqh fatwa ': 'Fiqh', 'Laws': 'Law', 'Econs': 'Economics'} , inplace = True)


# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.519543Z","iopub.execute_input":"2024-03-02T03:12:04.520677Z","iopub.status.idle":"2024-03-02T03:12:04.538881Z","shell.execute_reply.started":"2024-03-02T03:12:04.520638Z","shell.execute_reply":"2024-03-02T03:12:04.537355Z"}}
cdata['Course'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.540407Z","iopub.execute_input":"2024-03-02T03:12:04.541472Z","iopub.status.idle":"2024-03-02T03:12:04.550035Z","shell.execute_reply.started":"2024-03-02T03:12:04.541432Z","shell.execute_reply":"2024-03-02T03:12:04.548903Z"}}
cdata['CGPA'].unique()

# %% [markdown]
# There are white spaces that need to remove so that the range of CGPA will be accurate. The strip() method will be applied.  

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.551395Z","iopub.execute_input":"2024-03-02T03:12:04.552234Z","iopub.status.idle":"2024-03-02T03:12:04.571774Z","shell.execute_reply.started":"2024-03-02T03:12:04.552203Z","shell.execute_reply":"2024-03-02T03:12:04.570336Z"}}
cdata['CGPA'] = cdata['CGPA'].str.strip()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.574548Z","iopub.execute_input":"2024-03-02T03:12:04.575286Z","iopub.status.idle":"2024-03-02T03:12:04.583280Z","shell.execute_reply.started":"2024-03-02T03:12:04.575218Z","shell.execute_reply":"2024-03-02T03:12:04.581953Z"}}
cdata['CGPA'].unique()

# %% [markdown]
# There is a missing value in the Age column. Refer to data.info(). 

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.585230Z","iopub.execute_input":"2024-03-02T03:12:04.585944Z","iopub.status.idle":"2024-03-02T03:12:04.596924Z","shell.execute_reply.started":"2024-03-02T03:12:04.585907Z","shell.execute_reply":"2024-03-02T03:12:04.595841Z"}}
median_age = cdata['Age'].median()
median_age 

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.598788Z","iopub.execute_input":"2024-03-02T03:12:04.599511Z","iopub.status.idle":"2024-03-02T03:12:04.606633Z","shell.execute_reply.started":"2024-03-02T03:12:04.599476Z","shell.execute_reply":"2024-03-02T03:12:04.605770Z"}}
cdata['Age'] = cdata['Age'].fillna(median_age)

# %% [markdown]
# Type casting the Age and TimeStamp columns to integer type and datatime using the astype()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.608336Z","iopub.execute_input":"2024-03-02T03:12:04.608990Z","iopub.status.idle":"2024-03-02T03:12:04.620407Z","shell.execute_reply.started":"2024-03-02T03:12:04.608959Z","shell.execute_reply":"2024-03-02T03:12:04.619326Z"}}
cdata['Age'] = cdata['Age'].astype(int)

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.629786Z","iopub.execute_input":"2024-03-02T03:12:04.630817Z","iopub.status.idle":"2024-03-02T03:12:04.646559Z","shell.execute_reply.started":"2024-03-02T03:12:04.630772Z","shell.execute_reply":"2024-03-02T03:12:04.645290Z"}}
cdata.info()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.647854Z","iopub.execute_input":"2024-03-02T03:12:04.648842Z","iopub.status.idle":"2024-03-02T03:12:04.656642Z","shell.execute_reply.started":"2024-03-02T03:12:04.648807Z","shell.execute_reply":"2024-03-02T03:12:04.655513Z"}}
cdata['Depression'].unique()

# %% [markdown]
# The target variable in the data is the depression-related column. For this reason, renaming it is essential in order to improve readability and enable visualization.Other columns with yes and no response will be rename also. 

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.658121Z","iopub.execute_input":"2024-03-02T03:12:04.659037Z","iopub.status.idle":"2024-03-02T03:12:04.666267Z","shell.execute_reply.started":"2024-03-02T03:12:04.659003Z","shell.execute_reply":"2024-03-02T03:12:04.665004Z"}}
cdata['Depression'] = cdata['Depression'].replace({'Yes': 'Depressed', 'No': 'Not depressed'})

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.667947Z","iopub.execute_input":"2024-03-02T03:12:04.668991Z","iopub.status.idle":"2024-03-02T03:12:04.680294Z","shell.execute_reply.started":"2024-03-02T03:12:04.668942Z","shell.execute_reply":"2024-03-02T03:12:04.679183Z"}}
cdata['Depression'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.681741Z","iopub.execute_input":"2024-03-02T03:12:04.682643Z","iopub.status.idle":"2024-03-02T03:12:04.691309Z","shell.execute_reply.started":"2024-03-02T03:12:04.682604Z","shell.execute_reply":"2024-03-02T03:12:04.689953Z"}}
cdata.columns

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.693165Z","iopub.execute_input":"2024-03-02T03:12:04.693748Z","iopub.status.idle":"2024-03-02T03:12:04.704299Z","shell.execute_reply.started":"2024-03-02T03:12:04.693715Z","shell.execute_reply":"2024-03-02T03:12:04.702750Z"}}
cdata['Panic attack'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.706524Z","iopub.execute_input":"2024-03-02T03:12:04.706903Z","iopub.status.idle":"2024-03-02T03:12:04.715782Z","shell.execute_reply.started":"2024-03-02T03:12:04.706859Z","shell.execute_reply":"2024-03-02T03:12:04.714096Z"}}
cdata['Panic attack'] = cdata['Panic attack'].replace({'Yes': 'Panic attack', 'No': 'No panic_attack'})

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.717893Z","iopub.execute_input":"2024-03-02T03:12:04.718307Z","iopub.status.idle":"2024-03-02T03:12:04.730798Z","shell.execute_reply.started":"2024-03-02T03:12:04.718265Z","shell.execute_reply":"2024-03-02T03:12:04.728036Z"}}
cdata['Panic attack'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.733762Z","iopub.execute_input":"2024-03-02T03:12:04.734529Z","iopub.status.idle":"2024-03-02T03:12:04.741123Z","shell.execute_reply.started":"2024-03-02T03:12:04.734490Z","shell.execute_reply":"2024-03-02T03:12:04.740175Z"}}
cdata['Anxiety'] = cdata['Anxiety'].replace({'Yes': 'Anxiety_present', 'No': 'No_anxiety'})

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.742735Z","iopub.execute_input":"2024-03-02T03:12:04.743393Z","iopub.status.idle":"2024-03-02T03:12:04.762611Z","shell.execute_reply.started":"2024-03-02T03:12:04.743360Z","shell.execute_reply":"2024-03-02T03:12:04.761742Z"}}
cdata['Anxiety'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.764585Z","iopub.execute_input":"2024-03-02T03:12:04.765296Z","iopub.status.idle":"2024-03-02T03:12:04.772259Z","shell.execute_reply.started":"2024-03-02T03:12:04.765236Z","shell.execute_reply":"2024-03-02T03:12:04.771343Z"}}
cdata['Treatment'] = cdata['Treatment'].replace({'Yes': 'Treatment', 'No': 'No_treatment'})

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.773656Z","iopub.execute_input":"2024-03-02T03:12:04.774173Z","iopub.status.idle":"2024-03-02T03:12:04.786692Z","shell.execute_reply.started":"2024-03-02T03:12:04.774139Z","shell.execute_reply":"2024-03-02T03:12:04.785707Z"}}
cdata['Treatment'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.788462Z","iopub.execute_input":"2024-03-02T03:12:04.789041Z","iopub.status.idle":"2024-03-02T03:12:04.796296Z","shell.execute_reply.started":"2024-03-02T03:12:04.789007Z","shell.execute_reply":"2024-03-02T03:12:04.795318Z"}}
cdata['Marital status'] = cdata['Marital status'].replace({'Yes': 'Married', 'No': 'Not_married'})

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.797886Z","iopub.execute_input":"2024-03-02T03:12:04.798524Z","iopub.status.idle":"2024-03-02T03:12:04.825734Z","shell.execute_reply.started":"2024-03-02T03:12:04.798488Z","shell.execute_reply":"2024-03-02T03:12:04.824910Z"}}
cdata

# %% [markdown]
# **DATA EXPLORATION**

# %% [markdown]
# In this section variables are explore in order to understand the distribution of variables, the range and variability of the data.

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.827036Z","iopub.execute_input":"2024-03-02T03:12:04.827539Z","iopub.status.idle":"2024-03-02T03:12:04.834673Z","shell.execute_reply.started":"2024-03-02T03:12:04.827508Z","shell.execute_reply":"2024-03-02T03:12:04.833182Z"}}
cdata.columns

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:04.836201Z","iopub.execute_input":"2024-03-02T03:12:04.836729Z","iopub.status.idle":"2024-03-02T03:12:05.683864Z","shell.execute_reply.started":"2024-03-02T03:12:04.836697Z","shell.execute_reply":"2024-03-02T03:12:05.682643Z"}}
numerical_variables = ['Age', 'CGPA']
colors = ['#546D64', '#689F7D']

for column, color in zip(numerical_variables, colors):
    plt.figure(figsize=(10, 5))
    sns.histplot(data=cdata, x = column, kde=True, color = color)
    plt.title(f'Distribution of {column}')
    plt.show()
    
    

# %% [markdown]
# Majority of the students fall in the age range 18-19 and 23-24.This distribution account for the fact that most of the students are in the first year. 
# Most of the students have excellent performance, reasons why the majority of students fall in the CGPA ranges of 3.0 - 3.49 and 3.5 - 4.00.

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:05.687112Z","iopub.execute_input":"2024-03-02T03:12:05.688034Z","iopub.status.idle":"2024-03-02T03:12:05.698131Z","shell.execute_reply.started":"2024-03-02T03:12:05.687993Z","shell.execute_reply":"2024-03-02T03:12:05.697229Z"}}
cdata['Depression'].value_counts()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:05.700162Z","iopub.execute_input":"2024-03-02T03:12:05.700572Z","iopub.status.idle":"2024-03-02T03:12:05.830865Z","shell.execute_reply.started":"2024-03-02T03:12:05.700538Z","shell.execute_reply":"2024-03-02T03:12:05.829292Z"}}
cdata['Depression'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#546D64','#689F7D'])
plt.title('Distribution of Depression Levels Among Students')
plt.show()

# %% [markdown]
# From the total number of students in the data 34.7% account for students that are depressed, although the rate is below 50%  this is rate is a  call for concern given that mental deals with the overall wellbeing of the students. 

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:05.832197Z","iopub.execute_input":"2024-03-02T03:12:05.832605Z","iopub.status.idle":"2024-03-02T03:12:05.928340Z","shell.execute_reply.started":"2024-03-02T03:12:05.832573Z","shell.execute_reply":"2024-03-02T03:12:05.926440Z"}}
cdata['Gender'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#546D64','#689F7D'])
plt.title('Distribution of Gender')
plt.show()

# %% [markdown]
# 74.3% of the student population is made up of female which shows that there are more female in the data compare to 25.7% of male.

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:05.930337Z","iopub.execute_input":"2024-03-02T03:12:05.930684Z","iopub.status.idle":"2024-03-02T03:12:06.253353Z","shell.execute_reply.started":"2024-03-02T03:12:05.930654Z","shell.execute_reply":"2024-03-02T03:12:06.251866Z"}}
categorical_variables = [ 'Year of Study', 'Anxiety']
colors = ['#546D64', '#689F7D']

for column, color in zip(categorical_variables, colors):
    plt.figure(figsize=(10, 5))
    sns.countplot(data=cdata, y=column, color=color)
    plt.title(f'Distribution of {column}')
plt.tight_layout()
plt.show()


# %% [markdown]
# Majority of the students are in the first year. this could be one of the reasons why the dominant age group is 18 and 19. For the count of students with anxiety a considerable count(30 and above) for students expiriencing anxiety and 60% and above are not experiencing anxiety. 

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:06.256980Z","iopub.execute_input":"2024-03-02T03:12:06.257336Z","iopub.status.idle":"2024-03-02T03:12:06.264131Z","shell.execute_reply.started":"2024-03-02T03:12:06.257310Z","shell.execute_reply":"2024-03-02T03:12:06.263018Z"}}
len(cdata['Course'].unique())

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:06.265601Z","iopub.execute_input":"2024-03-02T03:12:06.266754Z","iopub.status.idle":"2024-03-02T03:12:06.560283Z","shell.execute_reply.started":"2024-03-02T03:12:06.266721Z","shell.execute_reply":"2024-03-02T03:12:06.559331Z"}}
plt.figure(figsize=(10, 8))
cdata.Course.value_counts().iloc[:18].plot(kind='barh',color= '#689F7D' )
plt.title('Top 18 Course by Order Count', fontsize=15)
plt.xlabel('Order Count')
plt.ylabel('Course')
plt.show()

# %% [markdown]
# The majority of the students are enrolled engineering and BCS relative to other courses has less number of students.

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:06.561670Z","iopub.execute_input":"2024-03-02T03:12:06.562483Z","iopub.status.idle":"2024-03-02T03:12:06.568923Z","shell.execute_reply.started":"2024-03-02T03:12:06.562453Z","shell.execute_reply":"2024-03-02T03:12:06.567974Z"}}
cdata['Depression'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:06.570200Z","iopub.execute_input":"2024-03-02T03:12:06.570692Z","iopub.status.idle":"2024-03-02T03:12:06.604100Z","shell.execute_reply.started":"2024-03-02T03:12:06.570658Z","shell.execute_reply":"2024-03-02T03:12:06.602180Z"}}
depression_df = cdata[cdata.Depression == 'Depressed']
depression_df

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:06.605860Z","iopub.execute_input":"2024-03-02T03:12:06.606887Z","iopub.status.idle":"2024-03-02T03:12:06.618425Z","shell.execute_reply.started":"2024-03-02T03:12:06.606849Z","shell.execute_reply":"2024-03-02T03:12:06.617225Z"}}
depression_df.info()

# %% [markdown]
# Numbers of depressed student across gender

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:06.619648Z","iopub.execute_input":"2024-03-02T03:12:06.619927Z","iopub.status.idle":"2024-03-02T03:12:06.637907Z","shell.execute_reply.started":"2024-03-02T03:12:06.619899Z","shell.execute_reply":"2024-03-02T03:12:06.635178Z"}}
depression_df.Gender.value_counts(normalize=True)

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:06.639424Z","iopub.execute_input":"2024-03-02T03:12:06.640096Z","iopub.status.idle":"2024-03-02T03:12:06.809966Z","shell.execute_reply.started":"2024-03-02T03:12:06.640032Z","shell.execute_reply":"2024-03-02T03:12:06.808418Z"}}
colors = ['#546D64', '#689F7D']

plt.figure(figsize=(6, 4))
sns.countplot(data=depression_df, x='Gender', palette=colors)
plt.title('Distribution of Depressed Student Across Gender', fontsize=10)
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# %% [markdown]
# The distribution of Depressed Student Across Gender shows that more female student are depressed relative to the male students. 

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:06.811536Z","iopub.execute_input":"2024-03-02T03:12:06.811890Z","iopub.status.idle":"2024-03-02T03:12:06.963126Z","shell.execute_reply.started":"2024-03-02T03:12:06.811857Z","shell.execute_reply":"2024-03-02T03:12:06.962006Z"}}
colors = ['#546D64', '#689F7D']

plt.figure(figsize=(6, 4))
sns.countplot(data=depression_df, x='Marital status', palette=colors)
plt.title('Distribution of Depressed Student Across Marital status', fontsize=10)
plt.xlabel('Marital status')
plt.ylabel('Count')
plt.show()

# %% [markdown]
# For the distribution of students across married and unmarried. we see that the number of unmarried students that are depressed are more. this might be because the married students have support from their spouse which makes it easier for them to navigate life thus low number of married people being deoressed or due to the fact that a married student most have built a stronger attitude due to the expiriences gathered in matrimony which makes them more mentally stable than the unmarried.

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:06.964924Z","iopub.execute_input":"2024-03-02T03:12:06.965257Z","iopub.status.idle":"2024-03-02T03:12:07.133841Z","shell.execute_reply.started":"2024-03-02T03:12:06.965209Z","shell.execute_reply":"2024-03-02T03:12:07.132338Z"}}
colors = ['#546D64', '#689F7D', '#8ABF99', '#AED6B1', '#CDE8D0', '#E9F8EB']

plt.figure(figsize=(6, 4))
sns.countplot(data=depression_df, x='Age', palette=colors)
plt.title('Distribution of Depressed Students Across Age', fontsize=10)
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# %% [markdown]
# Across student age the proportion of students who are depressed falls with age 18 and 19. At 20 to 22 the proportion of depressed students decrease. There is also a noticeable increase in the number of depressed studnt form Age 24. 

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:07.135327Z","iopub.execute_input":"2024-03-02T03:12:07.135611Z","iopub.status.idle":"2024-03-02T03:12:07.302962Z","shell.execute_reply.started":"2024-03-02T03:12:07.135585Z","shell.execute_reply":"2024-03-02T03:12:07.301743Z"}}
colors = ['#546D64', '#689F7D', '#8ABF99']
plt.figure(figsize=(6, 4))
sns.countplot(data=depression_df, x='CGPA', palette=colors)
plt.title('Distribution of Depressed Student Across CGPA', fontsize=10)
plt.xlabel('CGPA')
plt.ylabel('Count')
plt.show()

# %% [markdown]
# There are more depressed students with CGPA OF 3.00 - 3.49, followed by students with CGPA of 3.50 - 4.00. Only a few proportion of student with CGPA OF 2.50 - 2.99 are depressed.

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:07.304197Z","iopub.execute_input":"2024-03-02T03:12:07.304475Z","iopub.status.idle":"2024-03-02T03:12:07.310396Z","shell.execute_reply.started":"2024-03-02T03:12:07.304451Z","shell.execute_reply":"2024-03-02T03:12:07.309018Z"}}
depression_df.columns

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:07.311556Z","iopub.execute_input":"2024-03-02T03:12:07.312453Z","iopub.status.idle":"2024-03-02T03:12:07.721159Z","shell.execute_reply.started":"2024-03-02T03:12:07.312417Z","shell.execute_reply":"2024-03-02T03:12:07.720209Z"}}
colors = {'Panic attack': '#546D64', 'No panic_attack': '#689F7D'}

plt.figure(figsize=(8, 5))
sns.displot(data=depression_df, x='Age', hue='Panic attack', palette=colors, kind='kde', fill=True)
plt.title('Distribution of Age with Panic Attack', fontsize=12)
plt.xlabel('Age')
plt.ylabel('Density')
plt.show()

# %% [markdown]
# The kernel density plot indicate that there are variations in the distribution of ages between students  with panick and no panick attack. Also, students  within the age of 18 and above without panick attack are more likely to be depressed relative to student with panick attack.

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:07.722312Z","iopub.execute_input":"2024-03-02T03:12:07.722830Z","iopub.status.idle":"2024-03-02T03:12:08.051841Z","shell.execute_reply.started":"2024-03-02T03:12:07.722795Z","shell.execute_reply":"2024-03-02T03:12:08.050488Z"}}
plt.figure(figsize=(10, 8))
sns.swarmplot(data=depression_df, x='Anxiety', y='Course', hue='Gender', palette=['#546D64', '#50FFB1'])
plt.title('Swarm Plot of Anxiety by Course and Gender')
plt.xlabel('Anxiety')
plt.ylabel('Course')
plt.show()

# %% [markdown]
#  More female students enrolled in engineering with no anxiety have depression compare to male student enrolled in engineering. while female student enrolled in IT with anxiety are more likely to have depression relative to male enrolled in IT. Overall there are variation in depression status across course of study and anxiety level.

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:08.053197Z","iopub.execute_input":"2024-03-02T03:12:08.053555Z","iopub.status.idle":"2024-03-02T03:12:08.294574Z","shell.execute_reply.started":"2024-03-02T03:12:08.053526Z","shell.execute_reply":"2024-03-02T03:12:08.293208Z"}}
plt.figure(figsize=(10, 8))
sns.swarmplot(data=depression_df, x='Anxiety', y='CGPA', hue='Gender', palette=['#546D64', '#50FFB1'])
plt.title('Swarm Plot of Anxiety by Course and Gender')
plt.xlabel('Anxiety')
plt.ylabel('CGPA')
plt.show()

# %% [markdown]
# More female students  with CGPA of 3.00 to 3.49 with no anxiety have depression compare to female student with depression. while female student with anxiety and CGPA of 3.50 to 4.00 are more likely to have depression relative female students without depression and same CGPA . Overall there are variations in depression Among female student according to their CGPA. Female students with high CGPA tends to be anxious and this can lead to depression. further analysis is needed to prove the relationship between this variables. 

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T03:12:08.295867Z","iopub.execute_input":"2024-03-02T03:12:08.296149Z","iopub.status.idle":"2024-03-02T03:12:08.558649Z","shell.execute_reply.started":"2024-03-02T03:12:08.296123Z","shell.execute_reply":"2024-03-02T03:12:08.557297Z"}}
plt.figure(figsize=(10, 8))
sns.swarmplot(data=depression_df, x='Anxiety', y='Year of Study', hue='Gender', palette=['#546D64', '#50FFB1'])
plt.title('Swarm Plot of Anxiety by year of study and Gender')
plt.xlabel('Anxiety')
plt.ylabel('Year of Study')
plt.show()

# %% [markdown]
# More female students in year with no anxiety have depression compare to female students with anxiety in same level. For year 2, more female student with no anxiety have depression. In year 3, more female students with anxiety have depression. Overall there are variations in depression accross gender, Anxiety and year of students.
