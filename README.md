# EXNO-6-DS-DATA VISUALIZATION USING SEABORN LIBRARY

# Aim:
  To Perform Data Visualization using seaborn python library for the given datas.

# EXPLANATION:
Data visualization is the graphical representation of information and data. By using visual elements like charts, graphs, and maps, data visualization tools provide an accessible way to see and understand trends, outliers, and patterns in data.

# Algorithm:
STEP 1:Include the necessary Library.

STEP 2:Read the given Data.

STEP 3:Apply data visualization techniques to identify the patterns of the data.

STEP 4:Apply the various data visualization tools wherever necessary.

STEP 5:Include Necessary parameters in each functions.

# Coding and Output:

import seaborn as sns
import matplotlib.pyplot as plt
x=[1,2,3,4,5]
y=[3,6,2,7,1]
sns.lineplot(x=x,y=y)

output <img width="956" height="556" alt="image" src="https://github.com/user-attachments/assets/a5fdb038-77c5-4538-bf18-36c06d707ebb" />

df=sns.load_dataset("tips")
df

output <img width="686" height="452" alt="image" src="https://github.com/user-attachments/assets/2dbbd4ee-d6e9-4db6-91a2-ab48ea4018f9" />

sns.lineplot(x="total_bill",y="tip",data=df,hue="sex",linestyle='solid',legend="auto")

output <img width="942" height="589" alt="image" src="https://github.com/user-attachments/assets/719321cf-9fd7-45be-962a-100160391458" />

x = [1, 2, 3, 4, 5]
y1 = [3, 5, 2, 6, 1]
y2 = [1, 6, 4, 3, 8]
y3 = [5, 2, 7, 1, 4]
sns.lineplot(x=x, y=y1)
sns.lineplot(x=x, y=y2)
sns.lineplot(x=x, y=y3)
# Set plot title and axis labels
plt.title('Multi-Line Plot')
plt.xlabel('X Label')
plt.ylabel("Y Label" )

output <img width="1003" height="604" alt="image" src="https://github.com/user-attachments/assets/9ba6f20b-5878-45b2-96f2-a39e725f36cb" />

import seaborn as sns
import matplotlib.pyplot as plt
tips = sns.load_dataset('tips')
avg_total_bill = tips.groupby('day')['total_bill'].mean()
avg_tip = tips.groupby('day')['tip'].mean()
plt.figure(figsize=(8, 6))
p1 = plt.bar(avg_total_bill.index, avg_total_bill, label='Total Bill')
p2 = plt.bar(avg_tip.index, avg_tip, bottom=avg_total_bill, label='Tip')
plt.xlabel('Day of the Week')
plt.ylabel('Amount')
plt.title('Average Total Bill and Tip by Day')
plt.legend()

output <img width="1171" height="723" alt="image" src="https://github.com/user-attachments/assets/f525e43c-d268-4803-88b6-5ea0bfc85203" />

avg_total_bill = tips.groupby('time')['total_bill'].mean()
avg_tip = tips.groupby('time' ) ['tip' ].mean()
p1 = plt.bar(avg_total_bill.index, avg_total_bill, label='Total Bill', width=0.4)
p2 = plt.bar(avg_tip.index, avg_tip, bottom = avg_total_bill, label='Tip', width=0.4)
plt.xlabel('Time of Day')
plt.ylabel('Amount')
plt.title('Average Total Bill and Tip by Time of Day')
plt.legend()

output <img width="1014" height="625" alt="image" src="https://github.com/user-attachments/assets/b9558e03-62f4-49a7-aa31-14a4431fc5a8" />

years = range(2000, 2012)
apples = [0.895, 0.91, 0.919, 0.926, 0.929, 0.931, 0.934, 0.936, 0.937, 0.9375, 0.9372, 0.939]
oranges = [0.962, 0.941, 0.930, 0.923, 0.918, 0.908, 0.907, 0.904, 0.901, 0.898, 0.9, 0.896, ]
plt.bar(years, apples)
plt.bar(years, oranges, bottom=apples)

output <img width="972" height="562" alt="image" src="https://github.com/user-attachments/assets/b3b69d15-551f-483e-87fc-35b3a0041155" />

import seaborn as sns
dt= sns. load_dataset ( 'tips' )
sns.barplot (x='day', y='total_bill', hue='sex', data=dt, palette='Set1' )
plt.xlabel( 'Day of the Week' )
plt.ylabel( 'Total Bill')
plt.title('Total Bill by Day and Gender')

output <img width="921" height="619" alt="image" src="https://github.com/user-attachments/assets/1c0cbd3a-4ef6-48a8-9519-677746107793" />

import pandas as pd
tit=pd.read_csv("titanic_dataset.csv")
tit

output <img width="1330" height="445" alt="image" src="https://github.com/user-attachments/assets/92b7052f-9ba5-465e-af45-f05fd0b2f323" />

plt.figure(figsize=(8,5))
sns.barplot (x='Embarked',y='Fare' ,data=tit, palette='rainbow' )
plt.title("Fare of Passenger by Embarked Town")

output <img width="1146" height="643" alt="image" src="https://github.com/user-attachments/assets/b554503b-4715-4a26-bc40-6dafdc6af354" />

plt.figure(figsize=(8,5))
sns.barplot (x='Embarked',y='Fare' , data=tit, palette='rainbow', hue='Pclass' )
plt.title("Fare of Passenger by Embarked Town, Divided by Class")

output <img width="1246" height="648" alt="image" src="https://github.com/user-attachments/assets/66bcc6db-a642-4979-b861-724034c26b49" />

import seaborn as sns
tips = sns.load_dataset('tips')
sns.scatterplot(x='total_bill', y='tip', hue='sex' , data=tips)
plt.xlabel('Total Bill')
plt.ylabel('Tip Amount')
plt.title('Scatter Plot of Total Bill vs. Tip Amount')

output <img width="934" height="625" alt="image" src="https://github.com/user-attachments/assets/316b8f6e-d127-4f94-90c7-845221012206" />

import seaborn as sns
import numpy as np
import pandas as pd
np.random.seed(1)
num_var = np.random.randn(1000)
num_var = pd.Series(num_var, name = "Numerical Variable")
num_var

output <img width="811" height="268" alt="image" src="https://github.com/user-attachments/assets/55c34220-0496-4abd-95e3-185c957156fd" />

sns.histplot(data=num_var,kde=True)

output <img width="971" height="578" alt="image" src="https://github.com/user-attachments/assets/4eeb45d8-f155-46be-9b90-e3c43a366767" />

df=pd.read_csv("titanic_dataset.csv")
df

output <img width="1333" height="456" alt="image" src="https://github.com/user-attachments/assets/0ae258c4-eae7-40a7-b124-f349d9de7121" />

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed (0)
marks = np.random.normal(loc=70, scale=10, size=100)
marks

output <img width="933" height="428" alt="image" src="https://github.com/user-attachments/assets/1e5f3928-2160-4b19-a979-61fe39a2a88f" />

sns.histplot(data=marks, bins=10, kde=True, stat='count', cumulative=False, multiple='stack', element='bars', palette='Set1', shrink=0.7)
plt.xlabel('Marks')
plt.ylabel('Density')
plt.title('Histogram of Students Marks')

output <img width="1351" height="724" alt="image" src="https://github.com/user-attachments/assets/86a4b2a8-1911-4ee4-b3a6-c7920db44e04" />

import seaborn as sns
import pandas as pd
tips = sns. load_dataset('tips' )
sns.boxplot (x=tips['day'], y=tips['total_bill'], hue=tips['sex' ])

output <img width="1144" height="586" alt="image" src="https://github.com/user-attachments/assets/031fc319-f996-4939-939a-c9df1dc20d43" />

sns.boxplot(x="day", y="total_bill", hue="smoker", data=tips, linewidth=2, width=0.6,boxprops={"facecolor": "lightblue", "edgecolor": "darkblue"},
whiskerprops={"color": "black", "linestyle": "--", "linewidth": 1.5 }, capprops={"color": "black", "linestyle": "--", "linewidth": 1.5})

output <img width="1030" height="593" alt="image" src="https://github.com/user-attachments/assets/149acd98-6c37-4863-bf3d-bb283d173e3e" />

sns.boxplot (x='Pclass',y='Age' , data=df, palette='rainbow' )
plt.title("Age by Passenger Class, Titanic")

output <img width="985" height="611" alt="image" src="https://github.com/user-attachments/assets/ac385740-d6cb-499c-b91c-89f4a7244a5b" />

sns.violinplot (x="day", y="total_bill", hue="smoker", data=tips, linewidth=2, width=0.6,palette="Set3", inner="quartile")
plt.xlabel("Day of the Week")
plt.ylabel("Total Bill")
plt.title("Violin Plot of Total Bill by Day and Smoker Status")

output <img width="1117" height="621" alt="image" src="https://github.com/user-attachments/assets/7609029d-ce91-4caa-93e0-9811fece999a" />

sns.kdeplot(data=tips,x='total_bill',hue='time',multiple='fill',linewidth=3,palette='Set2',alpha=0.8)

output <img width="984" height="595" alt="image" src="https://github.com/user-attachments/assets/10cb05ee-9f4d-4c91-84da-b79a5a95d2bf" />





# Result:
 Include your result here
