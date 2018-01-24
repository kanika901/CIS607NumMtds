
# Import pylab to provide scientific Python libraries (NumPy, SciPy, Matplotlib)
# import pylab as pl
# import the Image display module
from IPython.display import Image
import pandas as pd
import scipy as sp
from scipy import stats
import numpy as np
import matplotlib.pylab as py
from scipy.stats import expon
import matplotlib.pyplot as plt
import re as re
import collections as c
import pprint as pp


Image(url='http://assets.zipfianacademy.com/data/data-science-workflow/animate.gif', width=700)

'''
To perform some interesting statistical analyses, we first need to "join" our CSV files in order to associate businesses 
with their inspection scores. This data currently resides in SFBusinesses/businesses.csv and SFBusinesses/inspections.csv
'''

# import pandas library which provides an R like environment for python.
# if you do not have it installed: sudo easy_install pandas.



# store relevant file paths in variables since we may use them frequently
root_dir = 'data/SFBusinesses/'
businesses = root_dir + 'businesses.csv'
inspections = root_dir + 'inspections.csv'


# load each file into a Pandas DataFrame, pandas automatically converts the first line into a header for the columns

df_business = pd.read_csv(businesses, encoding = "ISO-8859-1")
df_inspection = pd.read_csv(inspections, encoding = "ISO-8859-1")

# inspect the first 10 rows of the DataFrame
df_inspection.head(10)

'''
we can 'join' DataFrames just as we would database tables
pandas uses a left-outer join by default, meaning that all 
the records from the businesses will be present even if there
is not a corresponding row in the inspections.
'''

# join the two DataFrames on the 'business_id' column
big_table = df_business.merge(df_inspection, on='business_id')

# the joined DataFrame columns: frame1 columns + frame2 columns
# in our case it is the concatenation of df_business and df_inspection columns
print('Business:\t' + str(df_business.columns) + '\n')
print('Inspection:\t' + str(df_inspection.columns) + '\n')
print('Big Table:\t' + str(big_table.columns))

# allows for row and column indexing succinctly
big_table.iloc[:10, :4]

# let us first group our data by business so we can find its most recent score for the inspections

grouped_business = big_table.groupby('business_id')

# a funtion that takes a DataFrame and returns the row with the newest date
def most_recent(df, column='date'):
    return df.sort_index(by=column)[-1:]
    
# input to most_recent is the DataFrame of each group, in this case 
# all of the rows and columns for each business (grouped on business_id). 
most_recent_inspection_results = grouped_business.apply(most_recent)
 
# We applied the most_recent function to extract the row
# of the DataFrame with the most recent inspection.
most_recent_inspection_results.head()

# Filter out records without lat/long for mapping
r = most_recent_inspection_results

zero_filtered = r[(r['latitude'] != 0) & (r['latitude'] != 0)]

filtered = zero_filtered.dropna(subset=['latitude', 'longitude'])[['business_id','name', 'address', 'Score', 'date', 'latitude', 'longitude']]

filtered.to_csv('geolocated_rest.csv', index=False)


# create a matplotlib figure with size [15,7]
plt.figure(figsize=[15,7])

# pandas built-in histogram function automatically distributes and counts bin values 
h = most_recent_inspection_results['Score'].hist(bins=100)

# create x-axis ticks of even numbers 0-100
plt.xticks(np.arange(40, 100, 2))

# add a title to the current figure, our histogram
h.set_title("Histogram of Inspection Scores")

# Now that we have a basic idea of the distribution, let us look at some more interesting statistics
scores =  most_recent_inspection_results['Score']
mean = scores.mean()
median = scores.median()

# compute descriptive summary statistics of the inspection scores
summary = scores.describe()

mode = sp.stats.mode(scores)
skew = scores.skew()

# compute quantiles
ninety = scores.quantile(0.9)
eighty = scores.quantile(0.8)
seventy = scores.quantile(0.7)
sixty = scores.quantile(0.6)

print("Skew: " + str(skew))
print("90%: " + str(ninety))
print("80%: " + str(eighty))
print("70%: " + str(seventy))
print("60%: " + str(sixty))
print(summary)

'''
let us compute a histogram of these descriptions as well
'''

# first we need to discretize the numerical values, this can be 
# thought of as converting a continuous variable into a categorical one.
descriptions = ['Poor', 'Needs Improvement', 'Adequate', 'Good']
bins = [-1, 70, 85, 90, 100]

# copy the scores from our grouped DataFrame, DataFrames manipulate
# in place if we do not explicitly copy them.
scores = most_recent_inspection_results['Score'].copy()
score_transform = most_recent_inspection_results.copy()

# built-in pandas funcion which assigns each data point in 
# 'scores' to an interval in 'bins' with labels of 'descriptions'
discretized_scores = pd.cut(scores, bins ,labels=descriptions)

# tranform the original DataFrame's "Score" column with the new descriptions
score_transform['Score'] = discretized_scores

score_transform[['name', 'date','Score']].head(15)

Image(url='http://assets.zipfianacademy.com/data/data-science-workflow/evaluate.png', width=500)

'''
plot a histogram of the discretized scores
'''

# create a figure with 2 subplots
fig = plt.figure(figsize=[30,7])
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

# count each occurance of descriptions in the 'Score' column,
# and reverse this result so 'Poor' is left most and 'Good' right most
counts = score_transform['Score'].value_counts()[::-1]
plot_score = counts.plot(kind='bar', ax=ax2)

# decorate the plot and axis with text
ax2.set_title("Restaurant Inspections (%i total)" % sum(counts))
ax2.set_ylabel("Counts")
ax2.set_xlabel("Description")

# let us add some labels to each bar
for x, y in enumerate(counts):
    plt.text(x + 0.5, y + 200, '%.f' % y, ha='left', va= 'top')
    
# plot the original raw scores (same grapoh as earlier)
most_recent_inspection_results['Score'].hist(bins=100, ax= ax1)
# create x-axis ticks of even numbers 0-100
ax1.set_xticks(np.arange(40, 100, 2))

# add a title to the current figure, our histogram
ax1.set_title("Histogram of Inspection Scores")
ax1.set_ylabel("Counts")
ax1.set_xlabel("Score")

py.savefig('histograms.png', bbox_inches=0)

# create a matplotlib figure with size [15,7]
plt.figure(figsize=[15,7])
# pandas built-in histogram function automatically distributes and counts bin values 
h = most_recent_inspection_results['Score'].hist(bins=100)

# summary statistics vertical lines
ax1.axvline(x=mean,color='red',ls='solid', lw="3", label="mean")
ax1.axvline(x=median,color='green',ls='solid', lw="3", label="median")
ax1.axvline(x=mode[0][0],color='orange',ls='solid', lw="3", label="mode")

# 25th quantile
ax1.axvline(x=summary['25%'],color='maroon',ls='dashed', lw="3", label="25th")
ax1.axvspan(40, summary['25%'], facecolor="maroon", alpha=0.3)

# 75th quantile
ax1.axvline(x=summary['75%'],color='black',ls='dashed', lw="3", label="75th")
ax1.axvspan(40, summary['75%'], facecolor="yellow", alpha=0.3 )

# create x-axis ticks of even numbers 0-100
plt.xticks(np.arange(40, 104, 2))

# add legend to graph
plt.legend(loc=2)

# add a title to the current figure, our histogram
h.set_title("Histogram of Inspection Scores")

plt.savefig('quantiles.png', bbox_inches=0, transparent=True)

print(summary)


# first let us form a 'big table' by joining the violations to the most recent inspection scores
file="data/SFFoodProgram_Complete_Data/violations_plus.csv"

df_violations = pd.read_csv(file)

violation_table = most_recent_inspection_results.merge(df_violations, on=['business_id','date'])
violation_table.head()

# Let us see how the violations are distributed
plt.figure(figsize=[18,7])

violation_hist = violation_table['description'].value_counts().plot(kind="bar")

# Let us see what violations a restaurant can have and still get a perfect score

plt.figure(figsize=[18,7])

perfect_scores = violation_table[violation_table['Score'] == 100]

violation_hist = perfect_scores['description'].value_counts().plot(kind="bar")

perfect_scores

# Hmmm, apparently high risk vermin infestations are minor violations
# If that is minor, what is a severe violation

df_violations['ViolationSeverity'].drop_duplicates()

# well aparently there are only minor violations

# Let us bin health violations using the cities quantizations

descriptions = ['Poor', 'Needs Improvement', 'Adequate', 'Good']
bins = [-1, 70, 85, 90, 100]

# copy the scores from our grouped DataFrame, DataFrames manipulate
# in place if we do not explicitly copy them.
scores = violation_table['Score'].copy()
violation_transform = violation_table.copy()

# built-in pandas funcion which assigns each data point in 
# 'scores' to an interval in 'bins' with labels of 'descriptions'
discretized_scores = pd.cut(scores, bins ,labels=descriptions)
violation_transform["Scores"] = discretized_scores

grouped = violation_transform.groupby('Scores')

# let us find the most common violations for each group

# a funtion that takes a DataFrame and returns the top violations
def common_offenses(df):
    return pd.DataFrame(df['description'].value_counts(normalize=True) * 100).head(10)

top_offenses = grouped.apply(common_offenses)

f = plt.figure(figsize=[18,7])
colors = ['r', 'b', 'y', 'g']

for name, group in grouped:
    group['description'].value_counts().plot(kind="bar", axes=f, alpha=0.5, color=colors.pop())

Image(url='http://assets.zipfianacademy.com/data/data-science-workflow/communicate.png', width=500)

