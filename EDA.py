#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis (EDA)
# # Analyze and investigate data sets and summarize their main characteristics
# 1. understanding the format and contents of the DataFrame at a glance
# 2. understanding the structure of the DataFrame and identifying any missing data
# 3. drops the rows from the DataFrame df that contain missing values (NaN) and assigns the cleaned DataFrame to a new variable df_cleaned
# 4. provides an overview of the cleaned data, displays the summary information of the cleaned DataFrame df_cleaned, including the column names, data types, and the count of non-null values for each column
# 5. counts the number of tracks associated with each unique artist in the cleaned DataFrame df_cleaned and stores the counts in the artist_counts variable
# 6. counts the number of tracks associated with each unique album in the cleaned DataFrame df_cleaned and stores the counts in the album_counts variable
# 7. generates a histogram to visualize the distribution of Spotify stream numbers in the cleaned DataFrame df_cleaned
# 8. creates a bar plot to display the top 10 artists with the highest track counts from the artist_counts variable
# 9. creates a scatter plot to visualize the correlation between Spotify stream numbers and YouTube views in the cleaned DataFrame df_cleaned
# 10. generates a histogram to visualize the distribution of the danceability feature in the cleaned DataFrame df_cleaned
# 11. analyzing the distribution of numerical features
# 12. investigating the relationship between numerical features
# 13. analyzing the distribution of categorical variables

# In[1]:


import pandas as pd
import pyarrow.parquet as pq

# Load the Parquet file into a DataFrame
df = pq.read_table(r"C:/Users/GOGA/Desktop/spotify.parquet").to_pandas()


# In[2]:


#1 Display the first few rows of the DataFrame
print(df.head())


# In[3]:


#2 Display the summary information of the DataFrame
print(df.info())


# In[4]:


#3 Drop rows with missing values
df_cleaned = df.dropna()

#4 Verify the updated DataFrame
print(df_cleaned.info())


# In[5]:


import matplotlib.pyplot as plt

#5 Count the number of tracks for each artist
artist_counts = df_cleaned['Artist'].value_counts()
print(artist_counts)

#6 Count the number of tracks for each album
album_counts = df_cleaned['Album'].value_counts()
print(album_counts)

#7 Explore the distribution of Spotify stream numbers
plt.hist(df_cleaned['Spotify_stream_num'], bins=20)
plt.xlabel('Spotify Stream Numbers')
plt.ylabel('Count')
plt.title('Distribution of Spotify Stream Numbers')
plt.show()


# In[6]:


#8 Plot the top 10 artists by track count
top_artists = artist_counts.head(10)
plt.figure(figsize=(10, 6))
top_artists.plot(kind='bar')
plt.xlabel('Artist')
plt.ylabel('Count')
plt.title('Top 10 Artists by Track Count')
plt.xticks(rotation=45)
plt.show()


# In[7]:


#9 Plot the correlation between Spotify stream numbers and YouTube views
plt.scatter(df_cleaned['Spotify_stream_num'], df_cleaned['Youtube_Views'])
plt.xlabel('Spotify Stream Numbers')
plt.ylabel('YouTube Views')
plt.title('Correlation between Spotify Stream Numbers and YouTube Views')
plt.show()


# In[8]:


import matplotlib.pyplot as plt

#10 Visualize the distribution of danceability
plt.hist(df_cleaned['Danceability'], bins=20)
plt.xlabel('Danceability')
plt.ylabel('Count')
plt.title('Distribution of Danceability')
plt.show()


# In[14]:


import matplotlib.pyplot as plt

#11 Histogram of 'Energy'
plt.hist(df_cleaned['Energy'], bins=20)
plt.xlabel('Energy')
plt.ylabel('Count')
plt.title('Distribution of Energy')
plt.show()

# Box plot of 'Loudness'
plt.boxplot(df_cleaned['Loudness'])
plt.xlabel('Loudness')
plt.title('Distribution of Loudness')
plt.show()


# In[15]:


import seaborn as sns

#12 Scatter plot of 'Tempo' vs. 'Energy'
sns.scatterplot(x='Tempo', y='Energy', data=df_cleaned)
plt.xlabel('Tempo')
plt.ylabel('Energy')
plt.title('Relationship between Tempo and Energy')
plt.show()

# Correlation matrix
correlation_matrix = df_cleaned[['Energy', 'Loudness', 'Speechiness', 'Acousticness', 'Valence']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[16]:


#13 Bar plot of 'Album_type'
album_type_counts = df_cleaned['Album_type'].value_counts()
plt.bar(album_type_counts.index, album_type_counts.values)
plt.xlabel('Album Type')
plt.ylabel('Count')
plt.title('Distribution of Album Types')
plt.xticks(rotation=45)
plt.show()

# Pie chart of 'Is_official_video'
is_official_video_counts = df_cleaned['Is_official_video'].value_counts()
plt.pie(is_official_video_counts.values, labels=is_official_video_counts.index, autopct='%1.1f%%')
plt.title('Distribution of Official Videos')
plt.show()

