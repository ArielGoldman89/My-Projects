#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('darkgrid')
pd.options.display.float_format = '{:.2f}'.format

from ydata_profiling import ProfileReport
import missingno as msno

# Geopandas
import geopandas
from geodatasets import get_path
import geodatasets

# Folium
import folium

# Plotly
import plotly.express as px
# plotly subplots
import plotly.subplots as sp
from plotly.subplots import make_subplots

# Cufflinks 
import cufflinks as cf

# Altair
import altair as alt

# Bokeh
import bokeh

# Panel (Dashboard)
import panel as pn


# In[ ]:


# pip install ydata-profiling


# In[3]:


pn.extension('plotly')


# In[4]:


df = pd.read_csv('hotel_booking.csv')


# In[5]:


ProfileReport(df)


# In[6]:


df.head()


# In[7]:


df.tail()


# In[8]:


df.duplicated().sum()


# In[10]:


ax = msno.bar(df)


# In[11]:


df.info()


# In[12]:


print('Total number of observations before dropping:', df.shape[0])
before_drop = df.shape[1]
print('Total number of attributes before dropping:', before_drop)


# In[13]:


print('Columns Before dropping:\n\n',df.columns)


# In[14]:


# Removing the unnecassary features from the dataset which are not relavent

df.drop(['name', 'email','phone-number', 'credit_card'],axis=1,inplace=True)


# In[15]:


print('Columns after dropping:\n\n', df.columns)


# In[16]:


print('Total number of attributes before dropping:', before_drop)
print('Total number of attributes after dropping:', df.shape[1])


# # Exploratory Data Analysis

# In[17]:


# Target Varaible 
colors= ['green','blue']
df['is_canceled'].value_counts().plot(kind='bar',color=colors);

df['is_canceled'].value_counts()


# In[18]:


# From which Country the Guests come from?

country_arrival = df[df['is_canceled'] == 0]['country'].value_counts().reset_index()
country_arrival.columns = ['Country', 'Monthly Arrivals']

# Top 10 arrival countries
country_arrival.head(10)


# ## Pie Chart
# 
# https://plotly.com/python/pie-charts/

# In[19]:


top10_countries = country_arrival.head(10)

Top10CountriesPie = px.pie(top10_countries, names='Country', 
                           values='Monthly Arrivals',
                           title='Top 10 Arrival Countries',
                           color_discrete_sequence=px.colors.sequential.RdBu)

Top10CountriesPie.update_traces(textposition='inside', textinfo='percent+label')

Top10CountriesPie.show()


# ## Displaying Arrival Countries on the World Map
# 
# https://plotly.com/python/choropleth-maps/

# In[20]:


ArrivalCountries = px.choropleth(country_arrival, 
                                 locations='Country',
                                 color='Monthly Arrivals',
                                 range_color=[20, 3000],
#                                  hover_name='Country',
                                 custom_data=['Country', 'Monthly Arrivals'])                   

ArrivalCountries.update_layout(title_text='Arrival Countries',
                               geo=dict(projection={'type': 'natural earth'}))

ArrivalCountries.show();


# ### Subsets Target Variable

# In[21]:


df_resort = df[(df['hotel'] == 'Resort Hotel') & (df['is_canceled'] == 0)]
df_city = df[(df['hotel'] == 'City Hotel') & (df['is_canceled'] == 0)] 


# ## ADR fluctuations per nigh per hotel 

# In[22]:


df['adr'].sort_values(ascending=False).head()


# In[23]:


# Dropping outlier with 'adr=5400' as is not representative.
df = df.drop(index=48515)


# In[24]:


df['adr'].sort_values().head()


# In[25]:


# Dropping negative 'adr=-6.38' as is not representative.
df = df.drop(index=14969)


# ### ADR Resort Hotel

# In[26]:


adr_resort = df_resort.groupby('arrival_date_month')['adr'].mean().reset_index()
adr_resort


# ### ADR City Hotel

# In[27]:


adr_city = df_city.groupby('arrival_date_month')['adr'].mean().reset_index()
adr_city


# ### Merged both ADR Hotels

# In[28]:


adr_both_hotels = adr_resort.merge(adr_city, on='arrival_date_month')
adr_both_hotels.columns = ['Month', 'ADR Resort', 'ADR City']

adr_both_hotels


# #### Sorting Months Chronologically 

# In[29]:


### We need to sort the months accordingly for readability

month = ['January', 'February', 'March', 'April', 'May', 'June',
         'July', 'August', 'September', 'October', 'November', 'December']

adr_both_hotels['Month'] = pd.Categorical(adr_both_hotels['Month'], categories=month, ordered=True)

# Sort Months
adr_both_hotels.sort_values('Month', inplace=True)

adr_both_hotels


# # Line charts
# 
# # Trends that change over time across the ADR.
# * https://plotly.com/python/line-charts/

# ## Line Chart ADR

# In[30]:


LineChartADR = px.line(adr_both_hotels, x='Month', y=['ADR City', 'ADR Resort'])

LineChartADR.update_traces(mode='lines+markers', marker=dict(size=12),
                           line=dict(width=3, color='purple'), selector=dict(name='ADR City'))

LineChartADR.update_traces(mode='lines+markers', marker=dict(size=12),
                           line=dict(width=3, color='green'), selector=dict(name='ADR Resort'))

LineChartADR.update_layout(title='Monthly ADR Comparison',
                           xaxis_title='Months',
                           yaxis_title='Average Daily Rate')

LineChartADR.show()


# ### Arrivals Resort Hotel

# In[31]:


arrivals_resort = df_resort['arrival_date_month'].value_counts().reset_index()
arrivals_resort.columns=['Month','Arrivals Resort']
arrivals_resort


# ### Arrivals City Hotel

# In[32]:


arrivals_city = df_city['arrival_date_month'].value_counts().reset_index()
arrivals_city.columns=['Month','Arrivals City']
arrivals_city


# ### Merged both Guests Hotels

# In[33]:


both_arrivals = arrivals_resort.merge(arrivals_city, on='Month')
both_arrivals.columns = ['Month', 'Arrivals Resort', 'Arrivals City']
both_arrivals


# #### Sorting Months Chronologically

# In[34]:


# We need to sort the months accordingly for readability

month = ['January', 'February', 'March', 'April', 'May', 'June',
         'July', 'August', 'September', 'October', 'November', 'December']

both_arrivals['Month'] = pd.Categorical(both_arrivals['Month'], categories=month, ordered=True)

# Sort Months
both_arrivals.sort_values('Month', inplace=True)

both_arrivals


# ## Line Chart Arrivals

# In[35]:


LineChartArrivals = px.line(both_arrivals, x='Month', y=['Arrivals City', 'Arrivals Resort'])

LineChartArrivals.update_traces(mode='lines+markers', marker=dict(size=12),
                              line=dict(width=3, color='purple'), selector=dict(name='Arrivals City'))

LineChartArrivals.update_traces(mode='lines+markers', marker=dict(size=12),
                              line=dict(width=3, color='green'), selector=dict(name='Arrivals Resort'))

LineChartArrivals.update_layout(title='Monthly Arrivals Comparison',
                              xaxis_title='Months',
                              yaxis_title='Arrivals')

LineChartArrivals.show()


# # Histograms 
# 
# # Distribution for Categorical Variables 
# * https://plotly.com/python/histograms/

# ### Market Segment 

# In[36]:


MarketSegment = px.histogram(df, 
                         x='market_segment',
                         color='is_canceled',
                         barmode='group',
                         title='Market Segment',
                         labels={'market_segment': 'Market Segment'},
                         animation_frame='arrival_date_month',
                         category_orders={'arrival_date_month': ['January', 'February', 'March', 'April', 'May', 'June',
                                                                 'July', 'August', 'September', 'October', 'November',
                                                                 'December']},
                         color_discrete_map={0: 'darkolivegreen', 1: 'darkorange'}  
                         )

MarketSegment.show()


# ### CustomerType

# In[37]:


CustomerType = px.histogram(df, 
                         x='customer_type',
                         color='is_canceled',
                         barmode='group',
                         title='Customer Type',
                         labels={'customer_type': 'Customer Type'},
                         animation_frame='arrival_date_month',
                         category_orders={'arrival_date_month': ['January', 'February', 'March', 'April', 'May', 'June',
                                                                 'July', 'August', 'September', 'October', 'November',
                                                                 'December']},
                         color_discrete_map={0: 'darkolivegreen', 1: 'darkorange'}  
                         )

CustomerType.show()


# # Box Plot 
# 
# # Distribution of the target variable for Categorical Variables
# 
# https://plotly.com/python/box-plots/

# In[38]:


# total_night column was added for analysis
df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
df.head()


# In[39]:


(df['total_nights'] == 0).sum()


# In[40]:


boxplot = sp.make_subplots(rows=1, cols=5,
                           subplot_titles=['Lead Time', 'Week Nº', 'Month Day',
                                           'ADR', 'Total Nights'])
                                       
box1 = px.box(df, x='is_canceled', y='lead_time', color='is_canceled')
box2 = px.box(df, x='is_canceled', y='arrival_date_week_number', color='is_canceled')
box3 = px.box(df, x='is_canceled', y='arrival_date_day_of_month', color='is_canceled')
box4 = px.box(df, x='is_canceled', y='adr', color='is_canceled')
box5 = px.box(df, x='is_canceled', y='total_nights', color='is_canceled')

# No Canceled
boxplot.add_trace(box1['data'][0], row=1, col=1)
boxplot.add_trace(box2['data'][0], row=1, col=2)
boxplot.add_trace(box3['data'][0], row=1, col=3)
boxplot.add_trace(box4['data'][0], row=1, col=4)
boxplot.add_trace(box5['data'][0], row=1, col=5)

# Canceled
boxplot.add_trace(box1['data'][1], row=1, col=1)
boxplot.add_trace(box2['data'][1], row=1, col=2)
boxplot.add_trace(box3['data'][1], row=1, col=3)
boxplot.add_trace(box4['data'][1], row=1, col=4)
boxplot.add_trace(box5['data'][1], row=1, col=5)

boxplot.update_layout(title_text='Features Vs. Target Variable')
boxplot.show()


# ## ADR Room Type

# In[41]:


BoxPlotRoom = px.box(data_frame = df, x = 'reserved_room_type', y = 'adr', color = 'hotel',
                     color_discrete_map={'City Hotel': 'darkblue', 'Resort Hotel': 'green'})

BoxPlotRoom.update_layout(title='ADR Room Type',
                          xaxis_title='Reserved Room Type',
                          yaxis_title='ADR',
                          showlegend=True)

BoxPlotRoom.show()


# # Dashboard

# In[42]:


dashboard = pn.Column(
    pn.pane.Markdown('# Hotel Booking Dashboard Analysis'),
    pn.Row(pn.Column(ArrivalCountries), pn.Column(Top10CountriesPie)),
    pn.Row(pn.Column(LineChartArrivals), pn.Column(LineChartADR)),
    pn.Row(pn.Column(boxplot), pn.Column(BoxPlotRoom)),
    pn.Row(pn.Column(MarketSegment), pn.Column(CustomerType))
)


# In[43]:


dashboard.servable()


# In[ ]:




