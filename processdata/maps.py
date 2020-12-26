import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)

import seaborn as sns
import datetime
import os
import itertools

#plots-core
import plotly.express as px
from plotly.offline import plot
import plotly.graph_objs as go

import plotly.figure_factory as ff
from plotly.colors import n_colors
from plotly.subplots import make_subplots

from PIL import Image

import re
from collections import Counter
from . import getdata
import requests
import json

data = None
Data_per_country = None
Data_world = None
tweet_date = None
top_mention = None
covid = None
devices = None
show_mode = False

# all the pre-processing and data preparation
# original path '../input/novel-corona-virus-2019-dataset/covid_19_data.csv'
# local path 'C:\Users\Haim\COVID-19\covid_19_data.csv'
data = pd.read_csv('processdata/data/covid_19_data.csv')
data["Province/State"] = data["Province/State"].fillna('Unknown')
data[["Confirmed", "Deaths", "Recovered"]] = data[["Confirmed", "Deaths", "Recovered"]].astype(int)
data['Country/Region'] = data['Country/Region'].replace('Mainland China', 'China')
data['Active_case'] = data['Confirmed'] - data['Deaths'] - data['Recovered']
Data = data[data['ObservationDate'] == max(data['ObservationDate'])].reset_index()
# for ring() #1e
Data_world = Data.groupby(["ObservationDate"])[
    ["Confirmed", "Active_case", "Recovered", "Deaths"]].sum().reset_index()
# for play() # 2
data_over_time = Data.groupby(["ObservationDate"])[
    ["Confirmed", "Active_case", "Recovered", "Deaths"]]\
    .sum().reset_index().sort_values("ObservationDate", ascending=True).reset_index(drop=True)
data_per_country = data.groupby(["Country/Region", "ObservationDate"])[["Confirmed","Active_case","Recovered","Deaths"]].sum().reset_index().sort_values("ObservationDate",ascending=True).reset_index(drop=True)
# for activity() # 1e
covid=pd.read_csv('processdata/data/covid19_tweets.csv')

country_code=pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')
covid['country_name']=covid['user_location'].str.split(',').str[-1]
covid['only_date']=pd.to_datetime(covid['date']).dt.date

#Keeping countries with valid country name
# without_country_name=pd.read_csv('../input/country-tweet/without_country_name.csv',low_memory=False)  # NA

# Useful for code matching with countries - Plotly Chlorepeth MAP
country_code=pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')

with_country_name=covid[covid['country_name'].isin(list(country_code['COUNTRY']))]
with_country_name['filtered_name']=covid['country_name']
# tweet_df=with_country_name.append(without_country_name)
# tweet_state_count=tweet_df['filtered_name'].value_counts().to_frame().reset_index().rename(columns={'index':'country','filtered_name':'count'})
tweet_state_count=with_country_name['filtered_name'].value_counts().to_frame().reset_index().rename(columns={'index':'country','filtered_name':'count'})
all_tweet_location=pd.merge(tweet_state_count,country_code[['COUNTRY','CODE']],left_on="country",right_on="COUNTRY",how="left")
all_tweet_location=all_tweet_location[all_tweet_location['COUNTRY'].notnull()]
covid['tweet_date']=pd.to_datetime(covid['date']).dt.date
tweet_date=covid['tweet_date'].value_counts().to_frame().reset_index().rename(columns={'index':'date','tweet_date':'count'})
tweet_date['date']=pd.to_datetime(tweet_date['date'])
tweet_date=tweet_date.sort_values('date',ascending=False)

devices=covid['source'].value_counts().to_frame().reset_index().rename(columns={'index':'source','source':'count'})[:15]


# for wordcloud
def remove_tag(string):
    text=re.sub('<.*?>','',string)
    return text


def remove_mention(text):
    line=re.sub(r'@\w+','',text)
    return line


def remove_hash(text):
    line=re.sub(r'#\w+','',text)
    return line


def remove_newline(string):
    text=re.sub('\n','',string)
    return text


def remove_url(string):
    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','',string)
    return text


def remove_number(text):
    line=re.sub(r'[0-9]+','',text)
    return line


def remove_punct(text):
    line = re.sub(r'[!"\$%&\'()*+,\-.\/:;=#@?\[\\\]^_`{|}~]*','',text)
    #string="".join(line)
    return line


def text_strip(string):
    line=re.sub('\s{2,}', ' ', string.strip())
    return line


def remove_thi_amp_ha_words(string):
    line=re.sub(r'\bamp\b|\bthi\b|\bha\b',' ',string)
    return line


covid['refine_text']=covid['text'].str.lower()
covid['refine_text']=covid['refine_text'].apply(lambda x:remove_tag(str(x)))
covid['refine_text']=covid['refine_text'].apply(lambda x:remove_mention(str(x)))
covid['refine_text']=covid['refine_text'].apply(lambda x:remove_hash(str(x)))
covid['refine_text']=covid['refine_text'].apply(lambda x:remove_newline(x))
covid['refine_text']=covid['refine_text'].apply(lambda x:remove_url(x))
covid['refine_text']=covid['refine_text'].apply(lambda x:remove_number(x))
covid['refine_text']=covid['refine_text'].apply(lambda x:remove_punct(x))
covid['refine_text']=covid['refine_text'].apply(lambda x:remove_thi_amp_ha_words(x))
covid['refine_text']=covid['refine_text'].apply(lambda x:text_strip(x))

covid['text_length']=covid['refine_text'].str.split().map(lambda x: len(x))




# hashtags
def find_hash(text):
    line=re.findall(r'(?<=#)\w+',text)
    return " ".join(line)
covid['hash']=covid['text'].apply(lambda x:find_hash(x))
hastags=list(covid[(covid['hash'].notnull())&(covid['hash']!="")]['hash'])
hastags = [each_string.lower() for each_string in hastags]
hash_df=dict(Counter(hastags))
top_hash=pd.DataFrame(list(hash_df.items()),columns = ['word','count']).sort_values('count',ascending=False)[:20]


# DONE
# 2 - something cool about corona
# plays a heat map of confirmed cases for all countries over time
def play():
    # missing : data_per_country,
    fig = px.choropleth(data_per_country, 
                        locations=data_per_country['Country/Region'],
                        locationmode='country names',
                        color=data_per_country['Confirmed'],
                        color_continuous_scale="Viridis",
                        hover_name=data_per_country['Country/Region'],
                        animation_frame="ObservationDate"
                        
                        )
    
    fig.update_layout(
        title_text='Confirmed Cases Over Time',
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='equirectangular'
        ), 
        paper_bgcolor='rgba(0,0,0,0)',
                      margin={"r":0,"t":0,"l":0,"b":0},
                template="plotly_dark",
    )
    plot_div = plot(fig, include_plotlyjs=False, output_type='div', config={'displayModeBar': False})
    return plot_div


# DONE
# 1e - ring graph NOTE: not about tweets
def ring():
    colors = ['#132f65', '#8a8678', '#fae839']
    # Missing: Data_world
    labels = ["Active cases", "Recovered", "Deaths"]
    values = Data_world.loc[0, ["Active_case", "Recovered", "Deaths"]]
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0, 0, 0.2], hole=0.5)])
    
    fig.update_traces(hoverinfo='label+percent', textinfo='value',
                  marker=dict(colors=colors, line=dict(color='#000000', width=0.5)))
    
    fig.update_layout( paper_bgcolor='rgba(0,0,0,0)', )
    plot_div = plot(fig, include_plotlyjs=False, output_type='div', config={'displayModeBar': False})
    if show_mode:
        fig.show()
    return plot_div


# 1e alternative - graph of number of tweets per day
def activity():
    # Missing : tweet_date
    fig = go.Figure(go.Scatter(x=tweet_date['date'],
                               y=tweet_date['count'],
                               mode='markers+lines',
                               name="Submissions",
                               marker_color='dodgerblue'))

    fig.update_layout(template="plotly_dark",)

    plot_div = plot(fig, include_plotlyjs=False, output_type='div', config={'displayModeBar': False})
    if show_mode:
        fig.show()
    return plot_div


    # 2e - hashtags
def hashtag():
    # Missing: top_mention
    fig = go.Figure(
        go.Bar(
        x=top_hash['word'], y=top_hash['count'],
        marker={'color': top_hash['count'],
                'colorscale': 'blues'},
        text=top_hash['count'],
        textposition="outside",
    ))
    
    fig.update_layout(template="plotly_dark", )
    
    plot_div = plot(fig, include_plotlyjs=False, output_type='div', config={'displayModeBar': False})
    if show_mode:
        fig.show()
    return plot_div



# 4e - device usage
def sources():
    # Missing: devices
    fig = go.Figure(go.Bar(
        x=devices['source'], y=devices['count'],
        marker={'color': devices['count'],
                'colorscale': 'blues'},
        text=devices['count'],
        textposition="outside",
    ))

    fig.update_layout(template="plotly_dark")
    plot_div = plot(fig, include_plotlyjs=False, output_type='div', config={'displayModeBar': False})
    if show_mode:
        fig.show()
    return plot_div


# 5e - violin
# violin graph of average tweet length over time
def violin():
    fig = go.Figure(data=go.Violin(y=covid['text_length'], box_visible=True, line_color='black',
                                   meanline_visible=True, fillcolor='lightblue ', opacity=0.6,
                                   x0='Tweet Text Length '))

    fig.update_layout(yaxis_zeroline=False, template="plotly_dark",)
    
    plot_div = plot(fig, include_plotlyjs=False, output_type='div', config={'displayModeBar': False})
    return plot_div


# vars for Israel
all_dates = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/'
                        'csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
all_dates = [date for date in all_dates.columns if date[0].isdigit()]
pathURLDeath = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/' \
               'csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
pathURLConfirmed = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/' \
                   'csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
pathURLRecoverd = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/' \
                  'csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'


# functions for israel graph
def date_range(start_date, days):
    start = all_dates.index(start_date)
    indexes = list(range(start, start+days))
    return [all_dates[i] for i in indexes if i < len(all_dates)]


def get_isreal_data():
    # data pre-processing and preparation
    drop_list = ['Lat', 'Long', 'Country/Region', 'Province/State']
    # get death cases info
    countries_csv_death = pd.read_csv(pathURLDeath)
    mask = [True if value == 'Israel' else False for value in countries_csv_death['Country/Region']]
    countries_data_death = countries_csv_death[mask]
    countries_data_death = countries_data_death.drop(columns=drop_list)
    countries_data_death['criteria'] = 'deaths'
    # get confirmed cases info
    countries_csv_confirmed = pd.read_csv(pathURLConfirmed)
    mask = [True if value == 'Israel' else False for value in countries_csv_confirmed['Country/Region']]
    countries_csv_confirmed = countries_csv_confirmed[mask]
    countries_csv_confirmed = countries_csv_confirmed.drop(columns=drop_list)
    countries_csv_confirmed['criteria'] = 'confirmed'
    # get recovered cases info
    countries_csv_recovered = pd.read_csv(pathURLRecoverd)
    mask = [True if value == 'Israel' else False for value in countries_csv_recovered['Country/Region']]
    countries_csv_recovered = countries_csv_recovered[mask]
    countries_csv_recovered = countries_csv_recovered.drop(columns=drop_list)
    countries_csv_recovered['criteria'] = 'recovered'
    # add the data to a single dataframe
    israel_data = countries_data_death.append(countries_csv_confirmed, ignore_index=True)\
        .append(countries_csv_recovered, ignore_index=True)
    return israel_data


# 1 - something about israel
def israel():
    israel_data = get_isreal_data()
    israel_data = israel_data.drop(columns='criteria')
    deaths = israel_data.values[0]
    confirmed = israel_data.values[1]
    recovered = israel_data.values[2]
    # israel_data.columns = pd.to_datetime(israel_data.index)
    # israel_data.columns = israel_data.index.strftime('%Y-%m-%d')
    dates = israel_data.columns
    
    deaths_scatter = go.Scatter(name='Deaths', x=dates, y=deaths, line=dict(color="#fae839", width=4))
    confirmed_scatter = go.Scatter(name='Confirmed', x=dates, y=confirmed, line=dict(color="#132f65", width=4))
    recovered_scatter = go.Scatter(name='Recovered', x=dates, y=recovered, line=dict(color="#8a8678", width=4))
    
    fig = go.Figure(data=[deaths_scatter, confirmed_scatter, recovered_scatter])
    
    fig.update_layout(showlegend= False, margin = dict(t=0, l=0, r=0, b=200) ,         
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                hovermode="closest",
                yaxis= dict(automargin= True, gridcolor= "#32325d"),
                xaxis= dict(automargin= True, showgrid= False),
                font= dict(color = '#ced4da'),
                )
    
    plot_div = plot(fig, include_plotlyjs=False, output_type='div', config={'displayModeBar': False})

    # fig.show()
    return plot_div