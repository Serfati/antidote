import numpy as np
import pandas as pd
import plotly.express as px
from plotly.offline import plot
import plotly.graph_objs as go
import re
from collections import Counter

show_mode = False

covid = pd.read_csv('processdata/data/covid19_tweets.csv')
covid['country_name'] = covid['user_location'].str.split(',').str[-1]
covid['only_date'] = pd.to_datetime(covid['date']).dt.date
country_code = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')
with_country_name = covid[covid['country_name'].isin(list(country_code['COUNTRY']))]
with_country_name['filtered_name'] = covid['country_name']
covid['tweet_date'] = pd.to_datetime(covid['date']).dt.date

df = pd.read_csv('processdata/data/covid_19_data.csv')
df["Province/State"] = df["Province/State"].fillna('Unknown')
df[["Confirmed", "Deaths", "Recovered"]] = df[["Confirmed", "Deaths", "Recovered"]].astype(int)
df['Country/Region'] = df['Country/Region'].replace('Mainland China', 'China')


def remove_tag(string):
    text = re.sub('<.*?>', '', string)
    return text


def remove_mention(text):
    line = re.sub(r'@\w+', '', text)
    return line


def remove_hash(text):
    line = re.sub(r'#\w+', '', text)
    return line


def remove_newline(string):
    text = re.sub('\n', '', string)
    return text


def remove_url(string):
    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', string)
    return text


def remove_number(text):
    line = re.sub(r'[0-9]+', '', text)
    return line


def remove_punct(text):
    line = re.sub(r'[!"\$%&\'()*+,\-.\/:;=#@?\[\\\]^_`{|}~]*', '', text)
    # string="".join(line)
    return line


def text_strip(string):
    line = re.sub('\s{2,}', ' ', string.strip())
    return line


def remove_thi_amp_ha_words(string):
    line = re.sub(r'\bamp\b|\bthi\b|\bha\b', ' ', string)
    return line


def find_hash(text):
    line = re.findall(r'(?<=#)\w+', text)
    return " ".join(line)


# DONE
# 2 - something cool about corona
# plays a heat map of confirmed cases for all countries over time
def play():
    global df
    data_per_country = df.groupby(["Country/Region", "ObservationDate"])[["Confirmed"]].sum().reset_index()\
        .sort_values("ObservationDate", ascending=True).reset_index(drop=True)
    fig = px.choropleth(data_per_country,
                        locations=data_per_country['Country/Region'],
                        locationmode='country names',
                        color=data_per_country['Confirmed'],
                        color_continuous_scale="Cividis",
                        hover_name=data_per_country['Country/Region'],
                        animation_frame="ObservationDate"
                        )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
    )
    plot_div = plot(fig, include_plotlyjs=False, output_type='div', config={'displayModeBar': False})
    return plot_div


def marker():
    global df
    mark = df[df['ObservationDate'] == max(df['ObservationDate'])].reset_index()
    markers = mark.groupby(["Country/Region"])["Confirmed"].sum().reset_index().sort_values("Confirmed", ascending=False).reset_index(drop=True)
    fig = go.Figure(data=[go.Scatter(
        x=markers['Country/Region'][0:10],
        y=markers['Confirmed'][0:10],
        mode='markers',
        marker=dict(
            color=[145, 140, 135, 130, 125, 120, 115, 110, 105, 100],
            size=(markers['Confirmed'][0:10] / 25000),
            showscale=True
        )
    )])

    fig.update_layout(
        yaxis_title="Confirmed Cases",
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
    )

    plot_div = plot(fig, include_plotlyjs=False, output_type='div', config={'displayModeBar': False})
    return plot_div


def death_ratio():
    global df
    mark = df[df['ObservationDate'] == max(df['ObservationDate'])].reset_index()
    markers = mark.groupby(["Country/Region"])["Deaths"].sum().reset_index().sort_values("Deaths", ascending=False).reset_index(drop=True)
    fig = go.Figure(data=[go.Scatter(
        x=markers['Country/Region'][0:10],
        y=markers['Deaths'][0:10],
        mode='markers',
        marker=dict(
            color=[145, 140, 135, 130, 125, 120, 115, 110, 105, 100],
            size=markers['Deaths'][0:10] / 1000,
            showscale=True
        )
    )])

    fig.update_layout(
        yaxis_title="Deaths",
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
    )

    plot_div = plot(fig, include_plotlyjs=False, output_type='div', config={'displayModeBar': False})
    return plot_div


def ring():
    global df
    mark = df[df['ObservationDate'] == max(df['ObservationDate'])].reset_index()
    data_world = mark.groupby(["ObservationDate"])[
        ["Confirmed", "Active_case", "Recovered", "Deaths"]].sum().reset_index()
    colors = ['#132f65', '#8a8678', '#fae839']
    labels = ["Active cases", "Recovered", "Deaths"]
    values = data_world.loc[0, ["Active_case", "Recovered", "Deaths"]]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0, 0, 0.2], hole=0.5)])

    fig.update_traces(hoverinfo='label+percent', textinfo='value',
                      marker=dict(colors=colors, line=dict(color='#000000', width=0.5)))

    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', )
    plot_div = plot(fig, include_plotlyjs=False, output_type='div', config={'displayModeBar': False})
    if show_mode:
        fig.show()
    return plot_div


# 1e alternative - graph of number of tweets per day
def activity():
    global covid
    tweet_d = covid['tweet_date'].value_counts().to_frame().reset_index().rename(
        columns={'index': 'date', 'tweet_date': 'count'})
    tweet_d['date'] = pd.to_datetime(tweet_d['date'])
    tweet_d = tweet_d.sort_values('date', ascending=False)
    fig = go.Figure(go.Scatter(x=tweet_d['date'],
                               y=tweet_d['count'],
                               mode='markers+lines',
                               name="Submissions",
                               marker_color='dodgerblue'))

    fig.update_layout(template="plotly_dark", )

    plot_div = plot(fig, include_plotlyjs=False, output_type='div', config={'displayModeBar': False})
    if show_mode:
        fig.show()
    return plot_div

    # 2e - hashtags


def hashtag():
    covid['hash'] = covid['text'].apply(lambda x: find_hash(x))
    hashtags = list(covid[(covid['hash'].notnull()) & (covid['hash'] != "")]['hash'])
    hashtags = [each_string.lower() for each_string in hashtags]
    hash_df = dict(Counter(hashtags))
    top_hash = pd.DataFrame(list(hash_df.items()), columns=['word', 'count']).sort_values('count', ascending=False)[:20]
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
    global covid
    devices = covid['source'].value_counts().to_frame().reset_index().rename(
        columns={'index': 'source', 'source': 'count'})[:15]
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
    global covid
    covid['refine_text'] = covid['text'].str.lower()
    covid['refine_text'] = covid['refine_text'].apply(lambda x: remove_tag(str(x)))
    covid['refine_text'] = covid['refine_text'].apply(lambda x: remove_mention(str(x)))
    covid['refine_text'] = covid['refine_text'].apply(lambda x: remove_hash(str(x)))
    covid['refine_text'] = covid['refine_text'].apply(lambda x: remove_newline(x))
    covid['refine_text'] = covid['refine_text'].apply(lambda x: remove_url(x))
    covid['refine_text'] = covid['refine_text'].apply(lambda x: remove_number(x))
    covid['refine_text'] = covid['refine_text'].apply(lambda x: remove_punct(x))
    covid['refine_text'] = covid['refine_text'].apply(lambda x: remove_thi_amp_ha_words(x))
    covid['refine_text'] = covid['refine_text'].apply(lambda x: text_strip(x))

    covid['text_length'] = covid['refine_text'].str.split().map(lambda x: len(x))
    fig = go.Figure(data=go.Violin(y=covid['text_length'], box_visible=True, line_color='black',
                                   meanline_visible=True, fillcolor='lightblue ', opacity=0.6,
                                   x0='Tweet Text Length '))

    fig.update_layout(yaxis_zeroline=False, template="plotly_dark", )

    plot_div = plot(fig, include_plotlyjs=False, output_type='div', config={'displayModeBar': False})
    return plot_div


def get_israel_data():
    # vars for Israel
    all_dates = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/'
                            'csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
    all_dates = [date for date in all_dates.columns if date[0].isdigit()]
    path_url_death = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/' \
                     'csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
    path_url_confirmed = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/' \
                         'csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    path_url_recoverd = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/' \
                        'csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'

    # data pre-processing and preparation
    drop_list = ['Lat', 'Long', 'Country/Region', 'Province/State']
    # get death cases info
    countries_csv_death = pd.read_csv(path_url_death)
    mask = [True if value == 'Israel' else False for value in countries_csv_death['Country/Region']]
    countries_data_death = countries_csv_death[mask]
    countries_data_death = countries_data_death.drop(columns=drop_list)
    countries_data_death['criteria'] = 'deaths'
    # get confirmed cases info
    countries_csv_confirmed = pd.read_csv(path_url_confirmed)
    mask = [True if value == 'Israel' else False for value in countries_csv_confirmed['Country/Region']]
    countries_csv_confirmed = countries_csv_confirmed[mask]
    countries_csv_confirmed = countries_csv_confirmed.drop(columns=drop_list)
    countries_csv_confirmed['criteria'] = 'confirmed'
    # get recovered cases info
    countries_csv_recovered = pd.read_csv(path_url_recoverd)
    mask = [True if value == 'Israel' else False for value in countries_csv_recovered['Country/Region']]
    countries_csv_recovered = countries_csv_recovered[mask]
    countries_csv_recovered = countries_csv_recovered.drop(columns=drop_list)
    countries_csv_recovered['criteria'] = 'recovered'
    # add the data to a single dataframe
    israel_data = countries_data_death.append(countries_csv_confirmed, ignore_index=True) \
        .append(countries_csv_recovered, ignore_index=True)
    return israel_data


def israel():
    israel_data = get_israel_data()
    israel_data = israel_data.drop(columns='criteria')
    deaths = israel_data.values[0]
    confirmed = israel_data.values[1]
    recovered = israel_data.values[2]
    dates = israel_data.columns

    deaths_scatter = go.Scatter(name='Deaths', x=dates, y=deaths, line=dict(color="#fae839", width=4))
    confirmed_scatter = go.Scatter(name='Confirmed', x=dates, y=confirmed, line=dict(color="#132f65", width=4))
    recovered_scatter = go.Scatter(name='Recovered', x=dates, y=recovered, line=dict(color="#8a8678", width=4))

    fig = go.Figure(data=[deaths_scatter, confirmed_scatter, recovered_scatter])

    fig.update_layout(showlegend=False, margin=dict(t=0, l=0, r=0, b=200),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      hovermode="closest",
                      yaxis=dict(automargin=True, gridcolor="#32325d"),
                      xaxis=dict(automargin=True, showgrid=False),
                      font=dict(color='#ced4da'),
                      )

    plot_div = plot(fig, include_plotlyjs=False, output_type='div', config={'displayModeBar': False})
    return plot_div
