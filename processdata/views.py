import json

from django.http import HttpResponse
from django.shortcuts import render

from . import getdata, maps


def index(request):
    context = {
        'ring': maps.ring(),
        'israel': maps.israel(),
    }
    return render(request, template_name='index.html', context=context)


def twitterpage(request):
    context = {
        'violin': maps.violin(),
        'hashtag': maps.hashtag(),
        'activity': maps.activity(),
        'sources': maps.sources(),
    }
    return render(request, template_name='pages/twitter.html', context=context)


def globalpage(request):
    context = {
        'play': maps.play(),
        'marker': maps.marker(),
        'death_ratio': maps.death_ratio()
    }
    return render(request, template_name='pages/global.html', context=context)


def report(request):
    df = getdata.daily_report(date_string=None)
    df = df[['Confirmed', 'Deaths', 'Recovered']].sum()
    death_rate = f'{(df.Deaths / df.Confirmed) * 100:.02f}%'

    data = {
        'num_confirmed': int(df.Confirmed),
        'num_recovered': int(df.Recovered),
        'num_deaths': int(df.Deaths),
        'death_rate': death_rate
    }

    data = json.dumps(data)

    return HttpResponse(data, content_type='application/json')


def trends(request):
    df = getdata.percentage_trends()

    data = {
        'confirmed_trend': int(round(df.Confirmed)),
        'deaths_trend': int(round(df.Deaths)),
        'recovered_trend': int(round(df.Recovered)),
        'death_rate_trend': float(df.Death_rate)
    }

    data = json.dumps(data)

    return HttpResponse(data, content_type='application/json')


def global_cases(request):
    df = getdata.global_cases()
    return HttpResponse(df.to_json(orient='records'), content_type='application/json')


def realtime_growth(request):
    import pandas as pd
    df = getdata.realtime_growth()

    df.index = pd.to_datetime(df.index)
    df.index = df.index.strftime('%Y-%m-%d')

    return HttpResponse(df.to_json(orient='columns'), content_type='application/json')


def daily_growth(request):
    df_confirmed = getdata.daily_confirmed()[['date', 'World']]
    df_deaths = getdata.daily_deaths()[['date', 'World']]

    df_confirmed = df_confirmed.set_index('date')
    df_deaths = df_deaths.set_index('date')

    json_string = '{' + \
                  '"confirmed": ' + df_confirmed.to_json(orient='columns') + ',' + \
                  '"deaths": ' + df_deaths.to_json(orient='columns') + \
                  '}'

    return HttpResponse(json_string, content_type='application/json')


def daily_growth2(request):
    df_confirmed = getdata.daily_confirmed()[['date', 'World']]
    df_deaths = getdata.daily_deaths()[['date', 'World']]

    df_confirmed = df_confirmed.set_index('date')
    df_deaths = df_deaths.set_index('date')

    json_string = '{' + \
                  '"confirmed": ' + df_confirmed.to_json(orient='columns') + ',' + \
                  '"deaths": ' + df_deaths.to_json(orient='columns') + \
                  '}'

    return HttpResponse(json_string, content_type='application/json')


def daily_report(request):
    df = getdata.daily_report()

    df.drop(['FIPS', 'Admin2', 'Province_State', 'Country_Region', 'Last_Update', 'Deaths', 'Recovered', 'Active',
             'Incident_Rate', 'Case_Fatality_Ratio'], axis=1, inplace=True)

    return HttpResponse(df.to_json(orient='columns'), content_type='application/json')
