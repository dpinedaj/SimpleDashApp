import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd
import numpy as np

import psycopg2

# Creating dash application
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])


# Loading Data--------------------


connection = psycopg2.connect(
    host=""
    port=5432,
    user='postgres',
    password="",
    database=""
)
cursor = connection.cursor()


df = pd.read_sql('select * from data', con=connection,
                 parse_dates=['date'])

dfsinpicos = pd.read_sql('select * from DatosSinPicos', con=connection,
                         parse_dates=['date'])

ypredic = pd.read_sql('select * from predictedFinal', con=connection,
                      parse_dates=['date'])

predicCompleta = pd.read_sql('select * from prediccioncompleta', con=connection,
                             parse_dates=['date'])


metricas = pd.read_sql('select * from metrics', con=connection)


ypredicgroupmonth = pd.read_sql(
    'select * from ypredictgroupmonth', con=connection)
predicCompletagroupmonth = pd.read_sql(
    'select * from predicCompletagroupmonth', con=connection)


dfgroupmonth = pd.read_sql('select * from datagroupmonth', con=connection)

dfredes = pd.read_csv('Redespred.csv')


# Graphs---------------------------------------------
# ? price per year graph
groupfigure = make_subplots(specs=[[{"secondary_y": True}]])
for año in set(df.year):
    groupfigure.add_scatter(x=df[df.year == año].date, y=df[df.year == año].precio_bolsa_nacional,
                            name=año, mode='lines', secondary_y=False, showlegend=False)
groupfigure.update_layout(
    title='Price across time',
    xaxis_title='Date',
    template='plotly_dark'
)

# mean price graph
dfaño = df.groupby('year', as_index=False).mean()

groupfigure.add_scatter(x=dfaño.year, y=dfaño.precio_bolsa_nacional,
                        mode='lines+markers', secondary_y=True, name='Mean Price')
groupfigure.update_yaxes(title_text="Price", secondary_y=False)
groupfigure.update_yaxes(title_text="Mean Price", secondary_y=True)


# price per year graph
groupwpeaks = make_subplots(specs=[[{"secondary_y": True}]])
for año in set(dfsinpicos.year):
    groupwpeaks.add_scatter(x=dfsinpicos[dfsinpicos.year == año].date,
                            y=dfsinpicos[dfsinpicos.year ==
                                         año].precio_bolsa_nacional,
                            name=año, mode='lines', secondary_y=False, showlegend=False)
groupwpeaks.update_layout(
    title='Price across time',
    xaxis_title='Date',
    template='plotly_dark'
)

# mean price graph
dfsinpicosaño = dfsinpicos.groupby('year', as_index=False).mean()

groupwpeaks.add_scatter(x=dfsinpicosaño.year, y=dfsinpicosaño.precio_bolsa_nacional,
                        mode='lines+markers', name='Mean Price')


# ? Prediction graph


figurepred = go.Figure()
figurepred.add_scatter(x=dfsinpicos.groupby('year', as_index=False).mean()['year'],
                       y=dfsinpicos.groupby('year', as_index=False).mean()[
    'precio_bolsa_nacional'],
    name='Observed', mode='lines')

figurepred.add_scatter(x=predicCompletagroupmonth.groupby('year', as_index=False).mean()['year'],
                       y=predicCompletagroupmonth.groupby('year', as_index=False).mean()[
    'precio'],
    name='Forecast', mode='lines')


figurepred.update_layout(
    title='Model',
    xaxis_title='Date',
    yaxis_title='Price',
    template='plotly_dark'
)


redespred = go.Figure()
redespred.add_scatter(x=np.arange(0, 20, 1),
                       y=dfredes.test,
                       name='Observed', mode='lines')

redespred.add_scatter(x=np.arange(0, 20, 1),
                       y=dfredes.pred,
                       name='Forecast', mode='lines')


redespred.update_layout(
    title='Model',
    xaxis_title='Period',
    yaxis_title='Price',
    template='plotly_dark'
)


# Creating layout


r1column_1 = [

    html.H5("Select Year"),
    dcc.Dropdown(
        id='year-select',
        options=[{'label': str(label), 'value': label}
                 for label in range(2000, 2030)],
        value=2019
    ),
    html.Br(),
    html.H5("Select Month"),
    dcc.Dropdown(
        id='month-select',
        options=[{'label': str(label), 'value': i+1}
                 for i, label in enumerate([
                     'January', 'February', 'March', 'April', 'May',
                     'June', 'July', 'August', 'September', 'October',
                     'November', 'December'])],
        value=1
    )]


r1column_2 = [
    dbc.Card([
        html.H4("Price [COP/KWh]", style={'textAlign': 'center','marginTop':30}),
        html.Hr(style={'background-color': '#ffffff'}),
        html.P(children='Real Value', style={'marginLeft': 20}),
        html.P(id='RealValue', children='', style={'marginLeft': 20}),
        html.Br(),
        html.P(children='Predicted Value', style={'marginLeft': 20}),
        html.P(id='PredValue', children='', style={'marginLeft': 20}),

    ],
        style={'height': 300})
]


table_header = [
    html.Thead(html.Tr([html.Th("Metric"), html.Th("Sarima"), html.Th('NN')]),
    style={'textAlign':'center'})
]

table_body = [html.Tbody([
    html.Tr([html.Th('MSE'), html.Th(138.46), html.Th(192.65)]),
    html.Tr([html.Th('RMSE'), html.Th(11.78), html.Th(13.88)])],
    style={'textAlign':'center'})]


# [html.Tr([html.Td(metric.upper()), html.Td(round(value, 2))]) for metric, value in zip(
#       metricas.metric.values, metricas['values'].values)]

cuadro_Issue = html.Div(html.P(
    [f'We found a Issue at {str(df.loc[df.precio_bolsa_nacional ==df.precio_bolsa_nacional.max(),"date"].values[0])[0:10]} with a value of {round(df.precio_bolsa_nacional.max(),2)} COP/KWh']
),
    style={'height': 100})

r1column_3 = [
    dbc.Table(table_header + table_body, bordered=True,
              dark=True,
              hover=True,
              responsive=True,
              striped=True,
              style={'height': 200},
              size='sm'
              ),
    cuadro_Issue

]

r1column_4 = [
    dbc.Card(
        [
            html.H4("RESUME", className="card-title",
                    style={'textAlign': 'center', 'marginTop': 20, 'color': 'black'}),
            html.Hr(style={'background-color': '#ffffff'}),
            html.P(
                children=[
                    """ The results were obtained for a training period of 19 years (2000-2019)
                     with daily data, comparing a Sarima model with a Neural Network model, using datetime series and the 
                     value of the stock price.
                  """
                ],

                style={'color': 'black', 'marginLeft': 5, 'marginTop': 15}

            )
        ],
        color="secondary",
        style={'height': 300})
]


rvd_content = dbc.Card(
    dbc.CardBody(
        [
            html.P("Real Data", className="card-text"),
            dcc.Graph(id='rvdgraph',
                      figure=groupfigure, style={'margin': 20, 'width': 1000}),
        ]
    ),
    className="mt-3",
)

rve_content = dbc.Card(
    dbc.CardBody(
        [
            html.P("Data without Peaks", className="card-text"),
            dcc.Graph(id='rvegraph',
                      figure=groupwpeaks, style={'margin': 20, 'width': 1000}),
        ]
    ),
    className="mt-3",
)


rf1_content = dbc.Card(
    dbc.CardBody(
        [
            html.P("Seasonal Behavior", className="card-text"),
            html.Img(src="https://i.ibb.co/yhPn9SF/model-Seasonal.png",
                     alt="model-Seasonal", style={'height': '90%', 'width': '90%', 'marginLeft': 25}),
        ]
    ),
    className="mt-3",
)


rf2_content = dbc.Card(
    dbc.CardBody(
        [
            html.P("Model Diagnostics", className="card-text"),
            html.Img(src="https://i.ibb.co/9H1bdGn/Model-Diagnostics.png",
                     alt="Model-Diagnostics", style={'height': '90%', 'width': '90%', 'marginLeft': 25}),
        ]
    ),
    className="mt-3",
)
rf3_content = dbc.Card(
    dbc.CardBody(
        [
            html.P("Sarima Model", className="card-text"),
            dcc.Graph(id='rf3graph',
                      figure=figurepred, style={'margin': 20, 'width': 1000}),
        ]
    ),
    className="mt-3",
)
rf4_content = dbc.Card(
    dbc.CardBody(
        [
            html.P("Neural Network", className="card-text"),
            dcc.Graph(id='rf4graph',
                      figure=redespred, style={'margin': 20, 'width': 1000}),
        ]
    ),
    className="mt-3",
)

tabsgraph = dbc.Tabs(
    [
        dbc.Tab(rvd_content, label="RD"),
        dbc.Tab(rve_content, label="DwoP"),
        dbc.Tab(rf1_content, label='SS'),
        dbc.Tab(rf2_content, label='MD'),
        dbc.Tab(rf3_content, label='Sarima'),
        dbc.Tab(rf4_content, label='NN')

    ]
)


r2column_1 = [tabsgraph]


body = dbc.Container(

    [


        dbc.Row(
            [
                dbc.Col(r1column_1),
                dbc.Col(r1column_2),
                dbc.Col(r1column_3),
                dbc.Col(r1column_4, width=4)


            ]
        ),

        html.Br(),

        dbc.Row(
            [
                dbc.Col(r2column_1)


            ]
        )



    ], style={'margin': 50}

)


header = dbc.CardHeader(
    html.H1(
        children='Time Series Analysis: An application for Colombian Electricity Market')
)


# Setting layout for the application
app.layout = html.Div(
    children=[header, body])


@app.callback(
    [
        dash.dependencies.Output('RealValue', 'children'),
        dash.dependencies.Output('PredValue', 'children')

    ],
    [
        dash.dependencies.Input('year-select', 'value'),
        dash.dependencies.Input('month-select', 'value'),


    ]

)
def filter_date(year, month):
    if year != None and month != None:
        if year < 2020:
            Precio = round(dfgroupmonth.loc[(dfgroupmonth.year == year) & (dfgroupmonth.month == month),
                                            'precio_bolsa_nacional'], 2)
            Preciopred = round(predicCompletagroupmonth.loc[(predicCompletagroupmonth.year == year) & (predicCompletagroupmonth.month == month),
                                                            'precio'], 2)
        else:
            Precio = 'NA'
            Preciopred = round(ypredicgroupmonth.loc[(ypredicgroupmonth.year == year) & (ypredicgroupmonth.month == month),
                                                     'precio'], 2)

    else:
        Precio = 'Insert Year and Month'
        Preciopred = 'Insert Year and Month'

    return(Precio, Preciopred)


    # Starting the server
if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0',port='8090')
