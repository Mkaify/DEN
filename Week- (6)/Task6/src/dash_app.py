
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd

df = pd.read_csv("data/processed_spacex.csv")

app = dash.Dash(__name__)
app.title = "SpaceX Landing Explorer"

app.layout = html.Div(children=[
    html.H2("SpaceX First Stage Landing — Explorer"),
    html.Div([
        html.Label("Launch Site"),
        dcc.Dropdown(id='site', options=[{'label': s, 'value': s} for s in sorted(df['launch_site'].dropna().unique())], value=None),
    ], style={'maxWidth':'420px'}),
    dcc.Graph(id="ts"),
    dcc.Graph(id="scatter"),
])

@app.callback(Output("ts","figure"), Input("site","value"))
def _ts(site):
    d = df if site is None else df[df['launch_site']==site]
    yearly = d.groupby("year")['landing_success'].mean().reset_index()
    fig = px.line(yearly, x="year", y="landing_success", title="Landing Success Rate by Year")
    return fig

@app.callback(Output("scatter","figure"), Input("site","value"))
def _sc(site):
    d = df if site is None else df[df['launch_site']==site]
    fig = px.scatter(d, x="payload_mass_kg", y="flight_number", color="landing_success",
                     hover_data=["rocket_name","orbit"], title="Payload vs Flight Number")
    return fig

if __name__ == "__main__":
    app.run(debug=True)
