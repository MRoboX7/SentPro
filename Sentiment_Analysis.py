# Libraries
import pandas as pd
import numpy as np
import webbrowser
import pickle
import dash
import os
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib import pyplot as plt
#from wordcloud import WordCloud, STOPWORDS

# Global Variables
project_name = "Sentiment Analysis with Insights"
app = dash.Dash(external_stylesheets=[dbc.themes.QUARTZ])
# Functions

def load_model():
    global scrappedReviews
    scrappedReviews = pd.read_csv('scrappedReviews.csv')

    global pickle_model
    file = open("sentiment_analysis.pkl", 'rb')
    pickle_model = pickle.load(file)

    global vocab
    file = open("vocab_features.pkl", 'rb')
    vocab = pickle.load(file)
    
    
def pie_construct():
    temp = []
    for i in scrappedReviews['Reviews']:
        if(check_review(i)[0] == 1):
            temp.append("Positive")
        else:
            temp.append("Negative")
    scrappedReviews['Sentiment'] = temp
    global positive,negative
    positive = len(scrappedReviews[scrappedReviews['Sentiment']=="Positive"])
    negative = len(scrappedReviews[scrappedReviews['Sentiment']=="Negative"])
    explode = (0.1,0)  

    langs = ['Positive', 'Negative',]
    students = [positive,negative]
    colors = ['#41fc1c','red']
    plt.pie(students,explode=explode,startangle=90,colors=colors, labels = langs,autopct='%1.2f%%')
    cwd = os.getcwd()
    if 'assets' not in os.listdir(cwd):
        os.makedirs(cwd+'/assets')
    plt.savefig('assets/sentiment.png')
    pass
    

def open_browser():
    webbrowser.open_new('http://127.0.0.1:8050/')


def check_review(reviewText):

    transformer = TfidfTransformer()
    vec = CountVectorizer(decode_error="replace", vocabulary=vocab)
    vectorised_review = transformer.fit_transform(vec.fit_transform([reviewText]))

    return pickle_model.predict(vectorised_review)


def create_app_ui():
    main_layout = dbc.Col(
        html.Div(
        [
            dbc.NavbarSimple(children=[dbc.NavItem(dbc.NavLink("Analysis", href="http://127.0.0.1:8050/")),dbc.NavItem(dbc.NavLink("Visualisation", href="#"))],
                    brand=project_name,brand_href="http://127.0.0.1:8050/",color="primary",dark=True,),
            dbc.Container([
                html.H1(id = 'heading', children = "Review Analysis",style={'margin-top':'25px' , 'text-align':'Left', 'margin-left' : '0px'}),
                html.P(id = 'Custom_rev_inst' , children = "Enter Your Review Here :)"),
                dbc.InputGroup(children = 
                [
                    dbc.Input(id="input-group-button-input", placeholder="Enter Review Here ..."),
                    dbc.Button("Predict", id="input-group-button", n_clicks=0 , outline = False),
                ],
                style={'margin-top':'5px'}),
                dbc.Col([dbc.Badge(children = "Neutral",id = "pred_result", color="transparent", className="mx-0 w-100 fs-2 border border-2 border-transparent shadow-lg")],className="align-items-md-stretch my-2 p-0"),
                dbc.Card([dbc.Row([dbc.Col(dbc.CardImg(src="/assets/sentiment.png",className="img-fluid rounded-start",),className=" m-2",),
                dbc.Col(dbc.CardBody([html.H2("Positives : " + str(positive), className="card-title"),html.Small("Reviews With Rating More Than 3",className="card-text text-muted"),html.H2("Negatives : " + str(negative), className="card-title"),html.Small("Reviews With Rating Less Than 3",className="card-text text-muted")]),className="col-sm-8 colalign-self-end",),
                ],className="g-0 d-flex align-items-center",)],className="my-3",style={"maxWidth": "1320px"},),
                html.Div(dbc.Table.from_dataframe(scrappedReviews, striped=True, bordered=True, hover=True, responsive=True),style={'overflow-y':'scroll','max-height':'300px'}
                )],)
        ],
        className="h-1 p-0 bg-transparent border border-4 rounded-1 shadow-lg",style={'margin-top':'15px','margin-left':'5px','margin-right':'5px','margin-bottom':'15px'}
        ),
        sm=12,
        )
    return main_layout
    

@app.callback(
    [Output("pred_result", "children"),Output("pred_result", "color")],
    [Input("input-group-button", "n_clicks")],
    [State( 'input-group-button-input'  ,   'value'  )]
)
def update_app_ui(n_clicks,value):

    response = check_review(value)

    if (response[0] == 0):
        result = 'Negative'
        clr = 'Red'
    elif (response[0] == 1 ):
        result = 'Positive'
        clr = 'Green'
    else:
        result = 'Neutral'
        clr = 'Transparent'

    return result, clr


def main():
    load_model()
    #pie_construct()
    open_browser()

    global scrappedReviews
    global project_name
    global app

    app.title = project_name
    app.layout = create_app_ui()
    app.run_server()

    # Nullifying The Global Variables
    project_name = None
    scrappedReviews = None
    app = None


# Call_to_Main_Function
if __name__ == '__main__':
    main()
