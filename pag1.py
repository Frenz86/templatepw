import streamlit as st
import plotly.figure_factory as ff
from PIL import Image
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import date
import joblib 

def show_footer():
    st.markdown("***")
    st.markdown("**Like this infomanager tool?** Follow me on "
                "[Twitter](https://twitter.com/xxxxxx).")
def main():
    st.title('Prophet Test')

    url ="https://frenzy86.s3.eu-west-2.amazonaws.com/timeseries/Data/airline_passengers.csv"

    df = pd.read_csv(url)
    st.dataframe(df)
    df.columns = ['ds','y']
    df['ds'] = pd.to_datetime(df['ds'])

    model = joblib.load('model_prophet.pkl')
    future = model.make_future_dataframe(50, freq='MS')

    #prediction
    forecast = model.predict(future)
    fig = plot_plotly(model, forecast)
    fig.update_layout(title="Forecast Air Passengers",
                    yaxis_title='Air Passengers',
                    xaxis_title="Date",
                    )
    #################### vertical line ############
    #fig.add_vline(x=date.today(), line_width=3, line_dash="dash", line_color="red")

    ##############################################
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()


