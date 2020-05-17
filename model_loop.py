from functools import reduce
from typing import Generator, Tuple, Dict, Any, Optional
import os
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib
from bs4 import BeautifulSoup
import requests
import ipyvuetify as v
from traitlets import Unicode, List
from datetime import date, datetime, timedelta
import time
import altair as alt
from collections import namedtuple
from scipy.integrate import odeint

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# If the secrete provided matches the ENV, proceeed with the app
hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
    """
st.markdown(hide_menu_style, unsafe_allow_html=True)

url_parameters = "https://raw.githubusercontent.com/gabai/stream_KH/master/Parameters_code.csv"
parameters = pd.read_csv(url_parameters, index_col='model')

url_erie = 'https://raw.githubusercontent.com/gabai/stream_KH/master/Cases_Erie.csv'
erie = pd.read_csv(url_erie)
erie['Date'] = pd.to_datetime(erie['Date'])

# Convert columns to dates columns
date_cols = ['start_date', 'intervention1', 'intervention2', 'intervention3', 'step1', 'step2', 'step3', 'step4']
for i in date_cols:
        parameters[i] = pd.to_datetime(parameters[i])
        
# Add Calculated Variable Parameerse Dataset
parameters['start_day'] = 1
parameters['int1_delta'] = (parameters.intervention1 - parameters.start_date)
parameters['int2_delta'] = (parameters.intervention2 - parameters.start_date)
parameters['int3_delta'] = (parameters.intervention3 - parameters.start_date)
parameters['step1_delta'] = (parameters.step1 - parameters.start_date)
parameters['step2_delta'] = (parameters.step2 - parameters.start_date)
parameters['step3_delta'] = (parameters.step3 - parameters.start_date)
parameters['step4_delta'] = (parameters.step4 - parameters.start_date)

# Models
models_dict = {0:'SIR', 1:'SEIR', 2:'SEIRv2', 3:'SEIRaR0', 4:'SEIRaR0v2', 5:'SEIRhSD', 
               6:'SEIRDJA', 7:'SEIRDJAv2', 8:'SEIRDJAmit'}
               
# Groups
groups = ['hosp', 'icu', 'vent']

# Variables
n_days = 120
plot_projection_days = n_days - 10
as_date = False

incubation_period = 5.2
recovery_days = 11.0
infectious_period = 3.0
hosp_lag = 4.0
recovered = 0.0
regional_hosp_share = 1.0
current_hosp = erie.iloc[-1,1]
total_infections = erie.iloc[-1, 4]
beta_decay = 0.0

# Loop to create dictionary of parameters per/model
for i in models_dict:
    globals()['model'+str(i)] = parameters.iloc[i].to_dict()

models = {0: model0, 1:model1, 2:model2, 3:model3, 4:model4, 5:model5, 6:model6, 7:model7, 8:model8}


def sir(
    s: float, i: float, r: float, beta: float, gamma: float, n: float
    ) -> Tuple[float, float, float]:
    """The SIR model, one time step."""
    s_n = (-beta * s * i) + s
    i_n = (beta * s * i - gamma * i) + i
    r_n = gamma * i + r
    if s_n < 0.0:
        s_n = 0.0
    if i_n < 0.0:
        i_n = 0.0
    if r_n < 0.0:
        r_n = 0.0

    scale = n / (s_n + i_n + r_n)
    return s_n * scale, i_n * scale, r_n * scale
    
def gen_sir(
    s: float, i: float, r: float, beta: float, gamma: float, n_days: int
    ) -> Generator[Tuple[float, float, float], None, None]:
    """Simulate SIR model forward in time yielding tuples."""
    s, i, r = (float(v) for v in (s, i, r))
    n = s + i + r
    for _ in range(n_days + 1):
        yield s, i, r
        s, i, r = sir(s, i, r, beta, gamma, n)

def sim_sir(
    s: float, i: float, r: float, beta: float, gamma: float, n_days: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate the SIR model forward in time."""
    s, i, r = (float(v) for v in (s, i, r))
    n = s + i + r
    s_v, i_v, r_v = [s], [i], [r]
    for day in range(n_days):
        s, i, r = sir(s, i, r, beta, gamma, n)
        s_v.append(s)
        i_v.append(i)
        r_v.append(r)

    return (
        np.array(s_v),
        np.array(i_v),
        np.array(r_v),
    )
    
def sim_sir_df(
    p) -> pd.DataFrame:
    """Simulate the SIR model forward in time.

    p is a Parameters instance. for circuluar dependency reasons i can't annotate it.
    """
    return pd.DataFrame(
        data=gen_sir(S, total_infections, recovered, beta, gamma, n_days),
        columns=("Susceptible", "Infected", "Recovered"),
    )

def seir(
    s: float, e: float, i: float, r: float, beta: float, gamma: float, alpha: float, n: float
    ) -> Tuple[float, float, float, float]:
    """The SIR model, one time step."""
    s_n = (-beta * s * i) + s
    e_n = (beta * s * i) - alpha * e + e
    i_n = (alpha * e - gamma * i) + i
    r_n = gamma * i + r
    if s_n < 0.0:
        s_n = 0.0
    if e_n < 0.0:
        e_n = 0.0
    if i_n < 0.0:
        i_n = 0.0
    if r_n < 0.0:
        r_n = 0.0

    scale = n / (s_n + e_n+ i_n + r_n)
    return s_n * scale, e_n * scale, i_n * scale, r_n * scale

def sim_seir(
    s: float, e:float, i: float, r: float, beta: float, gamma: float, alpha: float, n_days: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate the SIR model forward in time."""
    s, e, i, r = (float(v) for v in (s, e, i, r))
    n = s + e + i + r
    s_v, e_v, i_v, r_v = [s], [e], [i], [r]
    for day in range(n_days):
        s, e, i, r = seir(s, e, i, r, beta, gamma, alpha, n)
        s_v.append(s)
        e_v.append(e)
        i_v.append(i)
        r_v.append(r)

    return (
        np.array(s_v),
        np.array(e_v),
        np.array(i_v),
        np.array(r_v),
    )

def gen_seir(
    s: float, e: float, i: float, r: float, beta: float, gamma: float, alpha: float, n_days: int
    ) -> Generator[Tuple[float, float, float, float], None, None]:
    """Simulate SIR model forward in time yielding tuples."""
    s, e, i, r = (float(v) for v in (s, e, i, r))
    n = s + e + i + r
    for _ in range(n_days + 1):
        yield s, e, i, r
        s, e, i, r = seir(s, e, i, r, beta, gamma, alpha, n)
# phase-adjusted https://www.nature.com/articles/s41421-020-0148-0     

def sim_seir_decay(
    s: float, e:float, i: float, r: float, beta: float, gamma: float, alpha: float, n_days: int,
    decay1:float, decay2:float, decay3: float, decay4: float, end_delta: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate the SIR model forward in time."""
    s, e, i, r = (float(v) for v in (s, e, i, r))
    n = s + e + i + r
    s_v, e_v, i_v, r_v = [s], [e], [i], [r]
    for day in range(n_days):
        if start_day<=day<=int1_delta:
            beta_decay=beta*(1-decay1)
        elif int1_delta<=day<=int2_delta:
            beta_decay=beta*(1-decay2)
        elif int2_delta<=day<=int3_delta:
            beta_decay=beta*(1-decay3)
        elif int3_delta<=day<=end_delta:
            beta_decay=beta*(1-decay4)
        else:
            beta_decay=beta*(1-decay5)
        s, e, i, r = seir(s, e, i, r, beta_decay, gamma, alpha, n)
        s_v.append(s)
        e_v.append(e)
        i_v.append(i)
        r_v.append(r)

    return (
        np.array(s_v),
        np.array(e_v),
        np.array(i_v),
        np.array(r_v),
    )

def get_dispositions(
    patient_state: np.ndarray, rates: Tuple[float, ...], regional_hosp_share: float = 1.0
    ) -> Tuple[np.ndarray, ...]:
    """Get dispositions of infected adjusted by rate and market_share."""
    return (*(patient_state * rate * regional_hosp_share for rate in rates),)

def build_admissions_df(
    dispositions) -> pd.DataFrame:
    """Build admissions dataframe from Parameters."""
    days = np.array(range(0, n_days + 1))
    data_dict = dict(
        zip(
            ["day", "hosp", "icu", "vent"], 
            [days] + [disposition for disposition in dispositions],
        )
    )
    projection = pd.DataFrame.from_dict(data_dict)
    
    # New cases
    projection_admits = projection.iloc[:-1, :] - projection.shift(1)
    projection_admits["day"] = range(projection_admits.shape[0])
    return projection_admits

def build_census_df(
    projection_admits: pd.DataFrame) -> pd.DataFrame:
    """ALOS for each category of COVID-19 case (total guesses)"""
    n_days = np.shape(projection_admits)[0]
    los_dict = {
    "hosp": hosp_los, "icu": icu_los, "vent": vent_los}

    census_dict = dict()
    for k, los in los_dict.items():
        census = (
            projection_admits.cumsum().iloc[:-los, :]
            - projection_admits.cumsum().shift(los).fillna(0)
        ).apply(np.ceil)
        census_dict[k] = census[k]

    census_df = pd.DataFrame(census_dict)
    census_df["day"] = census_df.index
    census_df = census_df[["day", "hosp", "icu", "vent"]]
    
    census_df = census_df.head(n_days-10)
    
    return census_df

# Add dates #
def add_date_column(
    df: pd.DataFrame, drop_day_column: bool = False, date_format: Optional[str] = None,
    ) -> pd.DataFrame:
    """Copies input data frame and converts "day" column to "date" column

    Assumes that day=0 is today and allocates dates for each integer day.
    Day range can must not be continous.
    Columns will be organized as original frame with difference that date
    columns come first.

    Arguments:
        df: The data frame to convert.
        drop_day_column: If true, the returned data frame will not have a day column.
        date_format: If given, converts date_time objetcts to string format specified.

    Raises:
        KeyError: if "day" column not in df
        ValueError: if "day" column is not of type int
    """
    if not "day" in df:
        raise KeyError("Input data frame for converting dates has no 'day column'.")
    if not pd.api.types.is_integer_dtype(df.day):
        raise KeyError("Column 'day' for dates converting data frame is not integer.")

    df = df.copy()
    # Prepare columns for sorting
    non_date_columns = [col for col in df.columns if not col == "day"]

    # Allocate (day) continous range for dates
    n_days = int(df.day.max())
    start = start_date
    end = start + timedelta(days=n_days + 1)
    # And pick dates present in frame
    dates = pd.date_range(start=start, end=end, freq="D")[df.day.tolist()]

    if date_format is not None:
        dates = dates.strftime(date_format)

    df["date"] = dates

    if drop_day_column:
        df.pop("day")
        date_columns = ["date"]
    else:
        date_columns = ["day", "date"]

    # sort columns
    df = df[date_columns + non_date_columns]

    return df


# Side-Bar
model_options = st.sidebar.radio(
    "Service", ('Inpatient', 'ICU', 'Ventilated'))
start_date = st.sidebar.date_input(
    "Suspected first contact", datetime(2020,3,1))
start_day = 1

# Slider and Date
n_days = st.slider("Number of days to project", 30, 200, 120, 1, "%i")
as_date = st.checkbox(label="Present result as dates", value=False)


# Model Loop
for m in models:
    # Specific Variables for Models using Parameters Table
    S = (models[m])['S']
    doubling_time = (models[m])['doubling_time']
    relative_contact_rate = (models[m])['relative_contact_rate']
    print(relative_contact_rate)
    intrinsic_growth_rate = 2 ** (1 / doubling_time) - 1
    hosp_rate = (models[m])['hosp_rate']
    hosp_los = (models[m])['hosp_los']
    icu_rate = (models[m])['icu_rate']
    icu_los = (models[m])['icu_los']
    vent_rate = (models[m])['vent_rate']
    vent_los = (models[m])['vent_los']
    alpha = 1/incubation_period
    # Rates Tuples
    RateLos = namedtuple("RateLos", ("rate", "length_of_stay"))
    hospitalized = RateLos(hosp_rate, hosp_los)
    icu = RateLos(icu_rate, icu_los)
    ventilated = RateLos(vent_rate, vent_los)
    rates = tuple(each.rate for each in (hospitalized, icu, ventilated))
    lengths_of_stay = tuple(each.length_of_stay for each in (hospitalized, icu, ventilated))
    
    
#############################################################################################
    # SIR
    if (models[m])['base_model'] == 'SIR':
        # Calculate Model Specific Variables
        gamma = 1 / recovery_days
        beta = (intrinsic_growth_rate + gamma) / S * (1-relative_contact_rate) # {rate based on doubling time} / {initial S}
        r_t = beta / gamma * S # r_t is r_0 after distancing
        r_naught = (intrinsic_growth_rate + gamma) / gamma
        doubling_time_t = 1/np.log2(beta*S - gamma +1) # doubling time after distancing
        total_infections = current_hosp / 1 / hosp_rate
        ## SIR model
        s, i, r = sim_sir(S-2, 1, 1 ,beta, gamma, n_days)
        susceptible, infected, recovered = s, i, r
        i_hospitalized, i_icu, i_ventilated = get_dispositions(i, rates, 1)
        r_hospitalized, r_icu, r_ventilated = get_dispositions(r, rates, 1)
        dispositions = (i_hospitalized + r_hospitalized, i_icu + r_icu, i_ventilated + r_ventilated)
        hospitalized, icu, ventilated = (i_hospitalized, i_icu, i_ventilated)
        ## Build Admissions Datasets
        globals()['projection_admits'+str(m)] = build_admissions_df(dispositions)
        ## Build Census Table
        globals()['census_table'+str(m)] =  build_census_df(globals()['projection_admits'+str(m)])
        
        
###########################################################################
    # SEIR
    if (models[m])['base_model'] == 'SEIR':
        # Vairables for SEIR Models
        alpha = 1/incubation_period
        beta = ((alpha+intrinsic_growth_rate)*(intrinsic_growth_rate + (1/infectious_period))) / (alpha*S)         
        gamma=1/infectious_period
        recovered=0.0 # Some issues here with recovered pulling from prior model as table. 
        exposed=beta*S*total_infections
        S2=S-exposed-total_infections
        ## SEIR Model
        s, e, i, r = sim_seir(S-2, 1 ,1, recovered, beta, gamma, alpha, n_days)
        susceptible, exposed, infected, recovered = s, e, i, r
        i_hospitalized, i_icu, i_ventilated = get_dispositions(i, rates, 1)
        r_hospitalized, r_icu, r_ventilated = get_dispositions(r, rates, 1)
        dispositions = (i_hospitalized + r_hospitalized, i_icu + r_icu, i_ventilated + r_ventilated)
        hospitalized, icu, ventilated = (i_hospitalized, i_icu, i_ventilated)
        ## Build Admissions Datasets
        globals()['projection_admits'+str(m)] = build_admissions_df(dispositions)
        ## Build Census Table
        globals()['census_table'+str(m)] = build_census_df(globals()['projection_admits'+str(m)])
        
        
###########################################################################
    if (models[m])['base_model'] == 'SEIRDJA':
        # Variables for modified SEIR Models w/ R_0 Phase Adjustment
        # Need to rename beta
        beta3 = ((alpha+intrinsic_growth_rate)*(intrinsic_growth_rate + (1/infectious_period))) / (alpha*S) *(1-relative_contact_rate)
    #if model+i = range(5, 7):
        # Variables for modified SEIR Models w/ adjusted R_O and deaths
    #else:
        # Variables for final model
        
    
        
###########################################################################
    if (models[m])['base_model'] == 'SEIRDJA':
        # Variables for modified SEIR Models w/ R_0 Phase Adjustment
        # Need to rename beta
        beta3 = ((alpha+intrinsic_growth_rate)*(intrinsic_growth_rate + (1/infectious_period))) / (alpha*S) *(1-relative_contact_rate)
    #if model+i = range(5, 7):
        # Variables for modified SEIR Models w/ adjusted R_O and deaths
    #else:
        # Variables for final model
        
    
# Comparison of Single line graph - Hospitalized, ICU, Ventilated
if model_options == "Inpatient":
    columns_comp = {"hosp": "Hospitalized"}
    fold_comp = ["Hospitalized"]
if model_options == "ICU":
    columns_comp = {"icu": "ICU"}
    fold_comp = ["ICU"]
if model_options == "Ventilated":
    columns_comp = {"vent": "Ventilated"}
    fold_comp = ["Ventilated"]

       
# Admissions Graph
def regional_admissions_chart(
    projection_admits: pd.DataFrame, 
    plot_projection_days: int,
    as_date:bool = False) -> alt.Chart:
    """docstring"""
    
    projection_admits = projection_admits = projection_admits.rename(columns={"hosp": "Hospitalized", "icu": "ICU", "vent": "Ventilated"})
    
    tooltip_dict = {False: "day", True: "date:T"}
    if as_date:
        projection_admits = add_date_column(projection_admits)
        x_kwargs = {"shorthand": "date:T", "title": "Date"}
    else:
        x_kwargs = {"shorthand": "day", "title": "Days from initial infection"}
    
    return (
        alt
        .Chart(projection_admits.head(plot_projection_days))
        .transform_fold(fold=fold_comp)
        .mark_line(point=False)
        .encode(
            x=alt.X(**x_kwargs),
            y=alt.Y("value:Q", title="Daily admissions"),
            color="key:N",
            tooltip=[
                tooltip_dict[as_date],
                alt.Tooltip("value:Q", format=".0f", title="Admissions"),
                "key:N",
            ],
        )
        .interactive()
    )

admits_graph0 = regional_admissions_chart(projection_admits0, 
        plot_projection_days, 
        as_date=as_date)
        
admits_graph1 = regional_admissions_chart(projection_admits1, 
        plot_projection_days, 
        as_date=as_date)


# for m in models:
    # globals()['admit_graph'+str(m)] = regional_admissions_chart((globals()['projection_admits'+str(m)]), 
    # plot_projection_days, as_date=as_date)

st.altair_chart(
    #admits_graph_seir
    #+ 
    #admits_graph 
    #+ 
    #vertical1
    #+ admits_graph_ecases
    #+ 
    admits_graph0
    + admits_graph1
    #+ erie_admit24_line
    , use_container_width=True)



