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
parameters['int1_delta'] = (parameters.intervention1).days - (parameters.start_date).days
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

# Slider and Date
n_days = st.slider("Number of days to project", 30, 200, 120, 1, "%i")
as_date = st.checkbox(label="Present result as dates", value=False)

# Variables
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

# SIR Model
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

# SEIR Model
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


# SEIR w/ R_0 Adjustment
def sim_seir_decay(
    s: float, e:float, i: float, r: float, beta: float, gamma: float, alpha: float, n_days: int,
    decay1:float, decay2:float, decay3: float, decay4: float, step1_delta: int
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
        elif int3_delta<=day<=step1_delta:
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


# SEIR w/ adjusted R_0 and fatality
def seird(
    s: float, e: float, i: float, r: float, d: float, beta: float, gamma: float, alpha: float, n: float, fatal: float
    ) -> Tuple[float, float, float, float]:
    """The SIR model, one time step."""
    s_n = (-beta * s * i) + s
    e_n = (beta * s * i) - alpha * e + e
    i_n = (alpha * e - gamma * i) + i
    r_n = (1-fatal)*gamma * i + r
    d_n = (fatal)*gamma * i +d
    if s_n < 0.0:
        s_n = 0.0
    if e_n < 0.0:
        e_n = 0.0
    if i_n < 0.0:
        i_n = 0.0
    if r_n < 0.0:
        r_n = 0.0
    if d_n < 0.0:
        d_n = 0.0

    scale = n / (s_n + e_n+ i_n + r_n + d_n)
    return s_n * scale, e_n * scale, i_n * scale, r_n * scale, d_n * scale

def sim_seird_decay(
        s: float, e:float, i: float, r: float, d: float, beta: float, gamma: float, alpha: float, n_days: int,
        decay1:float, decay2:float, decay3: float, decay4: float, step1_delta: int, fatal: float
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulate the SIR model forward in time."""
        s, e, i, r, d= (float(v) for v in (s, e, i, r, d))
        n = s + e + i + r + d
        s_v, e_v, i_v, r_v, d_v = [s], [e], [i], [r], [d]
        for day in range(n_days):
            if start_day<=day<=int1_delta:
                beta_decay=beta*(1-decay1)
            elif int1_delta<=day<=int2_delta:
                beta_decay=beta*(1-decay2)
            elif int2_delta<=day<=int3_delta:
                beta_decay=beta*(1-decay3)
            elif int3_delta<=day<=step1_delta:
                beta_decay=beta*(1-decay4)
            else:
                beta_decay=beta*(1-decay5)
                s, e, i, r,d = seird(s, e, i, r, d, beta_decay, gamma, alpha, n, fatal)
                s_v.append(s)
                e_v.append(e)
                i_v.append(i)
                r_v.append(r)
                d_v.append(d)

            return (
                np.array(s_v),
                np.array(e_v),
                np.array(i_v),
                np.array(r_v),
                np.array(d_v)
            )

def seijcrd(
    s: float, e: float, i: float, j:float, c:float, r: float, d: float, beta: float, gamma: float, alpha: float, n: float, fatal_hosp: float, hosp_rate:float, icu_rate:float, icu_days:float,crit_lag:float, death_days:float
    ) -> Tuple[float, float, float, float]:
    """The SIR model, one time step."""
    s_n = (-beta * s * (i+j+c)) + s
    e_n = (beta * s * (i+j+c)) - alpha * e + e
    i_n = (alpha * e - gamma * i) + i
    j_n = hosp_rate * i * gamma + (1-icu_rate)* c *icu_days + j
    c_n = icu_rate * j * (1/crit_lag) - c *  (1/death_days)
    r_n = (1-hosp_rate)*gamma * i + (1-icu_rate) * (1/crit_lag)* j + r
    d_n = (fatal_hosp)* c * (1/crit_lag)+d
    if s_n < 0.0:
        s_n = 0.0
    if e_n < 0.0:
        e_n = 0.0
    if i_n < 0.0:
        i_n = 0.0
    if j_n < 0.0:
        j_n = 0.0
    if c_n < 0.0:
        c_n = 0.0
    if r_n < 0.0:
        r_n = 0.0
    if d_n < 0.0:
        d_n = 0.0

    scale = n / (s_n + e_n+ i_n + j_n+ c_n+ r_n + d_n)
    return s_n * scale, e_n * scale, i_n * scale, j_n* scale, c_n*scale, r_n * scale, d_n * scale

def betanew(t,beta):
    if start_day<= t <= int1_delta:
        beta_decay=beta*(1-decay1)
    elif int1_delta<=t<int2_delta:
        beta_decay=beta*(1-decay2)
    elif int2_delta<=t<=int3_delta:
        beta_decay=beta*(1-decay3)
    elif int3_delta<=t<=step1_delta:
        beta_decay=beta*(1-decay4)
    elif step1_delta<=t<=step2_delta:
        beta_decay=beta*(1-decay5)
    else:
        beta_decay=beta*(1-decay6)    
    return beta_decay

#The SIR model differential equations with ODE solver.
def derivdecay(y, t, N, beta, gamma1, gamma2, alpha, p, hosp,q,l,n_days, decay1, decay2, decay3, decay4, decay5, decay6, start_day, int1_delta, int2_delta, int3_delta, step1_delta, step2_delta, fatal_hosp ):
    S, E, A, I,J, R,D,counter = y
    dSdt = - betanew(t, beta) * S * (q*I + l*J + A)/N 
    dEdt = betanew(t, beta) * S * (q*I + l*J + A)/N   - alpha * E
    dAdt = alpha * E*(1-p)-gamma1*A
    dIdt = p* alpha* E - gamma1 * I- hosp*I
    dJdt = hosp * I -gamma2*J
    dRdt = (1-fatal_hosp)*gamma2 * J + gamma1*(A+I)
    dDdt = fatal_hosp * gamma2 * J
    counter = (1-fatal_hosp)*gamma2 * J
    return dSdt, dEdt,dAdt, dIdt, dJdt, dRdt, dDdt, counter

def sim_seaijrd_decay_ode(
    s, e,a,i, j,r, d, beta, gamma1, gamma2, alpha, n_days,decay1,decay2,decay3, decay4, decay5, decay6, start_day, int1_delta, int2_delta, int3_delta, step1_delta, step2_delta, fatal_hosp, p, hosp, q,
    l):
    n = s + e + a + i + j+ r + d
    rh=0
    y0= s,e,a,i,j,r,d, rh
    
    t=np.arange(0, n_days, step=1)
    ret = odeint(derivdecay, y0, t, args=(n, beta, gamma1, gamma2, alpha, p, hosp,q,l, n_days, decay1, decay2, decay3, decay4, decay5, decay6, start_day, int1_delta, int2_delta, int3_delta, step1_delta, step2_delta, fatal_hosp))
    S_n, E_n,A_n, I_n,J_n, R_n, D_n ,RH_n= ret.T
    
    return (S_n, E_n,A_n, I_n,J_n, R_n, D_n, RH_n)


# Create Tables
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
relative_contact_rate = st.sidebar.number_input(
    "Social distancing (% reduction in social contact) Unadjusted Model", 0, 100, value=0, step=5, format="%i")/100.0

decay1 = st.sidebar.number_input(
    "Social distancing (% reduction in social contact) in Week 0-2", 0, 100, value=0, step=5, format="%i")/100.0

intervention1 = st.sidebar.date_input(
    "Date of change Social Distancing - School Closure", datetime(2020,3,22))
int1_delta = (intervention1 - start_date).days
    
decay2 = st.sidebar.number_input(
    "Social distancing (% reduction in social contact) in Week 3 - School Closure", 0, 100, value=15, step=5, format="%i")/100.0

intervention2 = st.sidebar.date_input(
    "Date of change in Social Distancing - Closure Businesses, Shelter in Place", datetime(2020,3,28))
int2_delta = (intervention2 - start_date).days

decay3 = st.sidebar.number_input(
    "Social distancing (% reduction in social contact) from Week 3 to change in SD - After Business Closure%", 0, 100, value=45 ,step=5, format="%i")/100.0

intervention3 = st.sidebar.date_input(
    "NYS Facemask Mandate", datetime(2020,4,15))
int3_delta = (intervention3 - start_date).days

decay4 = st.sidebar.number_input(
    "NYS Facemask Mandate", 0, 100, value=55 ,step=5, format="%i")/100.0

step1 = st.sidebar.date_input(
    "Step 1 reduction in social distancing", datetime(2020,5,15))
# Delta from start and end date for decay4
step1_delta = (step1 - start_date).days

decay5 = st.sidebar.number_input(
    "Step 1 reduction in social distancing %", 0, 100, value=45 ,step=5, format="%i")/100.0

step2 = st.sidebar.date_input(
    "Step 2 reduction in social distancing", datetime(2020,6,15))
# Delta from start and end date for decay4
step2_delta = (step2 - start_date).days

decay6 = st.sidebar.number_input(
    "Step 2 reduction in social distancing %", 0, 100, value=35 ,step=5, format="%i")/100.0
    


# Model Loop
for m in models:
    # Specific Variables for Models using Parameters Table
    S = (models[m])['S']
    doubling_time = (models[m])['doubling_time']
    relative_contact_rate = (models[m])['relative_contact_rate']
    intrinsic_growth_rate = 2 ** (1 / doubling_time) - 1
    hosp_rate = (models[m])['hosp_rate']
    hosp_los = (models[m])['hosp_los']
    icu_rate = (models[m])['icu_rate']
    icu_los = (models[m])['icu_los']
    vent_rate = (models[m])['vent_rate']
    vent_los = (models[m])['vent_los']
    alpha = 1/incubation_period
    fatal = (models[m])['fatal']
    start_date = (models[m])['start_date']
    decay1 = (models[m])['decay1']
    decay2 = (models[m])['decay2']
    decay3 = (models[m])['decay3']
    decay4 = (models[m])['decay4']
    decay5 = (models[m])['decay5']
    step1_delta = (models[m])['step1'].days
    step2_delta = (models[m])['step2'].days
    step3_delta = (models[m])['step3'].days
    step4_delta = (models[m])['step4'].days
    int1_delta = (models[m])['int1_delta'].days
    int2_delta = (models[m])['int2_delta'].days
    int3_delta = (models[m])['int3_delta'].days
    #Rates Tuples
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
        st.write(m)
        
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
        st.write(m)
        
###########################################################################
    ## SEIR model with phase adjusted R_0
    if (models[m])['base_model'] == 'SEIRR_0':
        # Vairables for SEIR Models
        alpha = 1/incubation_period
        beta = ((alpha+intrinsic_growth_rate)*(intrinsic_growth_rate + (1/infectious_period))) / (alpha*S)         
        gamma=1/infectious_period
        recovered=0.0 # Some issues here with recovered pulling from prior model as table. 
        exposed=beta*S*total_infections
        S2=S-exposed-total_infections
        ## SEIR model with phase adjusted R_0
        s, e, i, r = sim_seir_decay(S-2, 1 ,1, 0.0, beta, gamma,alpha, n_days, decay1, decay2, decay3, decay4, step1_delta)
        susceptible, exposed, infected, recovered = s, e, i, r
        i_hospitalized, i_icu, i_ventilated = get_dispositions(i, rates, 1)
        r_hospitalized, r_icu, r_ventilated = get_dispositions(r, rates, 1)
        dispositions_R = (i_hospitalized + r_hospitalized, i_icu + r_icu, i_ventilated + r_ventilated)
        hospitalized_R, icu_R, ventilated_R = (i_hospitalized, i_icu, i_ventilated)
        ## Build Admissions Datasets
        globals()['projection_admits'+str(m)] = build_admissions_df(dispositions)
        ## Build Census Table
        globals()['census_table'+str(m)] = build_census_df(globals()['projection_admits'+str(m)])
        st.write(m)
        
###########################################################################
    # if (models[m])['base_model'] == 'SEIRD':
        # # Vairables for SEIR Models
        # alpha = 1/incubation_period
        # beta = ((alpha+intrinsic_growth_rate)*(intrinsic_growth_rate + (1/infectious_period))) / (alpha*S)         
        # gamma=1/infectious_period
        # recovered=0.0 # Some issues here with recovered pulling from prior model as table. 
        # exposed=beta*S*total_infections
        # S2=S-exposed-total_infections
        # ## SEIR Model
        # s, e, i, r, d = sim_seird_decay(S-2, 1, 1 , 0.0, 0.0, beta, gamma, alpha, n_days, decay1, decay2, decay3, decay4, step1_delta, fatal)
        # susceptible, exposed, infected, recovered = s, e, i, r
        # i_hospitalized, i_icu, i_ventilated = get_dispositions(i, rates, 1)
        # r_hospitalized, r_icu, r_ventilated = get_dispositions(r, rates, 1)
        # dispositions = (i_hospitalized + r_hospitalized, i_icu + r_icu, i_ventilated + r_ventilated)
        # hospitalized, icu, ventilated = (i_hospitalized, i_icu, i_ventilated)
        # ## Build Admissions Datasets
        # globals()['projection_admits'+str(m)] = build_admissions_df(dispositions)
        # ## Build Census Table
        # globals()['census_table'+str(m)] = build_census_df(globals()['projection_admits'+str(m)])
        
    
st.dataframe(projection_admits0)
st.dataframe(projection_admits1)
st.dataframe(projection_admits2)
st.dataframe(projection_admits3)
st.dataframe(projection_admits4)  
st.dataframe(projection_admits5)    
        
    
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
        
admits_graph2 = regional_admissions_chart(projection_admits2, 
        plot_projection_days, 
        as_date=as_date)

admits_graph3 = regional_admissions_chart(projection_admits3, 
        plot_projection_days, 
        as_date=as_date)
        
admits_graph4 = regional_admissions_chart(projection_admits4, 
        plot_projection_days, 
        as_date=as_date)
        
admits_graph5 = regional_admissions_chart(projection_admits5, 
        plot_projection_days, 
        as_date=as_date)
                

# for i in admit_tables:
    # globals()['admit_graph'+str(m)] = regional_admissions_chart([i], plot_projection_days, as_date=as_date)

st.altair_chart(
    admits_graph0
    + admits_graph1
    + admits_graph2
    + admits_graph3
    + admits_graph4
    + admits_graph5
    , use_container_width=True)



