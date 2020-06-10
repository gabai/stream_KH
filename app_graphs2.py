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

hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
    """
st.markdown(hide_menu_style, unsafe_allow_html=True)


###########################
# Models and base functions
###########################
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

def build_admissions_df_n(
    dispositions) -> pd.DataFrame:
    """Build admissions dataframe from Parameters."""
    days = np.array(range(0, n_days))
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

def build_prev_df_n(
    dispositions) -> pd.DataFrame:
    """Build admissions dataframe from Parameters."""
    days = np.array(range(0, n_days))
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
    decay1:float, decay2:float, decay3: float, decay4: float, reopen1_delta: int
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
        elif int2_delta<=day<=reopen1_delta:
            beta_decay=beta*(1-decay3)
        else:
            beta_decay=beta*(1-decay4)
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
        decay1:float, decay2:float, decay3: float, decay4: float, reopen1_delta: int, fatal: float
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
            elif int2_delta<=day<=reopen1_delta:
                beta_decay=beta*(1-decay3)
            else:
                beta_decay=beta*(1-decay4)
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
    elif int2_delta<=t<int3_delta:
        beta_decay=beta*(1-decay3)    
    elif int3_delta<=t<=reopen1_delta:
        beta_decay=beta*(1-decay4)
    elif reopen1_delta<=t<=reopen2_delta:
        beta_decay=beta*(1-decay5)
    else:
        beta_decay=beta*(1-decay6)    
    return beta_decay

#The SIR model differential equations with ODE solver.
def derivdecay(y, t, N, beta, gamma1, gamma2, alpha, p, hosp,q,l,n_days, decay1, decay2, decay3, decay4, decay5, start_day, int1_delta, int2_delta, reopen1_delta, reopen2_delta, fatal_hosp ):
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
    s, e,a,i, j,r, d, beta, gamma1, gamma2, alpha, n_days,decay1,decay2,decay3, decay4, decay5, start_day, int1_delta, int2_delta,reopen1_delta, reopen2_delta, fatal_hosp, p, hosp, q,
    l):
    n = s + e + a + i + j+ r + d
    rh=0
    y0= s,e,a,i,j,r,d, rh
    
    t=np.arange(0, n_days, step=1)
    ret = odeint(derivdecay, y0, t, args=(n, beta, gamma1, gamma2, alpha, p, hosp,q,l, n_days, decay1, decay2, decay3, decay4, decay5, start_day, int1_delta, int2_delta, reopen1_delta, reopen2_delta, fatal_hosp))
    S_n, E_n,A_n, I_n,J_n, R_n, D_n ,RH_n= ret.T
    
    return (S_n, E_n,A_n, I_n,J_n, R_n, D_n, RH_n)

####The SIR model differential equations with ODE solver. Presymptomatic and masks
def betanew2(t,beta,x,p_m1, pm_2 ):
    if start_day<= t <= int1_delta:
        beta_decay=beta*(1-decay1)
    elif int1_delta<=t<int2_delta:
        beta_decay=beta*(1-decay2)
    elif int2_delta<=t<int3_delta:
        beta_decay=beta*(1-decay3)
    elif int2_delta<=t<=reopen1_delta:
        beta_decay=beta*(1-decay4)*(1-(x*p_m1))**2
    elif reopen1_delta<=t<=reopen2_delta:
        beta_decay=beta*(1-decay5)*(1-(x*p_m2))**2 
    else:
        beta_decay=beta*(1-decay6)*(1-(x*p_m2))**2    
    return beta_decay

def derivdecayP(y, t, beta, gamma1, gamma2, alpha, sym, hosp,q,l,n_days, decay1,decay2, decay3, decay4,decay5,start_day, int1_delta, int2_delta, reopen1_delta,
                reopen2_delta,fatal_hosp, x, p_m1, p_m2, delta_p ):
    S, E, P,A, I,J, R,D,counter = y
    N=S+E+P+A+I+J+R+D
    dSdt = - betanew2(t, beta, x, p_m1, p_m2) * S * (q*I + l*J +P+ A)/N 
    dEdt = betanew2(t, beta, x, p_m1, p_m2) * S * (q*I + l*J +P+ A)/N   - alpha * E
    dPdt = alpha * E - delta_p * P
    dAdt = delta_p* P *(1-sym)-gamma1*A
    dIdt = sym* delta_p* P - gamma1 * I- hosp*I
    dJdt = hosp * I -gamma2*J
    dRdt = (1-fatal_hosp)*gamma2 * J + gamma1*(A+I)
    dDdt = fatal_hosp * gamma2 * J
    counter = (1-fatal_hosp)*gamma2 * J
    return dSdt, dEdt,dPdt,dAdt, dIdt, dJdt, dRdt, dDdt, counter

def sim_sepaijrd_decay_ode(
    s, e,p,a,i, j,r, d, beta, gamma1, gamma2, alpha, n_days,decay1,decay2,decay3, decay4, decay5, start_day, int1_delta,
    int2_delta,reopen1_delta, reopen2_delta, fatal_hosp, sym, hosp, q,
    l,x, p_m1, p_m2, delta_p):
    n = s + e + p+a + i + j+ r + d
    rh=0
    y0= s,e,p,a,i,j,r,d, rh
    
    t=np.arange(0, n_days, step=1)
    ret = odeint(derivdecayP, y0, t, args=(beta, gamma1, gamma2, alpha, sym, hosp,q,l, n_days, decay1, decay2, decay3, decay4, decay5, start_day, int1_delta,
                                           int2_delta, reopen1_delta, reopen2_delta, fatal_hosp, x, p_m1, p_m2, delta_p))
    S_n, E_n,P_n,A_n, I_n,J_n, R_n, D_n ,RH_n= ret.T
    
    return (S_n, E_n,P_n,A_n, I_n,J_n, R_n, D_n, RH_n)

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

############################
# Erie County Data
############################
url = 'https://raw.githubusercontent.com/gabai/stream_KH/master/Cases_Erie.csv'
erie_df = pd.read_csv(url)
erie_df['Date'] = pd.to_datetime(erie_df['Date'])


###################
# General Variables
###################
start_date = date(2020,3,1)  # Suspected first contact
start_day = 1
intervention1 = date(2020,3,22) # Date of NYS School Closure
int1_delta = (intervention1 - start_date).days
intervention2 = date(2020,3,28) # Date of NYS Closure of non-essential businesses
int2_delta = (intervention2 - start_date).days
intervention3 = date(2020,4,15) # Date of NYS Face-Mask Mandate
int3_delta = (intervention3 - start_date).days
reopen1 = date(2020,5,19) # Date of reopening phase 1
reopen1_delta = (reopen1 - start_date).days
reopen2 = date(2020,6,9) # Date of reopening phase 2
reopen2_delta = (reopen2 - start_date).days

incubation_period = 3.1
recovery_days = 11
infectious_period = 3
incubation_period = 3.1
regional_hosp_share = 1.0
fatal = 0.05 # Needs review
fatal_hosp = 0.09
hosp_lag = 4
delta_p = 1.7 # "Days a person is pre-symptomatic"

known_infections = erie_df['Cases'].iloc[-1]
known_cases = erie_df['Admissions'].iloc[-1]
current_hosp = known_infections

gamma = 1 / recovery_days
alpha = 1/ incubation_period
recovered = 0.0

# Slider and Date
n_days = st.slider("Number of days to project", 30, 200, 120, 1, "%i")
as_date = st.checkbox(label="Present result as dates", value=False)

# Projection days
plot_projection_days = n_days - 10



############################
# Erie County Graph
############################

# Graph Function
if as_date:
    #erie_df = add_date_column(erie_df)
    day_date = 'Date:T'
    def erie_inpatient(projection_admits: pd.DataFrame) -> alt.Chart:
        """docstring"""
    
        projection_admits = projection_admits.rename(columns={"Admissions": "Erie County Inpatient Census"})
    
        return(
            alt
            .Chart(projection_admits)
            .transform_fold(fold=["Erie County Inpatient Census"])
            .mark_line(strokeWidth=3, strokeDash=[2,3], point=True)
            .encode(
                x=alt.X(day_date),
                y=alt.Y("value:Q", title="Census"),
                color="key:N",
                tooltip=[alt.Tooltip("value:Q", format=".0f"),"key:N"]
            )
            .interactive()
        )
else:
    day_date = 'day'
    def erie_inpatient(
        projection_admits: pd.DataFrame) -> alt.Chart:
        """docstring"""
    
        projection_admits = projection_admits.rename(columns={"Admissions": "Erie County Inpatient"})
    
        return(
            alt
            .Chart(projection_admits)
            .transform_fold(fold=["Erie County Inpatient"])
            .mark_line(strokeWidth=3, strokeDash=[2,3], point=True)
            .encode(
                x=alt.X(day_date),
                y=alt.Y("value:Q", title="Census"),
                color="key:N",
                tooltip=[alt.Tooltip("value:Q", format=".0f"),"key:N"]
            )
            .interactive()
        )

# Graph
erie_lines_ip = erie_inpatient(erie_df)



###################### 
# Vertical Lines Graph
######################

# Schools 3/18/20; Non-essential business 3/22/20; Face-Mask Mandate 4/15/20 
vertical = pd.DataFrame({'day': [int1_delta, int2_delta, int3_delta, reopen1_delta, reopen2_delta]})

# Graph Function
def vertical_chart(
    projection_admits: pd.DataFrame, 
    as_date:bool = False) -> alt.Chart:
    """docstring"""
    
    tooltip_dict = {False: "day", True: "date:T"}
    if as_date:
        projection_admits = add_date_column(projection_admits)
        x_kwargs = {"shorthand": "date:T", "title": "Date"}
    else:
        x_kwargs = {"shorthand": "day", "title": "Days from initial infection"}
    
    return (
        alt
        .Chart(projection_admits)
        .mark_rule(color='gray')
        .encode(
            x=alt.X(**x_kwargs),
            tooltip=[
                tooltip_dict[as_date]],
        )
    )

# Graph
vertical = vertical_chart(vertical, as_date=as_date)



#############
# SIR model
# 1
#############
S_default = 1500000
S = S_default
doubling_time=3
relative_contact_rate= 0
intrinsic_growth_rate = 2 ** (1 / doubling_time) - 1
beta = (intrinsic_growth_rate + gamma) / S * (1-relative_contact_rate)

hosp_rate = 0.1
icu_rate = 0.04
vent_rate = 0.02
hosp_los = 10
icu_los = 9
vent_los = 7
incubation_period = 5.2
total_infections = current_hosp / regional_hosp_share / hosp_rate

gamma_hosp = 1 / hosp_los
icu_days = 1 / icu_los
beta_decay = 0.0
RateLos = namedtuple("RateLos", ("rate", "length_of_stay"))
hospitalized=RateLos(hosp_rate, hosp_los)
icu=RateLos(icu_rate, icu_los)
ventilated=RateLos(vent_rate, vent_los)

rates = tuple(each.rate for each in (hospitalized, icu, ventilated))
lengths_of_stay = tuple(each.length_of_stay for each in (hospitalized, icu, ventilated))

s_v, i_v, r_v = sim_sir(S-2, 1, 1, beta, gamma, n_days)
susceptible_v, infected_v, recovered_v = s_v, i_v, r_v

i_hospitalized_v, i_icu_v, i_ventilated_v = get_dispositions(i_v, rates, regional_hosp_share)

r_hospitalized_v, r_icu_v, r_ventilated_v = get_dispositions(r_v, rates, regional_hosp_share)

dispositions_SIR = (
            i_hospitalized_v + r_hospitalized_v,
            i_icu_v + r_icu_v,
            i_ventilated_v + r_ventilated_v)

hospitalized_SIR, icu_SIR, ventilated_SIR = (
            i_hospitalized_v,
            i_icu_v,
            i_ventilated_v)

# New Cases Table
projection_admits_SIR = build_admissions_df(dispositions_SIR)
# Census Table
census_table_SIR = build_census_df(projection_admits_SIR)

# Graph Function
def sir_graph(
    census: pd.DataFrame,
    plot_projection_days: int,
    as_date:bool = False) -> alt.Chart:
    """docstring"""
    census = census.rename(columns={"hosp": "SIR Model"})

    tooltip_dict = {False: "day", True: "date:T"}
    if as_date:
        census = add_date_column(census.head(plot_projection_days))
        x_kwargs = {"shorthand": "date:T", "title": "Date"}
    else:
        x_kwargs = {"shorthand": "day", "title": "Days from initial infection"}

    return (
        alt
        .Chart(census)
        .transform_fold(fold=["SIR Model"])
        .mark_line(point=True)
        .encode(
            x=alt.X(**x_kwargs),
            y=alt.Y("value:Q", title="Census"),
            color=alt.value('green'),
            #color="key:N",
            tooltip=[
                tooltip_dict[as_date],
                alt.Tooltip("value:Q", format=".0f", title="Census"),
                "key:N",
            ],
        )
        .interactive()
    )

# Graph: SIR Model    
sir = sir_graph(census_table_SIR, plot_projection_days, as_date=as_date)
#################################################################################################



#######################
# SEIR model v1 (0% SD)
# 2
#######################
S_default = 1500000
S = S_default
doubling_time=3
relative_contact_rate=0
intrinsic_growth_rate = 2 ** (1 / doubling_time) - 1
beta = (intrinsic_growth_rate + gamma) / S * (1-relative_contact_rate)

hosp_rate = 0.1
icu_rate = 0.04
vent_rate = 0.02
hosp_los = 10
icu_los = 9
vent_los = 7
incubation_period = 5.2
total_infections = current_hosp / regional_hosp_share / hosp_rate

beta2 = (intrinsic_growth_rate + (1/infectious_period)) / S * (1-relative_contact_rate)
beta3 = ((alpha+intrinsic_growth_rate)*(intrinsic_growth_rate + (1/infectious_period))) / (alpha*S) *(1-relative_contact_rate)
beta4 = ((alpha+intrinsic_growth_rate)*(intrinsic_growth_rate + (1/infectious_period))) / (alpha*S) 
gamma2=1/infectious_period
exposed2=beta4*S*total_infections
S2=S-exposed2-total_infections

# Beta's need to be reviewed: we have beta3 here, just to confirm that this is the correct one. 

s_e, e_e, i_e, r_e = sim_seir(S-2, 1 ,1, recovered, beta3, gamma2, alpha, n_days)

susceptible_e, exposed_e, infected_e, recovered_e = s_e, e_e, i_e, r_e

i_hospitalized_e, i_icu_e, i_ventilated_e = get_dispositions(i_e, rates, regional_hosp_share)

r_hospitalized_e, r_icu_e, r_ventilated_e = get_dispositions(r_e, rates, regional_hosp_share)

dispositions_SEIR1 = (
            i_hospitalized_e + r_hospitalized_e,
            i_icu_e + r_icu_e,
            i_ventilated_e + r_ventilated_e)

hospitalized_SEIR1, icu_SEIR1, ventilated_SEIR1 = (
            i_hospitalized_e,
            i_icu_e,
            i_ventilated_e)

# New cases Table
projection_admits_SEIR1 = build_admissions_df(dispositions_SEIR1)
# Census Table
census_table_SEIR1 = build_census_df(projection_admits_SEIR1)

# Graph Function
def seir_graph1(
    census: pd.DataFrame,
    plot_projection_days: int,
    as_date:bool = False) -> alt.Chart:
    """docstring"""
    census = census.rename(columns={"hosp": "SEIR Model v1"})

    tooltip_dict = {False: "day", True: "date:T"}
    if as_date:
        census = add_date_column(census.head(plot_projection_days))
        x_kwargs = {"shorthand": "date:T", "title": "Date"}
    else:
        x_kwargs = {"shorthand": "day", "title": "Days from initial infection"}

    return (
        alt
        .Chart(census)
        .transform_fold(fold=['SEIR Model v1'])
        .mark_line(point=False)
        .encode(
            x=alt.X(**x_kwargs),
            y=alt.Y("value:Q", title="Census"),
            color=alt.value('blue'),
            #color="key:N",
            tooltip=[
                tooltip_dict[as_date],
                alt.Tooltip("value:Q", format=".0f", title="Census"),
                "key:N",
            ],
        )
        .interactive()
    )

# Graph: SEIR 0% SD     
seir_0SD = seir_graph1(census_table_SEIR1, plot_projection_days, as_date=as_date)

#################################################################################################



##########################
# SEIR model v2 (30% SD)
# 3
##########################
S_default = 1500000
S = S_default
doubling_time=3
relative_contact_rate=0.3
intrinsic_growth_rate = 2 ** (1 / doubling_time) - 1
beta = (intrinsic_growth_rate + gamma) / S * (1-relative_contact_rate)

hosp_rate = 0.14
icu_rate = 0.04
vent_rate = 0.02
hosp_los = 10
icu_los = 9
vent_los = 7
incubation_period = 5.2
total_infections = current_hosp / regional_hosp_share / hosp_rate

beta2 = (intrinsic_growth_rate + (1/infectious_period)) / S * (1-relative_contact_rate)
beta3 = ((alpha+intrinsic_growth_rate)*(intrinsic_growth_rate + (1/infectious_period))) / (alpha*S) *(1-relative_contact_rate)
beta4 = ((alpha+intrinsic_growth_rate)*(intrinsic_growth_rate + (1/infectious_period))) / (alpha*S) 
gamma2=1/infectious_period
exposed2=beta4*S*total_infections
S2=S-exposed2-total_infections

s_e, e_e, i_e, r_e = sim_seir(S-2, 1 ,1, recovered, beta3, gamma2, alpha, n_days)

susceptible_e, exposed_e, infected_e, recovered_e = s_e, e_e, i_e, r_e

i_hospitalized_e, i_icu_e, i_ventilated_e = get_dispositions(i_e, rates, regional_hosp_share)

r_hospitalized_e, r_icu_e, r_ventilated_e = get_dispositions(r_e, rates, regional_hosp_share)

dispositions_SEIR2 = (
            i_hospitalized_e + r_hospitalized_e,
            i_icu_e + r_icu_e,
            i_ventilated_e + r_ventilated_e)

hospitalized_SEIR2, icu_SEIR2, ventilated_SEIR2 = (
            i_hospitalized_e,
            i_icu_e,
            i_ventilated_e)

# New Cases Table
projection_admits_SEIR2 = build_admissions_df(dispositions_SEIR2)
# Census Table
census_table_SEIR2 = build_census_df(projection_admits_SEIR2)

def seir_graph2(
    census: pd.DataFrame,
    plot_projection_days: int,
    as_date:bool = False) -> alt.Chart:
    """docstring"""
    census = census.rename(columns={"hosp": "SEIR Model v2"})

    tooltip_dict = {False: "day", True: "date:T"}
    if as_date:
        census = add_date_column(census.head(plot_projection_days))
        x_kwargs = {"shorthand": "date:T", "title": "Date"}
    else:
        x_kwargs = {"shorthand": "day", "title": "Days from initial infection"}

    return (
        alt
        .Chart(census)
        .transform_fold(fold=["SEIR Model v2"])
        .mark_line(point=False)
        .encode(
            x=alt.X(**x_kwargs),
            y=alt.Y("value:Q", title="Census"),
            color=alt.value('red'),
            #color="key:N",
            tooltip=[
                tooltip_dict[as_date],
                alt.Tooltip("value:Q", format=".0f", title="Census"),
                "key:N",
            ],
        )
        .interactive()
    )
    
seir_30SD = seir_graph2(census_table_SEIR2, plot_projection_days, as_date=as_date)
#################################################################################################



#################################################
# SEIR model with phase adjusted R_0 (version 1)
# 4
#################################################
S_default = 1500000
S = S_default
doubling_time=4
relative_contact_rate=0
intrinsic_growth_rate = 2 ** (1 / doubling_time) - 1
beta = (intrinsic_growth_rate + gamma) / S * (1-relative_contact_rate)

decay1 = 0 # From 0-2 weeks
decay2 = 0.1 # From 2-3 weeks
decay3 = 0.3 # From 3 to face-mask mandate
decay4 = 0.3 # From face-mask forward, didn't have phase 1 reopen here

hosp_rate = 0.14
icu_rate = 0.04
vent_rate = 0.02
hosp_los = 10
icu_los = 9
vent_los = 7
incubation_period = 5.8

beta2 = (intrinsic_growth_rate + (1/infectious_period)) / S * (1-relative_contact_rate)
beta3 = ((alpha+intrinsic_growth_rate)*(intrinsic_growth_rate + (1/infectious_period))) / (alpha*S) *(1-relative_contact_rate)
beta4 = ((alpha+intrinsic_growth_rate)*(intrinsic_growth_rate + (1/infectious_period))) / (alpha*S) 
gamma2=1/infectious_period
exposed2=beta4*S*total_infections
S2=S-exposed2-total_infections

# Model
s_R, e_R, i_R, r_R = sim_seir_decay(S-2, 1 ,1, 0.0, beta4, gamma2, alpha, n_days, decay1, decay2, decay3, decay4, reopen1_delta)

susceptible_R, exposed_R, infected_R, recovered_R = s_R, e_R, i_R, r_R

i_hospitalized_R, i_icu_R, i_ventilated_R = get_dispositions(i_R, rates, regional_hosp_share)

r_hospitalized_R, r_icu_R, r_ventilated_R = get_dispositions(r_R, rates, regional_hosp_share)

dispositions_R1 = (
            i_hospitalized_R + r_hospitalized_R,
            i_icu_R + r_icu_R,
            i_ventilated_R + r_ventilated_R)

hospitalized_R1, icu_R1, ventilated_R1 = (
            i_hospitalized_R,
            i_icu_R,
            i_ventilated_R)

# New Cases Table
projection_admits_R1 = build_admissions_df(dispositions_R1)
# Census Table
census_table_R1 = build_census_df(projection_admits_R1)

# Graph Function
def seiraR0_graph1(
    census: pd.DataFrame,
    plot_projection_days: int,
    as_date:bool = False) -> alt.Chart:
    """docstring"""
    census = census.rename(columns={"hosp": "SEIRaR0 Model v1"})

    tooltip_dict = {False: "day", True: "date:T"}
    if as_date:
        census = add_date_column(census.head(plot_projection_days))
        x_kwargs = {"shorthand": "date:T", "title": "Date"}
    else:
        x_kwargs = {"shorthand": "day", "title": "Days from initial infection"}

    return (
        alt
        .Chart(census)
        .transform_fold(fold=["SEIRaR0 Model v1"])
        .mark_line(point=False)
        .encode(
            x=alt.X(**x_kwargs),
            y=alt.Y("value:Q", title="Census"),
            color=alt.value('yellow'),
            #color="key:N",
            tooltip=[
                tooltip_dict[as_date],
                alt.Tooltip("value:Q", format=".0f", title="Census"),
                "key:N",
            ],
        )
        .interactive()
    )

# Graph
seir_R0_g1 = seiraR0_graph1(census_table_R1, plot_projection_days, as_date=as_date)

#################################################################################################



#################################################
# SEIR model with phase adjusted R_0 (version 2)
# 5
#################################################
S_default = 1500000
S = S_default
doubling_time=3
relative_contact_rate=0
intrinsic_growth_rate = 2 ** (1 / doubling_time) - 1
beta = (intrinsic_growth_rate + gamma) / S * (1-relative_contact_rate)

decay1 = 0 # From 0-2 weeks
decay2 = 0.1 # From 2-3 weeks
decay3 = 0.3 # From 3 to face-mask mandate
decay4 = 0.3 # From face-mask forward, didn't have phase 1 reopen here

hosp_rate = 0.14
icu_rate = 0.0125
vent_rate = 0.01
hosp_los = 10
icu_los = 9
vent_los = 7
incubation_period = 5.8

beta2 = (intrinsic_growth_rate + (1/infectious_period)) / S * (1-relative_contact_rate)
beta3 = ((alpha+intrinsic_growth_rate)*(intrinsic_growth_rate + (1/infectious_period))) / (alpha*S) *(1-relative_contact_rate)
beta4 = ((alpha+intrinsic_growth_rate)*(intrinsic_growth_rate + (1/infectious_period))) / (alpha*S) 
gamma2= 1/infectious_period
exposed2= beta4*S*total_infections
S2= S-exposed2-total_infections

# Model
s_R, e_R, i_R, r_R = sim_seir_decay(S-2, 1 ,1, 0.0, beta4, gamma2,alpha, n_days, decay1, decay2, decay3, decay4, reopen1_delta)

susceptible_R, exposed_R, infected_R, recovered_R = s_R, e_R, i_R, r_R

i_hospitalized_R, i_icu_R, i_ventilated_R = get_dispositions(i_R, rates, regional_hosp_share)

r_hospitalized_R, r_icu_R, r_ventilated_R = get_dispositions(r_R, rates, regional_hosp_share)

dispositions_R2 = (
            i_hospitalized_R + r_hospitalized_R,
            i_icu_R + r_icu_R,
            i_ventilated_R + r_ventilated_R)

hospitalized_R2, icu_R2, ventilated_R2 = (
            i_hospitalized_R,
            i_icu_R,
            i_ventilated_R)

# New Cases Table
projection_admits_R2 = build_admissions_df(dispositions_R2)
# Census Table
census_table_R2 = build_census_df(projection_admits_R2)

# Graph Function
def seiraR0_graph2(
    census: pd.DataFrame,
    plot_projection_days: int,
    as_date:bool = False) -> alt.Chart:
    """docstring"""
    census = census.rename(columns={"hosp": "SEIRaR0 Model v2"})

    tooltip_dict = {False: "day", True: "date:T"}
    if as_date:
        census = add_date_column(census.head(plot_projection_days))
        x_kwargs = {"shorthand": "date:T", "title": "Date"}
    else:
        x_kwargs = {"shorthand": "day", "title": "Days from initial infection"}

    return (
        alt
        .Chart(census)
        .transform_fold(fold=["SEIRaR0 Model v2"])
        .mark_line(point=False)
        .encode(
            x=alt.X(**x_kwargs),
            y=alt.Y("value:Q", title="Census"),
            color=alt.value('aqua'),
            #color="key:N",
            tooltip=[
                tooltip_dict[as_date],
                alt.Tooltip("value:Q", format=".0f", title="Census"),
                "key:N",
            ],
        )
        .interactive()
    )

# Graph    
seir_R0_g2 = seiraR0_graph2(census_table_R2, plot_projection_days, as_date=as_date)

#################################################################################################



################################################
# SEIR model with phase adjusted R_0 (version 3)
# 6
################################################
S_default = 1500000
S = S_default
doubling_time=3
relative_contact_rate=0
intrinsic_growth_rate = 2 ** (1 / doubling_time) - 1
beta = (intrinsic_growth_rate + gamma) / S * (1-relative_contact_rate)

decay1 = 0 # From 0-2 weeks
decay2 = 0.1 # From 2-3 weeks
decay3 = 0.3 # From 3 to face-mask mandate
decay4 = 0.3 # From face-mask forward, didn't have phase 1 reopen here

hosp_rate = 0.05
icu_rate = 0.0125
vent_rate = 0.01
hosp_los = 5
icu_los = 8
vent_los = 8
incubation_period = 5.8

beta2 = (intrinsic_growth_rate + (1/infectious_period)) / S * (1-relative_contact_rate)
beta3 = ((alpha+intrinsic_growth_rate)*(intrinsic_growth_rate + (1/infectious_period))) / (alpha*S) *(1-relative_contact_rate)
beta4 = ((alpha+intrinsic_growth_rate)*(intrinsic_growth_rate + (1/infectious_period))) / (alpha*S) 
gamma2= 1/infectious_period
exposed2= beta4*S*total_infections
S2= S-exposed2-total_infections

# Model
s_R, e_R, i_R, r_R = sim_seir_decay(S-2, 1 ,1, 0.0, beta4, gamma2,alpha, n_days, decay1, decay2, decay3, decay4, reopen1_delta)

susceptible_R, exposed_R, infected_R, recovered_R = s_R, e_R, i_R, r_R

i_hospitalized_R, i_icu_R, i_ventilated_R = get_dispositions(i_R, rates, regional_hosp_share)

r_hospitalized_R, r_icu_R, r_ventilated_R = get_dispositions(r_R, rates, regional_hosp_share)

dispositions_R3 = (
            i_hospitalized_R + r_hospitalized_R,
            i_icu_R + r_icu_R,
            i_ventilated_R + r_ventilated_R)

hospitalized_R3, icu_R3, ventilated_R3 = (
            i_hospitalized_R,
            i_icu_R,
            i_ventilated_R)
            

# New Cases Table
projection_admits_R3 = build_admissions_df(dispositions_R3)
# Census Table
census_table_R3 = build_census_df(projection_admits_R3)

# Graph Function
def seiraR0_graph3(
    census: pd.DataFrame,
    plot_projection_days: int,
    as_date:bool = False) -> alt.Chart:
    """docstring"""
    census = census.rename(columns={"hosp": "SEIRaR0 Model v3"})

    tooltip_dict = {False: "day", True: "date:T"}
    if as_date:
        census = add_date_column(census.head(plot_projection_days))
        x_kwargs = {"shorthand": "date:T", "title": "Date"}
    else:
        x_kwargs = {"shorthand": "day", "title": "Days from initial infection"}

    return (
        alt
        .Chart(census)
        .transform_fold(fold=["SEIRaR0 Model v3"])
        .mark_line(point=False)
        .encode(
            x=alt.X(**x_kwargs),
            y=alt.Y("value:Q", title="Census"),
            color=alt.value('teal'),
            #color="key:N",
            tooltip=[
                tooltip_dict[as_date],
                alt.Tooltip("value:Q", format=".0f", title="Census"),
                "key:N",
            ],
        )
        .interactive()
    )

# Graph
seir_R0_g3 = seiraR0_graph3(census_table_R3, plot_projection_days, as_date=as_date)

#################################################################################################



##################################################################
# SEIR model with phase adjusted R_0 and Disease Related Fatality
# High Social Distancing
# 7
##################################################################

S_default = 1500000
S = S_default
current_hosp=1
doubling_time=2
relative_contact_rate=0
intrinsic_growth_rate = 2 ** (1 / doubling_time) - 1
beta = (intrinsic_growth_rate + gamma) / S * (1-relative_contact_rate)

decay1 = 2
decay2 = 0.52
decay3 = 0.83
decay4 = 0.75

hosp_rate = 0.4
icu_rate = 0.0125
vent_rate = 0.02
hosp_los = 5
icu_los = 8
vent_los = 8
incubation_period = 5.8

beta2 = (intrinsic_growth_rate + (1/infectious_period)) / S * (1-relative_contact_rate)
beta3 = ((alpha+intrinsic_growth_rate)*(intrinsic_growth_rate + (1/infectious_period))) / (alpha*S) *(1-relative_contact_rate)
beta4 = ((alpha+intrinsic_growth_rate)*(intrinsic_growth_rate + (1/infectious_period))) / (alpha*S) 
gamma2=1/infectious_period
exposed2=beta4*S*total_infections
S2=S-exposed2-total_infections

# Model
s_D, e_D, i_D, r_D, d_D = sim_seird_decay(S-2, 1, 1 , 0.0, 0.0, beta4, gamma2,alpha, n_days, decay1, decay2, decay3, decay4, reopen1_delta, fatal)

susceptible_D, exposed_D, infected_D, recovered_D = s_D, e_D, i_D, r_D

i_hospitalized_D, i_icu_D, i_ventilated_D = get_dispositions(i_D, rates, regional_hosp_share)

r_hospitalized_D, r_icu_D, r_ventilated_D = get_dispositions(r_D, rates, regional_hosp_share)

dispositions_D_highsocial = (
            i_hospitalized_D + r_hospitalized_D,
            i_icu_D + r_icu_D,
            i_ventilated_D + r_ventilated_D)

hospitalized_D_highsocial, icu_D_highsocial, ventilated_D_highsocial = (
            i_hospitalized_D,
            i_icu_D,
            i_ventilated_D)

# New Cases Table
projection_admits_D_highsocial = build_admissions_df(dispositions_D_highsocial)
# Census Table
census_table_D_highsocial = build_census_df(projection_admits_D_highsocial)

# Graph Function
def seird_hsd_graph(
    census: pd.DataFrame,
    plot_projection_days: int,
    as_date:bool = False) -> alt.Chart:
    """docstring"""
    census = census.rename(columns={"hosp": "SEIRhSD Model"})

    tooltip_dict = {False: "day", True: "date:T"}
    if as_date:
        census = add_date_column(census.head(plot_projection_days))
        x_kwargs = {"shorthand": "date:T", "title": "Date"}
    else:
        x_kwargs = {"shorthand": "day", "title": "Days from initial infection"}

    return (
        alt
        .Chart(census)
        .transform_fold(fold=["SEIRhSD Model"])
        .mark_line(point=False)
        .encode(
            x=alt.X(**x_kwargs),
            y=alt.Y("value:Q", title="Census"),
            color=alt.value('pink'),
            #color="key:N",
            tooltip=[
                tooltip_dict[as_date],
                alt.Tooltip("value:Q", format=".0f", title="Census"),
                "key:N",
            ],
        )
        .interactive()
    )

# Graph
seir_highsocial = seird_hsd_graph(census_table_D_highsocial, plot_projection_days, as_date=as_date)

#################################################################################################



##############################################################################
# SEIR model with phase adjusted R_0 and Disease Related Fatality (Version 1)
# 8
##############################################################################

S_default = 1500000
S = S_default
current_hosp = 1
doubling_time = 3
relative_contact_rate = 0
intrinsic_growth_rate = 2 ** (1 / doubling_time) - 1
beta = (intrinsic_growth_rate + gamma) / S * (1-relative_contact_rate)

decay1 = 0
decay2 = 0.15
decay3 = 0.3
decay4 = 0.3

hosp_rate = 0.04
icu_rate = 0.0125
vent_rate = 0.02
hosp_los = 5
icu_los = 8
vent_los = 8
incubation_period = 5.8

beta2 = (intrinsic_growth_rate + (1/infectious_period)) / S * (1-relative_contact_rate)
beta3 = ((alpha+intrinsic_growth_rate)*(intrinsic_growth_rate + (1/infectious_period))) / (alpha*S) *(1-relative_contact_rate)
beta4 = ((alpha+intrinsic_growth_rate)*(intrinsic_growth_rate + (1/infectious_period))) / (alpha*S) 
gamma2=1/infectious_period
exposed2=beta4*S*total_infections
S2=S-exposed2-total_infections

# Model
s_D, e_D, i_D, r_D, d_D = sim_seird_decay(S-2, 1, 1 , 0.0, 0.0, beta4, gamma2, alpha, n_days, decay1, decay2, decay3, decay4, reopen1_delta, fatal)

susceptible_D, exposed_D, infected_D, recovered_D = s_D, e_D, i_D, r_D

i_hospitalized_D, i_icu_D, i_ventilated_D = get_dispositions(i_D, rates, regional_hosp_share)

r_hospitalized_D, r_icu_D, r_ventilated_D = get_dispositions(r_D, rates, regional_hosp_share)

dispositions_D1 = (
            i_hospitalized_D + r_hospitalized_D,
            i_icu_D + r_icu_D,
            i_ventilated_D + r_ventilated_D)

hospitalized_D1, icu_D1, ventilated_D1 = (
            i_hospitalized_D,
            i_icu_D,
            i_ventilated_D)

# New Cases Table
projection_admits_D1 = build_admissions_df(dispositions_D1)
# Census Table
census_table_D1 = build_census_df(projection_admits_D1)

# Graph Function
def seird_graph(
    census: pd.DataFrame,
    plot_projection_days: int,
    as_date:bool = False) -> alt.Chart:
    """docstring"""
    census = census.rename(columns={"hosp": "SEIRD Model"})

    tooltip_dict = {False: "day", True: "date:T"}
    if as_date:
        census = add_date_column(census.head(plot_projection_days))
        x_kwargs = {"shorthand": "date:T", "title": "Date"}
    else:
        x_kwargs = {"shorthand": "day", "title": "Days from initial infection"}

    return (
        alt
        .Chart(census)
        .transform_fold(fold=["SEIRD Model"])
        .mark_line(point=False)
        .encode(
            x=alt.X(**x_kwargs),
            y=alt.Y("value:Q", title="Census"),
            color=alt.value('orange'),
            #color="key:N",
            tooltip=[
                tooltip_dict[as_date],
                alt.Tooltip("value:Q", format=".0f", title="Census"),
                "key:N",
            ],
        )
        .interactive()
    )

# Graph    
seir_D = seird_graph(census_table_D1, plot_projection_days, as_date=as_date)

#################################################################################################



#########################################################################################
# SEIR model with phase adjusted R_0, Disease Related Fatality, Asymptomatic Compartment
# 9
#########################################################################################

S_default = 1400000
S = S_default
current_hosp=1
doubling_time=3
relative_contact_rate=0
intrinsic_growth_rate = 2 ** (1 / doubling_time) - 1
beta = (intrinsic_growth_rate + gamma) / S * (1-relative_contact_rate)

decay1 = 0
decay2 = 0.15
decay3 = 0.45
decay4 = 0.35
decay5 = 0.25
decay6 = 0.15

hosp_rate = 0.04
icu_rate = 0.35
vent_rate = 0.35
hosp_los = 5
icu_los = 11
vent_los = 10
incubation_period = 5.8

beta2 = (intrinsic_growth_rate + (1/infectious_period)) / S * (1-relative_contact_rate)
beta3 = ((alpha+intrinsic_growth_rate)*(intrinsic_growth_rate + (1/infectious_period))) / (alpha*S) *(1-relative_contact_rate)
beta4 = ((alpha+intrinsic_growth_rate)*(intrinsic_growth_rate + (1/infectious_period))) / (alpha*S) 
gamma2=1/infectious_period
exposed2=beta4*S*total_infections
S2=S-exposed2-total_infections

asymptomatic = 0.25 # Needs review

## Asymptomatic, Hospitalization
E0=100
A0=100
I0=100
D0=0
R0=0
J0=0

S0=S-E0-A0-I0-D0-J0-R0
beta_j=0.9
q=0.6
l=0.6
gamma_hosp=1/hosp_lag
AAA=beta4*(1/gamma2)*S
beta_j=AAA*(1/(((1-asymptomatic)*1/gamma2)+(asymptomatic*q/(gamma2+hosp_rate))+(asymptomatic*hosp_rate*l/((gamma2+hosp_rate)*gamma_hosp))))

R0_n=beta_j* (((1-asymptomatic)*1/gamma2)+(asymptomatic*q/(gamma2+hosp_rate))+(asymptomatic*hosp_rate*l/((gamma2+hosp_rate)*gamma_hosp)))
beta_j=0.9
R0_n=beta_j* (((1-asymptomatic)*1/gamma2)+(asymptomatic*q/(gamma2+hosp_rate))+(asymptomatic*hosp_rate*l/((gamma2+hosp_rate)*gamma_hosp)))

S_n, E_n,A_n, I_n,J_n, R_n, D_n, RH_n=sim_seaijrd_decay_ode(S0, E0, A0,I0,J0, R0, D0, beta_j,gamma2, gamma_hosp, alpha, n_days,
                                                      decay1,decay2,decay3, decay4, decay5, start_day, int1_delta, int2_delta,
                                                      reopen1_delta, reopen2_delta, fatal_hosp,asymptomatic, hosp_rate, q,  l)


icu_curve= J_n*icu_rate
vent_curve=J_n*vent_rate

hosp_rate_n = 1.0
RateLos = namedtuple("RateLos", ("rate", "length_of_stay"))
hospitalized_n=RateLos(hosp_rate_n, hosp_los)
icu_rate_n= icu_rate
vent_rate_n= vent_rate
icu=RateLos(icu_rate_n, icu_los)
ventilated=RateLos(vent_rate_n, vent_los)


rates_n = tuple(each.rate for each in (hospitalized_n, icu, ventilated))
lengths_of_stay = tuple(each.length_of_stay for each in (hospitalized_n, icu, ventilated))


i_hospitalized_A, i_icu_A, i_ventilated_A = get_dispositions(J_n, rates_n, regional_hosp_share)

r_hospitalized_A, r_icu_A, r_ventilated_A = get_dispositions(RH_n, rates_n, regional_hosp_share)
d_hospitalized_A, d_icu_A, d_ventilated_A = get_dispositions(D_n, rates_n, regional_hosp_share)
dispositions_A = (
            i_hospitalized_A + r_hospitalized_A+ d_hospitalized_A ,
            i_icu_A+r_icu_A+d_icu_A,
            i_ventilated_A+r_ventilated_A +d_ventilated_A)

hospitalized_A, icu_A, ventilated_A = (
            i_hospitalized_A,
            i_icu_A,
            i_ventilated_A)
            

# New Cases Table
projection_admits_A = build_admissions_df_n(dispositions_A)
## Census Table
census_table_A = build_census_df(projection_admits_A)

# Graph Function
def seirdja_graph(
    census: pd.DataFrame,
    plot_projection_days: int,
    as_date:bool = False) -> alt.Chart:
    """docstring"""
    census = census.rename(columns={"hosp": "SEIRDJA Model"})

    tooltip_dict = {False: "day", True: "date:T"}
    if as_date:
        census = add_date_column(census.head(plot_projection_days))
        x_kwargs = {"shorthand": "date:T", "title": "Date"}
    else:
        x_kwargs = {"shorthand": "day", "title": "Days from initial infection"}

    return (
        alt
        .Chart(census)
        .transform_fold(fold=["SEIRDJA Model"])
        .mark_line(point=False)
        .encode(
            x=alt.X(**x_kwargs),
            y=alt.Y("value:Q", title="Census"),
            color=alt.value('orange'),
            #color="key:N",
            tooltip=[
                tooltip_dict[as_date],
                alt.Tooltip("value:Q", format=".0f", title="Census"),
                "key:N",
            ],
        )
        .interactive()
    )
    
seir_A = seirdja_graph(census_table_A, plot_projection_days, as_date=as_date)

#################################################################################################


##################################################################
# SEIR model with phase adjusted R_0 and Disease Related Fatality
# Asymptomatic, Hospitalization, Presymptomatic, and masks (Version 1)
# 10
##################################################################
S_default = 1400000
S = S_default
doubling_time=3
relative_contact_rate=0
intrinsic_growth_rate = 2 ** (1 / doubling_time) - 1
beta = (intrinsic_growth_rate + gamma) / S * (1-relative_contact_rate)

decay1 = 0
decay2 = 0.15
decay3 = 0.45
decay4 = 0.35
decay5 = 0.25
decay6 = 0.25

hosp_rate = 0.03
icu_rate = 0.25
vent_rate = 0.35
hosp_los = 5
icu_los = 11
icu_los = 11
vent_los = 10
incubation_period = 5.8

r_t = beta / gamma * S # r_t is r_0 after distancing
r_naught = (intrinsic_growth_rate + gamma) / gamma
doubling_time_t = 1/np.log2(beta*S - gamma +1) # doubling time after distancing

beta2 = (intrinsic_growth_rate + (1/infectious_period)) / S * (1-relative_contact_rate)
beta3 = ((alpha+intrinsic_growth_rate)*(intrinsic_growth_rate + (1/infectious_period))) / (alpha*S) *(1-relative_contact_rate)
beta4 = ((alpha+intrinsic_growth_rate)*(intrinsic_growth_rate + (1/infectious_period))) / (alpha*S) 
gamma2= 1/infectious_period
exposed2= beta4*S*total_infections
S2= S-exposed2-total_infections
gamma_hosp = 1 / hosp_los
icu_days = 1 / icu_los

asymptomatic = 0.32
q = 0.34
p_m1 = 0.38
p_m2 = 0.5
delta_p = 1.7 

## Asymptomatic, Hospitalization, Presymptomatic, and masks
E0=100
A0=100
I0=100
D0=0
R0=0
J0=0
P0=120
x=0.5
S0=S-E0-P0-A0-I0-D0-J0-R0
beta_j=0.6
q=0.583 # Does this q change compared to the one above? Do we need to change this one?
l=0.717
gamma_hosp=1/hosp_lag
AAA=beta4*(1/gamma2)*S
beta_j=AAA*(1/(((1-asymptomatic)*1/gamma2)+(asymptomatic*q/(gamma2+hosp_rate))+(asymptomatic*hosp_rate*l/((gamma2+hosp_rate)*gamma_hosp))))

R0_n=beta_j* (((1-asymptomatic)*1/gamma2)+(asymptomatic*q/(gamma2+hosp_rate))+(asymptomatic*hosp_rate*l/((gamma2+hosp_rate)*gamma_hosp)))
beta_j=0.51
R0_n=beta_j* (((1-asymptomatic)*1/gamma2)+(asymptomatic*q/(gamma2+hosp_rate))+(asymptomatic*hosp_rate*l/((gamma2+hosp_rate)*gamma_hosp)))

S_p, E_p,P_p,A_p, I_p,J_p, R_p, D_p, RH_p=sim_sepaijrd_decay_ode(S0, E0, P0,A0,I0,J0, R0, D0, beta_j,gamma2, gamma_hosp, alpha, n_days,
                                                      decay1,decay2,decay3, decay4, decay5, start_day, int1_delta, int2_delta,
                                                      reopen1_delta, reopen2_delta, fatal_hosp,asymptomatic, hosp_rate, q,  l,x, p_m1, p_m2, delta_p)

icu_curve= J_p*icu_rate
vent_curve=J_p*vent_rate

hosp_rate_p=1.0
RateLos = namedtuple("RateLos", ("rate", "length_of_stay"))
hospitalized_p=RateLos(hosp_rate_p, hosp_los)
icu_rate_p= icu_rate
vent_rate_p= vent_rate
icu=RateLos(icu_rate_p, icu_los)
ventilated=RateLos(vent_rate_p, vent_los)


rates_p = tuple(each.rate for each in (hospitalized_p, icu, ventilated))
lengths_of_stay = tuple(each.length_of_stay for each in (hospitalized_p, icu, ventilated))


i_hospitalized_P, i_icu_P, i_ventilated_P = get_dispositions(J_p, rates_p, regional_hosp_share)

r_hospitalized_P, r_icu_P, r_ventilated_P = get_dispositions(RH_p, rates_p, regional_hosp_share)
d_hospitalized_P, d_icu_P, d_ventilated_P = get_dispositions(D_p, rates_p, regional_hosp_share)

dispositions_mit1 = (
            i_hospitalized_P + r_hospitalized_P+ d_hospitalized_P ,
            i_icu_P+r_icu_P+d_icu_P,
            i_ventilated_P+r_ventilated_P +d_ventilated_P)

hospitalized_mit1, icu_mit1, ventilated_mit1 = (
            i_hospitalized_P,
            i_icu_P,
            i_ventilated_P)
            

# New Cases
projection_admits_mit1 = build_admissions_df_n(dispositions_mit1)
## Census Table
census_table_mit1 = build_census_df(projection_admits_mit1)

# Graph Function
def sepaijrd_mit_graph1(
    census: pd.DataFrame,
    plot_projection_days: int,
    as_date:bool = False) -> alt.Chart:
    """docstring"""
    census = census.rename(columns={"hosp": "SEPAIJRD Model Mitigatin v1"})

    tooltip_dict = {False: "day", True: "date:T"}
    if as_date:
        census = add_date_column(census.head(plot_projection_days))
        x_kwargs = {"shorthand": "date:T", "title": "Date"}
    else:
        x_kwargs = {"shorthand": "day", "title": "Days from initial infection"}

    return (
        alt
        .Chart(census)
        .transform_fold(fold=["SEPAIJRD Model Mitigation v1"])
        .mark_line(point=False)
        .encode(
            x=alt.X(**x_kwargs),
            y=alt.Y("value:Q", title="Census"),
            color=alt.value('gray'),
            #color="key:N",
            tooltip=[
                tooltip_dict[as_date],
                alt.Tooltip("value:Q", format=".0f", title="Census"),
                "key:N",
            ],
        )
        .interactive()
    )

# Graph
seir_mit1 = sepaijrd_mit_graph1(census_table_mit1, plot_projection_days, as_date=as_date)

#################################################################################################


######################################################################
# SEIR model with phase adjusted R_0 and Disease Related Fatality
# Asymptomatic, Hospitalization, Presymptomatic, and masks (Version 2)
# 11
######################################################################

S_default = 1400000
S = S_default
doubling_time=3
relative_contact_rate=0
intrinsic_growth_rate = 2 ** (1 / doubling_time) - 1
beta = (intrinsic_growth_rate + gamma) / S * (1-relative_contact_rate)

decay1 = 0
decay2 = 0.10
decay3 = 0.45
decay4 = 0.25
decay5 = 0.15
decay6 = 0.0

hosp_rate = 0.015
icu_rate = 0.25
vent_rate = 0.35
hosp_los = 6
icu_los = 11
vent_los = 10
incubation_period = 5.8

r_t = beta / gamma * S # r_t is r_0 after distancing
r_naught = (intrinsic_growth_rate + gamma) / gamma
doubling_time_t = 1/np.log2(beta*S - gamma +1) # doubling time after distancing

beta2 = (intrinsic_growth_rate + (1/infectious_period)) / S * (1-relative_contact_rate)
beta3 = ((alpha+intrinsic_growth_rate)*(intrinsic_growth_rate + (1/infectious_period))) / (alpha*S) *(1-relative_contact_rate)
beta4 = ((alpha+intrinsic_growth_rate)*(intrinsic_growth_rate + (1/infectious_period))) / (alpha*S) 
gamma2= 1/infectious_period
exposed2= beta4*S*total_infections
S2= S-exposed2-total_infections
gamma_hosp = 1 / hosp_los
icu_days = 1 / icu_los

asymptomatic = 0.36
q = 0.34
p_m1 = 0.4
p_m2 = 0.6
delta_p = 1.7 

## Asymptomatic, Hospitalization, Presymptomatic, and masks
E0=100
A0=100
I0=100
D0=0
R0=0
J0=0
P0=120
x=0.5
S0=S-E0-P0-A0-I0-D0-J0-R0
beta_j=0.6
q=0.583
l=0.717
gamma_hosp=1/hosp_lag
AAA=beta4*(1/gamma2)*S
beta_j=AAA*(1/(((1-asymptomatic)*1/gamma2)+(asymptomatic*q/(gamma2+hosp_rate))+(asymptomatic*hosp_rate*l/((gamma2+hosp_rate)*gamma_hosp))))

R0_n=beta_j* (((1-asymptomatic)*1/gamma2)+(asymptomatic*q/(gamma2+hosp_rate))+(asymptomatic*hosp_rate*l/((gamma2+hosp_rate)*gamma_hosp)))
beta_j=0.51
R0_n=beta_j* (((1-asymptomatic)*1/gamma2)+(asymptomatic*q/(gamma2+hosp_rate))+(asymptomatic*hosp_rate*l/((gamma2+hosp_rate)*gamma_hosp)))

S_p, E_p,P_p,A_p, I_p,J_p, R_p, D_p, RH_p=sim_sepaijrd_decay_ode(S0, E0, P0,A0,I0,J0, R0, D0, beta_j,gamma2, gamma_hosp, alpha, n_days,
                                                      decay1,decay2,decay3, decay4, decay5, start_day, int1_delta, int2_delta,
                                                      reopen1_delta, reopen2_delta, fatal_hosp,asymptomatic, hosp_rate, q,  l,x, p_m1, p_m2, delta_p)

icu_curve= J_p*icu_rate
vent_curve=J_p*vent_rate

hosp_rate_p=1.0
RateLos = namedtuple("RateLos", ("rate", "length_of_stay"))
hospitalized_p=RateLos(hosp_rate_p, hosp_los)
icu_rate_p= icu_rate
vent_rate_p= vent_rate
icu=RateLos(icu_rate_p, icu_los)
ventilated=RateLos(vent_rate_p, vent_los)


rates_p = tuple(each.rate for each in (hospitalized_p, icu, ventilated))
lengths_of_stay = tuple(each.length_of_stay for each in (hospitalized_p, icu, ventilated))


i_hospitalized_P, i_icu_P, i_ventilated_P = get_dispositions(J_p, rates_p, regional_hosp_share)

r_hospitalized_P, r_icu_P, r_ventilated_P = get_dispositions(RH_p, rates_p, regional_hosp_share)
d_hospitalized_P, d_icu_P, d_ventilated_P = get_dispositions(D_p, rates_p, regional_hosp_share)

dispositions_mit2 = (
            i_hospitalized_P + r_hospitalized_P+ d_hospitalized_P ,
            i_icu_P+r_icu_P+d_icu_P,
            i_ventilated_P+r_ventilated_P +d_ventilated_P)

hospitalized_mit2, icu_mit2, ventilated_mit2 = (
            i_hospitalized_P,
            i_icu_P,
            i_ventilated_P)

# New Cases Table
projection_admits_mit2 = build_admissions_df_n(dispositions_mit2)
## Census Table
census_table_mit2 = build_census_df(projection_admits_mit2)

# Graph Function
def sepaijrd_mit_graph2(
    census: pd.DataFrame,
    plot_projection_days: int,
    as_date:bool = False) -> alt.Chart:
    """docstring"""
    census = census.rename(columns={"hosp": "SEPAIJRD Model Mitigation v2"})

    tooltip_dict = {False: "day", True: "date:T"}
    if as_date:
        census = add_date_column(census.head(plot_projection_days))
        x_kwargs = {"shorthand": "date:T", "title": "Date"}
    else:
        x_kwargs = {"shorthand": "day", "title": "Days from initial infection"}

    return (
        alt
        .Chart(census)
        .transform_fold(fold=["SEPAIJRD Model Mitigation v2"])
        .mark_line(point=False)
        .encode(
            x=alt.X(**x_kwargs),
            y=alt.Y("value:Q", title="Census"),
            color=alt.value('brown'),
            #color="key:N",
            tooltip=[
                tooltip_dict[as_date],
                alt.Tooltip("value:Q", format=".0f", title="Census"),
                "key:N",
            ],
        )
        .interactive()
    )

# Graph
seir_mit2 = sepaijrd_mit_graph2(census_table_mit2, plot_projection_days, as_date=as_date)

#################################################################################################


st.header("""Projected Census Models for Erie County""")

st.subheader("Comparison of COVID-19 Disease Models")
st.altair_chart(
    alt.layer(sir.mark_line())
    + alt.layer(seir_0SD.mark_line())
    + alt.layer(seir_30SD.mark_line())
    + alt.layer(seir_R0_g1.mark_line())
    + alt.layer(seir_R0_g2.mark_line())
    + alt.layer(seir_R0_g3.mark_line())
    + alt.layer(seir_mit1.mark_line())
    + alt.layer(seir_mit2.mark_line())
    + alt.layer(seir_D.mark_line())
    + alt.layer(seir_A.mark_line())
    + alt.layer(seir_highsocial.mark_line())
    #+ alt.layer(erie_lines_ip)
    + alt.layer(vertical)
    , use_container_width=True)

st.subheader("Comparison of COVID-19 Disease Models: SIR and SEIR")
st.altair_chart(
    alt.layer(sir.mark_line())
    + alt.layer(seir_0SD.mark_line())
    + alt.layer(seir_30SD.mark_line())
    # + alt.layer(seir_R0_g1.mark_line())
    # + alt.layer(seir_R0_g2.mark_line())
    # + alt.layer(seir_R0_g3.mark_line())
    # + alt.layer(seir_mit1.mark_line())
    # + alt.layer(seir_mit2.mark_line())
    # + alt.layer(seir_D.mark_line())
    # + alt.layer(seir_A.mark_line())
    # + alt.layer(seir_highsocial.mark_line())
    # + alt.layer(erie_lines_ip)
    + alt.layer(vertical)
    , use_container_width=True)

st.subheader("Comparison of COVID-19 Disease Models: SEIR with adjusted R0")
st.altair_chart(
    #alt.layer(sir.mark_line())
    #+ alt.layer(seir_0SD.mark_line())
    # + alt.layer(seir_30SD.mark_line())
    # +
    alt.layer(seir_R0_g1.mark_line())
    + alt.layer(seir_R0_g2.mark_line())
    + alt.layer(seir_R0_g3.mark_line())
    # + alt.layer(seir_mit1.mark_line())
    # + alt.layer(seir_mit2.mark_line())
    # + alt.layer(seir_D.mark_line())
    # + alt.layer(seir_A.mark_line())
    # + alt.layer(seir_highsocial.mark_line())
    # + alt.layer(erie_lines_ip)
    + alt.layer(vertical)
    , use_container_width=True)


st.subheader("Comparison of COVID-19 Disease Models: SEIR with compartments")
st.altair_chart(
    # alt.layer(sir.mark_line())
    # + alt.layer(seir_0SD.mark_line())
    # + alt.layer(seir_30SD.mark_line())
    # + alt.layer(seir_R0_g1.mark_line())
    # + alt.layer(seir_R0_g2.mark_line())
    # + alt.layer(seir_R0_g3.mark_line())
    # + alt.layer(seir_mit1.mark_line())
    # + alt.layer(seir_mit2.mark_line())
    # + 
    alt.layer(seir_D.mark_line())
    + alt.layer(seir_A.mark_line())
    # + alt.layer(seir_highsocial.mark_line())
    # + alt.layer(erie_lines_ip)
    + alt.layer(vertical)
    , use_container_width=True)
    

st.subheader("Comparison of COVID-19 Disease Models: SEIR with adjusted R0, compartments and mitigation")
st.altair_chart(
    #alt.layer(sir_census.mark_line())
    #+ alt.layer(seir_0SD.mark_line())
    #+ alt.layer(seir_30SD.mark_line())
    #+ alt.layer(seir_R0_g1.mark_line())
    #+ alt.layer(seir_R0_g2.mark_line())
    #+ alt.layer(seir_R0_g3.mark_line())
    #+ 
    alt.layer(seir_mit1.mark_line()) # Missing line, don't know why
    + alt.layer(seir_mit2.mark_line())
    #+ alt.layer(seir_A.mark_line())
    #+ alt.layer(seir_highsocial.mark_line())
    + alt.layer(erie_lines_ip)
    + alt.layer(vertical)
    , use_container_width=True)






