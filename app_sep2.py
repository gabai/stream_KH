# Modified version for Erie County, New York
# Contact: ganaya@buffalo.edu

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

# Create S3 object to get the ENV variable from Heroku
#secret = os.environ['SECRET_KEY']

# Prompt the user for the secret
#password = st.text_input("Secret Handshake:", value="", type="password")

# If the secrete provided matches the ENV, proceeed with the app
#if password == secret:
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
    #projection_admits.loc[0,'hosp'] = 25
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
    projection_admits.loc[0,'hosp'] = 25
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
    census_df.loc[0,'hosp'] = 45

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

# def sim_seir_decay(
    # s: float, e:float, i: float, r: float, beta: float, gamma: float, alpha: float, n_days: int,
    # decay1:float, decay2:float, decay3: float
    # ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # """Simulate the SIR model forward in time."""
    # s, e, i, r = (float(v) for v in (s, e, i, r))
    # n = s + e + i + r
    # s_v, e_v, i_v, r_v = [s], [e], [i], [r]
    # for day in range(n_days):
        # if start_day<=day<=int1_delta:
            # beta_decay=beta*(1-decay1)
        # elif int1_delta<=day<=int2_delta:
            # beta_decay=beta*(1-decay2)
        # elif int2_delta<=day<=int3_delta:
            # beta_decay=beta*(1-decay3)
        # elif int3_delta<=day<=n_days:
            # beta_decay=beta*(1-decay4)
        # s, e, i, r = seir(s, e, i, r, beta_decay, gamma, alpha, n)
        # s_v.append(s)
        # e_v.append(e)
        # i_v.append(i)
        # r_v.append(r)

    # return (
        # np.array(s_v),
        # np.array(e_v),
        # np.array(i_v),
        # np.array(r_v),
    # )

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

# def sim_seird_decay(
        # s: float, e:float, i: float, r: float, d: float, beta: float, gamma: float, alpha: float, n_days: int,
        # decay1:float, decay2:float, decay3: float, decay4: float, step1_delta: int, fatal: float
        # ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # """Simulate the SIR model forward in time."""
        # s, e, i, r, d= (float(v) for v in (s, e, i, r, d))
        # n = s + e + i + r + d
        # s_v, e_v, i_v, r_v, d_v = [s], [e], [i], [r], [d]
        # for day in range(n_days):
            # if start_day<=day<=int1_delta:
                # beta_decay=beta*(1-decay1)
            # elif int1_delta<=day<=int2_delta:
                # beta_decay=beta*(1-decay2)
            # elif int2_delta<=day<=step1_delta:
                # beta_decay=beta*(1-decay3)
            # else:
                # beta_decay=beta*(1-decay4)
            # s, e, i, r,d = seird(s, e, i, r, d, beta_decay, gamma, alpha, n, fatal)
            # s_v.append(s)
            # e_v.append(e)
            # i_v.append(i)
            # r_v.append(r)
            # d_v.append(d)

        # return (
            # np.array(s_v),
            # np.array(e_v),
            # np.array(i_v),
            # np.array(r_v),
            # np.array(d_v)
        # )


# # Model with high social distancing
# def sim_seird_decay_social(
    # s: float, e:float, i: float, r: float, d: float, beta: float, gamma: float, alpha: float, n_days: int,
    # decay1:float, decay2:float, decay3: float, decay4: float, step1_delta: int, fatal: float
    # ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # """Simulate the SIR model forward in time."""
    # s, e, i, r, d= (float(v) for v in (s, e, i, r, d))
    # n = s + e + i + r + d
    # s_v, e_v, i_v, r_v, d_v = [s], [e], [i], [r], [d]
    # for day in range(n_days):
        # if start_day<=day<=int1_delta:
            # beta = (alpha+(2 ** (1 / 2) - 1))*((2 ** (1 / 2) - 1) + (1/infectious_period)) / (alpha*S)
            # beta_decay=beta*(1-.02)
        # elif int1_delta<=day<=int2_delta:
            # beta = (alpha+(2 ** (1 / 2) - 1))*((2 ** (1 / 2) - 1)+ (1/infectious_period)) / (alpha*S)
            # beta_decay=beta*(1-.52)
        # elif int2_delta<=day<=step1_delta:
            # beta = (alpha+(2 ** (1 / 2) - 1))*((2 ** (1 / 2) - 1)+ (1/infectious_period)) / (alpha*S)
            # beta_decay=beta*(1-.83)
        # else:
            # beta = (alpha+(2 ** (1 / 2) - 1))*((2 ** (1 / 2) - 1)+ (1/infectious_period)) / (alpha*S)
            # beta_decay=beta*(1-.73)
        # s, e, i, r,d = seird(s, e, i, r, d, beta_decay, gamma, alpha, n, fatal)
        # s_v.append(s)
        # e_v.append(e)
        # i_v.append(i)
        # r_v.append(r)
        # d_v.append(d)

    # return (
        # np.array(s_v),
        # np.array(e_v),
        # np.array(i_v),
        # np.array(r_v),
        # np.array(d_v)
    # )

# # Model with dynamic doubling time
# def sim_seird_decay_erie(
    # s: float, e:float, i: float, r: float, d: float, beta: float, gamma: float, alpha: float, n_days: int,
    # decay1:float, decay2:float, decay3: float, decay4: float, step1_delta: int, fatal: float
    # ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # """Simulate the SIR model forward in time."""
    # s, e, i, r, d= (float(v) for v in (s, e, i, r, d))
    # n = s + e + i + r + d
    # s_v, e_v, i_v, r_v, d_v = [s], [e], [i], [r], [d]
    # for day in range(n_days):
        # if start_day<=day<=int1_delta:
            # beta = (alpha+(2 ** (1 / 1.61) - 1))*((2 ** (1 / 1.61) - 1) + (1/infectious_period)) / (alpha*S)
            # beta_decay=beta*(1-.3)
        # elif int1_delta<=day<=int2_delta:
            # beta = (alpha+(2 ** (1 / 2.65) - 1))*((2 ** (1 / 2.65) - 1)+ (1/infectious_period)) / (alpha*S)
            # beta_decay=beta*(1-.3)
        # elif int2_delta<=day<=step1_delta:
            # beta = (alpha+(2 ** (1 / 5.32) - 1))*((2 ** (1 / 5.32) - 1)+ (1/infectious_period)) / (alpha*S)
            # beta_decay=beta*(1-.5)
        # else:
            # beta = (alpha+(2 ** (1 / 9.70) - 1))*((2 ** (1 / 9.70) - 1)+ (1/infectious_period)) / (alpha*S)
            # beta_decay=beta*(1-.30)
        # s, e, i, r,d = seird(s, e, i, r, d, beta_decay, gamma, alpha, n, fatal)
        # s_v.append(s)
        # e_v.append(e)
        # i_v.append(i)
        # r_v.append(r)
        # d_v.append(d)

    # return (
        # np.array(s_v),
        # np.array(e_v),
        # np.array(i_v),
        # np.array(r_v),
        # np.array(d_v)
    # )

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

# def sim_seijcrd_decay(
    # s: float, e:float, i: float, j:float, c: float, r: float, d: float, beta: float, gamma: float, alpha: float, n_days: int,
    # decay1:float, decay2:float, decay3: float, decay4: float, step1_delta: int, fatal_hosp: float, hosp_rate: float, icu_rate: float, icu_days:float, crit_lag: float, death_days:float
    # ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # """Simulate the SIR model forward in time."""
    # s, e, i, j, c, r, d= (float(v) for v in (s, e, i, c, j, r, d))
    # n = s + e + i + j+r + d
    # s_v, e_v, i_v, j_v, c_v, r_v, d_v = [s], [e], [i], [j], [c], [r], [d]
    # for day in range(n_days):
        # if 0<=day<=21:
            # beta = (alpha+(2 ** (1 / 1.61) - 1))*((2 ** (1 / 1.61) - 1) + (1/infectious_period)) / (alpha*S)
            # beta_decay=beta*(1-decay1)
        # elif 22<=day<=28:
            # beta = (alpha+(2 ** (1 / 2.65) - 1))*((2 ** (1 / 2.65) - 1)+ (1/infectious_period)) / (alpha*S)
            # beta_decay=beta*(1-decay2)
        # elif 29<=day<=step1_delta:
            # beta = (alpha+(2 ** (1 / 5.32) - 1))*((2 ** (1 / 5.32) - 1)+ (1/infectious_period)) / (alpha*S)
            # beta_decay=beta*(1-decay3)
        # else:
            # beta = (alpha+(2 ** (1 / 9.70) - 1))*((2 ** (1 / 9.70) - 1)+ (1/infectious_period)) / (alpha*S)
            # beta_decay=beta*(1-decay4)
        # s, e, i,j, c, r,d = seijcrd(s, e, i,j, c, r, d, beta_decay, gamma, alpha, n, fatal_hosp, hosp_rate, icu_rate, icu_days, crit_lag, death_days)
        # s_v.append(s)
        # e_v.append(e)
        # i_v.append(i)
        # j_v.append(j)
        # c_v.append(c)
        # r_v.append(r)
        # d_v.append(d)

    # return (
        # np.array(s_v),
        # np.array(e_v),
        # np.array(i_v),
        # np.array(j_v),
        # np.array(c_v),
        # np.array(r_v),
        # np.array(d_v)
    # )


# def betanew(t,beta):
    # if start_day<= t <= int1_delta:
        # beta_decay=beta*(1-decay1)
    # elif int1_delta<=t<int2_delta:
        # beta_decay=beta*(1-decay2)
    # elif int2_delta<=t<int3_delta:
        # beta_decay=beta*(1-decay3)
    # elif int3_delta<=t<=int4_delta:
        # beta_decay=beta*(1-decay4)
    # elif int4_delta<=t<=int5_delta:
        # beta_decay=beta*(1-decay5)
    # elif int5_delta<=t<=int6_delta:
        # beta_decay=beta*(1-decay6)
    # elif int6_delta<=t<=int7_delta:
        # beta_decay=beta*(1-decay7)
    # elif int7_delta<=t<=int8_delta:
        # beta_decay=beta*(1-decay8)
    # return beta_decay

#The SIR model differential equations with ODE solver.
def derivdecay(y, t, N, beta, gamma1, gamma2, alpha, p, hosp, q, l, n_days, decay1, decay2, decay3, decay4, decay5, decay6, decay7, decay8, decay9, decay10,
                start_day, int1_delta, int2_delta, int3_delta, int4_delta, int5_delta, fatal_hosp):
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
    s, e,a,i, j,r, d, beta, gamma1, gamma2, alpha, n_days, decay1, decay2, decay3, decay4, decay5, decay6, decay7, decay8, decay9, decay10,
    start_day, int1_delta, int2_delta, int3_delta, int4_delta, int5_delta, fatal_hosp, p, hosp, q, l):
    n = s + e + a + i + j+ r + d
    rh=0
    y0= s,e,a,i,j,r,d, rh

    t=np.arange(0, n_days, step=1)
    ret = odeint(derivdecay, y0, t, args=(n, beta, gamma1, gamma2, alpha, p, hosp,q,l, n_days, decay1, decay2, decay3, decay4, decay5, decay6, decay7, decay8, decay9, decay10,
    start_day, int1_delta, int2_delta, int3_delta, int4_delta, int5_delta, fatal_hosp))
    S_n, E_n,A_n, I_n,J_n, R_n, D_n ,RH_n= ret.T

    return (S_n, E_n, A_n, I_n,J_n, R_n, D_n, RH_n)


####The SIR model differential equations with ODE solver. Presymptomatic and masks
# def betanew2(t, beta, x, p_m1, pm_2, p_m3):
    # beta_decay = 0
    # if start_day<=t<= int1_delta:
        # beta_decay=beta*(1-decay1)*(1-(x*p_m1))**2
    # elif int1_delta<t<=int2_delta:
        # beta_decay=beta*(1-decay2)*(1-(x*p_m2))**2
    # elif int2_delta<t<=n_days:
        # beta_decay=beta*(1-decay3)*(1-(x*p_m3))**2
    # return beta_decay

# Modified 9/22/21
def betanew2(t, beta, x, p_m1, pm_2, p_m3, p_m4, p_m5, p_m6, p_m7, p_m8, p_m9, p_m10):
    beta_decay = 0.0
    if start_day<=t<=int1_delta:
        beta_decay=beta*(1-decay1)*(1-(x*p_m1))**2
    elif int1_delta<t<=int2_delta:
        beta_decay=beta*(1-decay2)*(1-(x*p_m2))**2
    elif int2_delta<t<=int3_delta:
        beta_decay=beta*(1-decay3)*(1-(x*p_m3))**2
    elif int3_delta<t<=int4_delta:
        beta_decay=beta*(1-decay4)*(1-(x*p_m4))**2
    elif int4_delta<t<=int5_delta:
        beta_decay=beta*(1-decay5)*(1-(x*p_m5))**2
    elif int5_delta<t<=int6_delta:
        beta_decay=beta*(1-decay6)*(1-(x*p_m6))**2
    elif int6_delta<t<=int7_delta:
        beta_decay=beta*(1-decay7)*(1-(x*p_m7))**2
    elif int7_delta<t<=int8_delta:
        beta_decay=beta*(1-decay8)*(1-(x*p_m8))**2
    elif int8_delta<t<=int9_delta:
        beta_decay=beta*(1-decay9)*(1-(x*p_m9))**2
    elif int9_delta<t<n_days:
        beta_decay=beta*(1-decay10)*(1-(x*p_m10))**2
    return beta_decay


### changing up new_strain
# def betanew3(t, beta, x, p_m1, pm_2, p_m3, p_m4, p_m5, p_m6, p_m7, p_m8, p_m9, p_m10, new_strain, fracNS):
    # beta_decay = 0.0
    # if start_day<=t<=int1_delta:
        # beta_decay=beta*(1-decay1)*(1-(x*p_m1))**2
    # elif int1_delta<t<=int2_delta:
        # beta_decay=beta*(1-decay2)*(1-(x*p_m2))**2
    # elif int2_delta<t<=int3_delta:
        # beta_decay=beta*(1-decay3)*(1-(x*p_m3))**2
    # elif int3_delta<t<=int4_delta:
        # beta_decay=beta*(1-decay4)*(1-(x*p_m4))**2
    # elif int4_delta<t<=int5_delta:
        # beta_decay=beta*(1-decay5)*(1-(x*p_m5))**2
    # elif int5_delta<t<=int6_delta:
        # beta_decay=((1-fracNS)*beta*(1-decay6)*(1-(x*p_m6))**2)+(fracNS*(1+new_strain)*beta*(1-decay6)*(1-(x*p_m6))**2)
    # elif int6_delta<t<=int7_delta, int8_delta:
        # beta_decay=((1-fracNS)*beta*(1-decay7)*(1-(x*p_m7))**2)+(fracNS*(1+new_strain)*beta*(1-decay7)*(1-(x*p_m7))**2)
    # else:
        # beta_decay=((1-fracNS)*beta*(1-decay8, decay9, decay10,)*(1-(x*p_m8, p_m9, p_m10,))**2)+(fracNS*(1+new_strain)*beta*(1-decay8, decay9, decay10,)*(1-(x*p_m8, p_m9, p_m10,))**2)
    # return beta_decay


# Modified 9/22/21
# Adding period specific changes to tranmission or percentage of population
def betanew3(t, beta, x, p_m1, pm_2, p_m3, p_m4, p_m5, p_m6, p_m7, p_m8, p_m9, p_m10, 
                new_strain6, new_strain7, new_strain8, new_strain9, new_strain10, 
                fracNS6, fracNS7, fracNS8, fracNS9, fracNS10):
    beta_decay = 0.0
    if start_day<=t<=int1_delta:
        beta_decay=beta*(1-decay1)*(1-(x*p_m1))**2
    elif int1_delta<t<=int2_delta:
        beta_decay=beta*(1-decay2)*(1-(x*p_m2))**2
    elif int2_delta<t<=int3_delta:
        beta_decay=beta*(1-decay3)*(1-(x*p_m3))**2
    elif int3_delta<t<=int4_delta:
        beta_decay=beta*(1-decay4)*(1-(x*p_m4))**2
    elif int4_delta<t<=int5_delta:
        beta_decay=beta*(1-decay5)*(1-(x*p_m5))**2
    elif int5_delta<t<=int6_delta:
        beta_decay=((1-fracNS6)*beta*(1-decay6)*(1-(x*p_m6))**2)+(fracNS6*(1+new_strain6)*beta*(1-decay6)*(1-(x*p_m6))**2)
    elif int6_delta<t<=int7_delta:
        beta_decay=((1-fracNS7)*beta*(1-decay7)*(1-(x*p_m7))**2)+(fracNS7*(1+new_strain7)*beta*(1-decay7)*(1-(x*p_m7))**2)
    elif int7_delta<t<=int8_delta:
        beta_decay=((1-fracNS8)*beta*(1-decay8)*(1-(x*p_m8))**2)+(fracNS8*(1+new_strain8)*beta*(1-decay8)*(1-(x*p_m8))**2)
    elif int8_delta<t<=int9_delta:
        beta_decay=((1-fracNS9)*beta*(1-decay9)*(1-(x*p_m9))**2)+(fracNS9*(1+new_strain9)*beta*(1-decay9)*(1-(x*p_m9))**2)
    elif int9_delta<t<n_days:
        beta_decay=((1-fracNS10)*beta*(1-decay10)*(1-(x*p_m10))**2)+(fracNS10*(1+new_strain10)*beta*(1-decay10)*(1-(x*p_m10))**2)
    return beta_decay



####### what if we want to do this for vaccination!

# def phinew2(t, phi):
    # phi_decay = 0.0
    # if start_day<=t<=int1_delta:
        # phi_decay = 0
    # elif int1_delta<t<=int4_delta:
        # phi_decay = 0
    # elif int4_delta<t<=int5_delta:
        # phi_decay = 0.003
    # elif int5_delta<t<=int6_delta:
        # phi_decay = 0.003
    # elif int6_delta<t<=int7_delta:
        # phi_decay = 0.003
    # else:
        # phi_decay = phi
    # return phi_decay


# Modified 9/22/21 # Addind additional timepoint - now at 10
def phinew2(t, phi5, phi6, phi7, phi8, phi9, phi10):
    phi_decay = 0.0
    if start_day<=t<=int1_delta:
        phi_decay = 0
    elif int1_delta<t<=int4_delta:
        phi_decay = 0
    elif int4_delta<t<=int5_delta:
        phi_decay = phi5
    elif int5_delta<t<=int6_delta:
        phi_decay = phi6
    elif int6_delta<t<=int7_delta:
        phi_decay = phi7
    elif int7_delta<t<=int8_delta:
        phi_decay = phi8
    elif int8_delta<t<=int9_delta:
        phi_decay = phi9
    elif int9_delta<t<n_days:
        phi_decay = phi10
    return phi_decay


# def betanewstrain(t, beta, x, p_m1, pm_2, p_m3, p_m4, p_m5):
    # beta_decay = 0.0
    # if start_day<=t<=int1_delta:
        # beta_decay=beta*(1-decay1)*(1-(x*p_m1))**2
    # elif int1_delta<t<=int2_delta:
        # beta_decay=beta*(1-decay2)*(1-(x*p_m2))**2
    # elif int2_delta<t<=int3_delta:
        # beta_decay=beta*(1-decay3)*(1-(x*p_m3))**2
    # elif int3_delta<t<=int4_delta:
        # beta_decay=beta*(1-decay4)*(1-(x*p_m4))**2
    # else:
        # beta_decay=beta*(1-decay5)*(1-(x*p_m5))**2
    # return beta_decay


def derivdecayP(y, t, beta, gamma1, gamma2, alpha, sym, hosp, q, l, n_days, decay1, decay2, decay3, decay4, decay5, decay6, decay7, decay8, decay9, decay10,
                start_day, int1_delta, int2_delta, int3_delta, int4_delta, int5_delta, int6_delta, int7_delta, int8_delta, int9_delta,
                fatal_hosp, x, p_m1, p_m2, p_m3, p_m4, p_m5, p_m6, p_m7, p_m8, p_m9, p_m10, delta_p):
    S, E, P,A, I,J, R,D,counter = y
    N=S+E+P+A+I+J+R+D
    dSdt = - betanew2(t, beta, x, p_m1, p_m2, p_m3, p_m4, p_m5, p_m6, p_m7, p_m8, p_m9, p_m10) * S * (q*I + l*J +P+ A)/N
    dEdt = betanew2(t, beta, x, p_m1, p_m2, p_m3, p_m4, p_m5, p_m6, p_m7, p_m8, p_m9, p_m10) * S * (q*I + l*J +P+ A)/N   - alpha * E
    dPdt = alpha * E - delta_p * P
    dAdt = delta_p* P *(1-sym)-gamma1*A
    dIdt = sym* delta_p* P - gamma1 * I- hosp*I
    dJdt = hosp * I -gamma2*J
    dRdt = (1-fatal_hosp)*gamma2 * J + gamma1*(A+I)
    dDdt = fatal_hosp * gamma2 * J
    counter = (1-fatal_hosp)*gamma2 * J
    return dSdt, dEdt,dPdt,dAdt, dIdt, dJdt, dRdt, dDdt, counter


## vaccination rate is standard across time
def derivdecayV(y, t, beta, gamma1, gamma2, alpha, sym, hosp, q, l, n_days, decay1, decay2, decay3, decay4, decay5, decay6, decay7, decay8, decay9, decay10,
                start_day, int1_delta, int2_delta, int3_delta, int4_delta, int5_delta, int6_delta, int7_delta, int8_delta, int9_delta,
                fatal_hosp, x, p_m1, p_m2, p_m3, p_m4, p_m5, p_m6, p_m7, p_m8, p_m9, p_m10, delta_p, sigma, phi):
    # here sigma=scaling rate of how effective the vaccine is
    # phi=rate suscepitble individuals are vaccinated at each time step
    S, V,E, P,A, I,J, R,D,counter = y
    N=S+E+P+A+I+J+R+D+V
    dSdt = - betanew2(t, beta, x, p_m1, p_m2, p_m3, p_m4, p_m5, p_m6, p_m7, p_m8, p_m9, p_m10) * S * (q*I + l*J +P+ A)/N - (phi*S)
    dVdt = (phi*S)-(sigma * betanew2(t, beta, x, p_m1, p_m2, p_m3, p_m4, p_m5, p_m6, p_m7, p_m8, p_m9, p_m10) * V * (q*I + l*J +P+ A)/N )
    dEdt = (betanew2(t, beta, x, p_m1, p_m2, p_m3, p_m4, p_m5, p_m6, p_m7, p_m8, p_m9, p_m10) * S * (q*I + l*J +P+ A)/N )+(sigma * betanew2(t, beta, x, p_m1, p_m2, p_m3, p_m4, p_m5, p_m6, p_m7, p_m8, p_m9, p_m10) * V * (q*I + l*J +P+ A)/N )  - alpha * E
    dPdt = alpha * E - delta_p * P
    dAdt = delta_p* P *(1-sym)-gamma1*A
    dIdt = sym* delta_p* P - gamma1 * I- hosp*I
    dJdt = hosp * I -gamma2*J
    dRdt = (1-fatal_hosp)*gamma2 * J + gamma1*(A+I)
    dDdt = fatal_hosp * gamma2 * J
    counter = (1-fatal_hosp)*gamma2 * J
    return dSdt, dVdt,dEdt,dPdt,dAdt, dIdt, dJdt, dRdt, dDdt, counter

### this adds in the fact that vaccination was not in the first part of this!
def derivdecayVtime(y, t, beta, gamma1, gamma2, alpha, sym, hosp, q, l, n_days, decay1, decay2, decay3, decay4, decay5, decay6, decay7, decay8, decay9, decay10,
                start_day, int1_delta, int2_delta, int3_delta, int4_delta, int5_delta, int6_delta, int7_delta, int8_delta, int9_delta,
                fatal_hosp, x, p_m1, p_m2, p_m3, p_m4, p_m5, p_m6, p_m7, p_m8, p_m9, p_m10, delta_p, sigma, phi):
    # here sigma=scaling rate of how effective the vaccine is
    # phi=rate suscepitble individuals are vaccinated at each time step
    S, V,E, P,A, I,J, R,D,counter = y
    N=S+E+P+A+I+J+R+D+V
    dSdt = - betanew2(t, beta, x, p_m1, p_m2, p_m3, p_m4, p_m5, p_m6, p_m7, p_m8, p_m9, p_m10) * S * (q*I + l*J +P+ A)/N - (phinew2(t,phi5, phi6, phi7, phi8, phi9, phi10)*S)
    dVdt = (phinew2(t,phi5, phi6, phi7, phi8, phi9, phi10)*S)-(sigma * betanew2(t, beta, x, p_m1, p_m2, p_m3, p_m4, p_m5, p_m6, p_m7, p_m8, p_m9, p_m10) * V * (q*I + l*J +P+ A)/N )
    dEdt = (betanew2(t, beta, x, p_m1, p_m2, p_m3, p_m4, p_m5, p_m6, p_m7, p_m8, p_m9, p_m10) * S * (q*I + l*J +P+ A)/N )+(sigma * betanew2(t, beta, x, p_m1, p_m2, p_m3, p_m4, p_m5, p_m6, p_m7, p_m8, p_m9, p_m10) * V * (q*I + l*J +P+ A)/N )  - alpha * E
    dPdt = alpha * E - delta_p * P
    dAdt = delta_p* P *(1-sym)-gamma1*A
    dIdt = sym* delta_p* P - gamma1 * I- hosp*I
    dJdt = hosp * I -gamma2*J
    dRdt = (1-fatal_hosp)*gamma2 * J + gamma1*(A+I)
    dDdt = fatal_hosp * gamma2 * J
    counter = (1-fatal_hosp)*gamma2 * J
    return dSdt, dVdt,dEdt,dPdt,dAdt, dIdt, dJdt, dRdt, dDdt, counter

###add in New Strain
def derivdecayVNS(y, t, beta, gamma1, gamma2, alpha, sym, hosp, q, l, n_days, decay1, decay2, decay3, decay4, decay5, decay6, decay7, decay8, decay9, decay10,
                start_day, int1_delta, int2_delta, int3_delta, int4_delta, int5_delta, int6_delta, int7_delta, int8_delta, int9_delta,
                fatal_hosp, x, p_m1, p_m2, p_m3, p_m4, p_m5, p_m6, p_m7, p_m8, p_m9, p_m10, delta_p, sigma, 
                phi5, phi6, phi7, phi8, phi9, phi10,
                new_strain6, new_strain7, new_strain8, new_strain9, new_strain10,
                fracNS6, fracNS7, fracNS8, fracNS9, fracNS10):
    # here sigma=scaling rate of how effective the vaccine is
    # phi=rate suscepitble individuals are vaccinated at each time step
    S, V,E, P,A, I,J, R,D,counter = y
    N=S+E+P+A+I+J+R+D+V
    dSdt = - betanew3(t, beta, x, p_m1, p_m2, p_m3, p_m4, p_m5, p_m6, p_m7, p_m8, p_m9, p_m10, new_strain6, new_strain7, new_strain8, new_strain9, new_strain10, fracNS6, fracNS7, fracNS8, fracNS9, fracNS10) * S * (q*I + l*J +P+ A)/N - (phinew2(t,phi5, phi6, phi7, phi8, phi9, phi10)*S)
    dVdt = (phinew2(t,phi5, phi6, phi7, phi8, phi9, phi10)*S)-(sigma * betanew3(t, beta, x, p_m1, p_m2, p_m3, p_m4, p_m5, p_m6, p_m7, p_m8, p_m9, p_m10, new_strain6, new_strain7, new_strain8, new_strain9, new_strain10, fracNS6, fracNS7, fracNS8, fracNS9, fracNS10) * V * (q*I + l*J +P+ A)/N )
    dEdt = (betanew3(t, beta, x, p_m1, p_m2, p_m3, p_m4, p_m5, p_m6, p_m7, p_m8, p_m9, p_m10, new_strain6, new_strain7, new_strain8, new_strain9, new_strain10, fracNS6, fracNS7, fracNS8, fracNS9, fracNS10) * S * (q*I + l*J +P+ A)/N )+(sigma * betanew3(t, beta, x, p_m1, p_m2, p_m3, p_m4, p_m5, p_m6, p_m7, p_m8, p_m9, p_m10, new_strain6, new_strain7, new_strain8, new_strain9, new_strain10, fracNS6, fracNS7, fracNS8, fracNS9, fracNS10) * V * (q*I + l*J +P+ A)/N )  - alpha * E
    dPdt = alpha * E - delta_p * P
    dAdt = delta_p* P *(1-sym)-gamma1*A
    dIdt = sym* delta_p* P - gamma1 * I- hosp*I
    dJdt = hosp * I -gamma2*J
    dRdt = (1-fatal_hosp)*gamma2 * J + gamma1*(A+I)
    dDdt = fatal_hosp * gamma2 * J
    counter = (1-fatal_hosp)*gamma2 * J
    return dSdt, dVdt,dEdt,dPdt,dAdt, dIdt, dJdt, dRdt, dDdt, counter


def sim_sepaijrd_decay_ode(
    s, e,p,a,i, j,r, d, beta, gamma1, gamma2, alpha, n_days, decay1, decay2, decay3, decay4, decay5, decay6, decay7, decay8, decay9, decay10,
    start_day, int1_delta, int2_delta, int3_delta, int4_delta, int5_delta, int6_delta, int7_delta, int8_delta, int9_delta,
    fatal_hosp, sym, hosp, q,
    l,x,
    p_m1, p_m2, p_m3, p_m4, p_m5, p_m6, p_m7, p_m8, p_m9, p_m10, delta_p):
    n = s + e + p+a + i + j+ r + d
    rh=0
    y0= s,e,p,a,i,j,r,d, rh

    t=np.arange(0, n_days, step=1)
    ret = odeint(derivdecayP, y0, t, args=(beta, gamma1, gamma2, alpha, sym, hosp, q , l, n_days,
    decay1, decay2, decay3, decay4, decay5, decay6, decay7, decay8, decay9, decay10,
    start_day, int1_delta, int2_delta, int3_delta, int4_delta, int5_delta, int6_delta, int7_delta, int8_delta, int9_delta, fatal_hosp, x,
    p_m1, p_m2, p_m3, p_m4, p_m5, p_m6, p_m7, p_m8, p_m9, p_m10, delta_p))
    S_n, E_n,P_n, A_n, I_n,J_n, R_n, D_n ,RH_n= ret.T

    return (S_n, E_n,P_n, A_n, I_n,J_n, R_n, D_n, RH_n)

def sim_svepaijrd_decay_ode(
    s,v, e,p,a,i, j,r, d, beta, gamma1, gamma2, alpha, n_days, decay1, decay2, decay3, decay4, decay5, decay6, decay7, decay8, decay9, decay10,
    start_day, int1_delta, int2_delta, int3_delta, int4_delta, int5_delta, int6_delta, int7_delta, int8_delta, int9_delta,
    fatal_hosp, sym, hosp, q,
    l,x,
    p_m1, p_m2, p_m3, p_m4, p_m5, p_m6, p_m7, p_m8, p_m9, p_m10, delta_p, sigma, phi):
    n = s + v+ e + p+a + i + j+ r + d
    rh=0
    y0= s,v,e,p,a,i,j,r,d, rh

    t=np.arange(0, n_days, step=1)
    ret = odeint(derivdecayVtime, y0, t, args=(beta, gamma1, gamma2, alpha, sym, hosp,q , l, n_days,
    decay1, decay2, decay3, decay4, decay5, decay6, decay7, decay8, decay9, decay10,
    start_day, int1_delta, int2_delta, int3_delta, int4_delta, int5_delta, int6_delta, int7_delta, int8_delta, int9_delta, fatal_hosp, x,
    p_m1, p_m2, p_m3, p_m4, p_m5, p_m6, p_m7, p_m8, p_m9, p_m10, delta_p, sigma, phi))
    S_n, V_n, E_n,P_n, A_n, I_n,J_n, R_n, D_n ,RH_n= ret.T

    return (S_n, V_n,E_n,P_n,A_n, I_n,J_n, R_n, D_n, RH_n)

def sim_svepaijrdNS_decay_ode(
    s,v, e,p,a,i, j,r, d, beta, gamma1, gamma2, alpha, n_days, decay1, decay2, decay3, decay4, decay5, decay6, decay7, decay8, decay9, decay10,
    start_day, int1_delta, int2_delta, int3_delta, int4_delta, int5_delta, int6_delta, int7_delta, int8_delta, int9_delta,
    fatal_hosp, sym, hosp, q,
    l,x,
    p_m1, p_m2, p_m3, p_m4, p_m5, p_m6, p_m7, p_m8, p_m9, p_m10, delta_p, sigma, phi5, phi6, phi7, phi8, phi9, phi10, new_strain6, new_strain7, new_strain8, new_strain9, new_strain10, fracNS6, fracNS7, fracNS8, fracNS9, fracNS10):
    n = s + v+ e + p+a + i + j+ r + d
    rh=0
    y0= s,v,e,p,a,i,j,r,d, rh

    t=np.arange(0, n_days, step=1)
    ret = odeint(derivdecayVNS, y0, t, args=(beta, gamma1, gamma2, alpha, sym, hosp,q , l, n_days,
    decay1, decay2, decay3, decay4, decay5, decay6, decay7, decay8, decay9, decay10,
    start_day, int1_delta, int2_delta, int3_delta, int4_delta, int5_delta, int6_delta, int7_delta, int8_delta, int9_delta, fatal_hosp, x,
    p_m1, p_m2, p_m3, p_m4, p_m5, p_m6, p_m7, p_m8, p_m9, p_m10, delta_p, sigma, phi5, phi6, phi7, phi8, phi9, phi10, new_strain6, new_strain7, new_strain8, new_strain9, new_strain10, fracNS6, fracNS7, fracNS8, fracNS9, fracNS10))
    S_n, V_n, E_n,P_n,A_n, I_n,J_n, R_n, D_n ,RH_n= ret.T
    return (S_n, V_n, E_n, P_n, A_n, I_n, J_n, R_n, D_n, RH_n)


# End Models #

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


url = 'https://raw.githubusercontent.com/gabai/stream_KH/master/Cases_Erie.csv'
erie_df = pd.read_csv(url)
erie_df['Date'] = pd.to_datetime(erie_df['Date'])
erie_df = erie_df[214:len(erie_df)]

# Populations and Infections
erie = 1400000
cases_erie = erie_df['Cases'].iloc[-1]
S_default = erie
known_infections = erie_df['Cases'].iloc[-1]
known_cases = erie_df['Admissions'].iloc[-1]
regional_hosp_share = 1.0
S = erie


# Widgets
current_hosp = st.sidebar.number_input(
    "Total Hospitalized Cases", value=known_cases, step=1.0, format="%f")

#doubling_time = st.sidebar.number_input(
#    "Doubling Time (days)", value=3.0, step=1.0, format="%f")
doubling_time = 3

start_date = st.sidebar.date_input(
    "Starting Date 1", datetime(2020,10,25))
start_day = 1

#relative_contact_rate = st.sidebar.number_input(
#    "Social distancing (% reduction in social contact) Unadjusted Model", 0, 100, value=0, step=5, format="%i")/100.0
relative_contact_rate = 0

decay1 = st.sidebar.number_input(
    "Social distancing 1 - Percent", 0, 100, value=5, step=5, format="%i")/100.0
p_m1 = (st.sidebar.number_input(
"Mask-wearing 1", 0.0, 100.0, value=7.0 ,step=5.0, format="%f")/100.0)


intervention1 = st.sidebar.date_input(
    "Date of change 2", datetime(2020,11,20))
int1_delta = (intervention1 - start_date).days
decay2 = st.sidebar.number_input(
    "Social distancing 2 - Percent", 0, 100, value=20, step=5, format="%i")/100.0
p_m2 = (st.sidebar.number_input(
"Mask-wearing 2", 0.0, 100.0, value=20.0 ,step=5.0, format="%f")/100.0)


intervention2 = st.sidebar.date_input(
    "Date of change 3", datetime(2020,11,26))
int2_delta = (intervention2 - start_date).days
decay3 = st.sidebar.number_input(
    "Social distancing 3 - Percent", 0, 100, value=20, step=5, format="%i")/100.0
p_m3 = (st.sidebar.number_input(
"Mask-wearing 3", 0.0, 100.0, value=30.0 ,step=5.0, format="%f")/100.0)


intervention3 = st.sidebar.date_input(
    "Date of change 4", datetime(2020,12,24))
int3_delta = (intervention3 - start_date).days
decay4 = st.sidebar.number_input(
    "Social distancing 4 - Percent", 0, 100, value=20, step=5, format="%i")/100.0
p_m4 = (st.sidebar.number_input(
"Mask-wearing 4", 0.0, 100.0, value=30.0 ,step=5.0, format="%f")/100.0)


intervention4 = st.sidebar.date_input(
    "Date of change 5", datetime(2021,1,6))
int4_delta = (intervention4 - start_date).days
decay5 = st.sidebar.number_input(
    "Social distancing 5 - Percent", 0, 100, value=20, step=5, format="%i")/100.0
p_m5 = (st.sidebar.number_input(
"Mask-wearing 5", 0.0, 100.0, value=30.0 ,step=5.0, format="%f")/100.0)
phi5 =  (st.sidebar.number_input(
"Vaccination Rate 5 (%)", 0.0, 100.0, value=0.3 ,step=0.5, format="%f")/100.0)


intervention5 = st.sidebar.date_input(
    "Date of change 6", datetime(2021,3,10))
int5_delta = (intervention5 - start_date).days
decay6 = st.sidebar.number_input(
    "Social distancing 6 - Percent", 0, 100, value=10, step=5, format="%i")/100.0
p_m6 = (st.sidebar.number_input(
"Mask-wearing 6", 0.0, 100.0, value=15.0 ,step=5.0, format="%f")/100.0)
phi6 = (st.sidebar.number_input(
"Vaccination Rate 6 (%)", 0.0, 100.0, value=0.3 ,step=0.5, format="%f")/100.0)
fracNS6 = (st.sidebar.number_input(
"Percent of Population with new strain 6 (%)", 0.0, 100.0, value=60.0 ,step=5.0, format="%f")/100.0)
new_strain6 = (st.sidebar.number_input(
"New Strain Increased Transmission w.r.t. Old Strain 6 (%)", 0.0, 1000.0, value=40.0 ,step=5.0, format="%f")/100.0)


intervention6 = st.sidebar.date_input(
    "Date of change 7", datetime(2021,4,10))
int6_delta = (intervention6 - start_date).days
decay7 = st.sidebar.number_input(
    "Social distancing 7 - Percent", 0, 100, value=20, step=5, format="%i")/100.0
p_m7 = (st.sidebar.number_input(
"Mask-wearing 7", 0.0, 100.0, value=30.0 ,step=5.0, format="%f")/100.0)
phi7 = (st.sidebar.number_input(
"Vaccination Rate 7 (%)", 0.0, 100.0, value=0.3 ,step=0.5, format="%f")/100.0)
fracNS7 = (st.sidebar.number_input(
"Percent of Population with new strain 7 (%)", 0.0, 100.0, value=60.0 ,step=5.0, format="%f")/100.0)
new_strain7 = (st.sidebar.number_input(
"New Strain Increased Transmission w.r.t. Old Strain 7 (%)", 0.0, 1000.0, value=40.0 ,step=5.0, format="%f")/100.0)


intervention7 = st.sidebar.date_input(
    "Date of change 8", datetime(2021,7,10))
int7_delta = (intervention7 - start_date).days
decay8 = st.sidebar.number_input(
    "Social distancing 8 - Percent", 0, 100, value=5, step=5, format="%i")/100.0
p_m8 = (st.sidebar.number_input(
"Mask-wearing 8", 0.0, 100.0, value=5.0 ,step=5.0, format="%f")/100.0)
phi8 = (st.sidebar.number_input(
"Vaccination Rate 8 (%)", 0.0, 100.0, value=0.1 ,step=0.5, format="%f")/100.0)
fracNS8 = (st.sidebar.number_input(
"Percent of Population with new strain 8 (%)", 0.0, 100.0, value=70.0, step=5.0, format="%f")/100.0)
new_strain8 = (st.sidebar.number_input(
"New Strain Increased Transmission w.r.t. Old Strain 8 (%)", 0.0, 1000.0, value=100.0 ,step=5.0, format="%f")/100.0)


intervention8 = st.sidebar.date_input(
    "Date of change 9", datetime(2021,8,15))
int8_delta = (intervention8 - start_date).days
decay9 = st.sidebar.number_input(
    "Social distancing 9 - Percent", 0, 100, value=20, step=5, format="%i")/100.0
p_m9 = (st.sidebar.number_input(
"Mask-wearing 9", 0.0, 100.0, value=30.0 ,step=5.0, format="%f")/100.0)
phi9 = (st.sidebar.number_input(
"Vaccination Rate 9 (%)", 0.0, 100.0, value=0.1 ,step=0.5, format="%f")/100.0)
fracNS9 = (st.sidebar.number_input(
"Percent of Population with new strain 9 (%)", 0.0, 100.0, value=90.0, step=5.0, format="%f")/100.0)
new_strain9 = (st.sidebar.number_input(
"New Strain Increased Transmission w.r.t. Old Strain 9 (%)", 0.0, 1000.0, value=100.0 ,step=5.0, format="%f")/100.0)

intervention9 = st.sidebar.date_input(
    "Date of change 9", datetime(2021,10,15))
int9_delta = (intervention9 - start_date).days
decay10 = st.sidebar.number_input(
    "Social distancing 10 - Percent", 0, 100, value=15, step=5, format="%i")/100.0
p_m10 = (st.sidebar.number_input(
"Mask-wearing 10", 0.0, 100.0, value=25.0 ,step=5.0, format="%f")/100.0)
phi10 = (st.sidebar.number_input(
"Vaccination Rate 10 (%)", 0.0, 100.0, value=0.1 ,step=0.5, format="%f")/100.0)
fracNS10 = (st.sidebar.number_input(
"Percent of Population with new strain 10 (%)", 0.0, 100.0, value=100.0, step=5.0, format="%f")/100.0)
new_strain10 = (st.sidebar.number_input(
"New Strain Increased Transmission w.r.t. Old Strain 10 (%)", 0.0, 1000.0, value=100.0 ,step=5.0, format="%f")/100.0)


hosp_rate = (
    st.sidebar.number_input("Hospitalization %", 0.0, 100.0, value=1.5, step=0.50, format="%f")/ 100.0)

icu_rate = (
    st.sidebar.number_input("ICU %", 0.0, 100.0, value=32.0, step=5.0, format="%f") / 100.0)

vent_rate = (
    st.sidebar.number_input("Ventilated %", 0.0, 100.0, value=35.0, step=5.0, format="%f")/ 100.0)

incubation_period =(
    st.sidebar.number_input("Incubation Period", 0.0, 12.0, value=3.1, step=0.1, format="%f"))

recovery_days =(
    st.sidebar.number_input("Recovery Period", 0.0, 21.0, value=11.0 ,step=0.1, format="%f"))

infectious_period =(
    st.sidebar.number_input("Infectious Period", 0.0, 18.0, value=3.0, step=0.1, format="%f"))

fatal = st.sidebar.number_input(
    "Overall Fatality (%)", 0.0, 100.0, value=0.5 ,step=0.1, format="%f")/100.0

fatal_hosp = st.sidebar.number_input(
    "Hospital Fatality (%)", 0.0, 100.0, value=9.9 ,step=0.1, format="%f")/100.0

#  death_days = st.sidebar.number_input(
#       "Days person remains in critical care or dies", 0, 20, value=4 ,step=1, format="%f")

# crit_lag = st.sidebar.number_input(
#       "Days person takes to go to critical care", 0, 20, value=4 ,step=1, format="%f")

hosp_lag = st.sidebar.number_input(
    "Days person remains in hospital or dies", 0, 20, value=4 ,step=1, format="%f")

asymptomatic = 1-(st.sidebar.number_input(
    "Asymptomatic (%)", 0.0, 100.0, value=32.2 ,step=0.1, format="%f")/100.0)

q = 1-(st.sidebar.number_input(
"Symptomatic Isolation Rate (contact tracing/quarantine when symptomatic)", 0.0, 100.0, value=34.8 ,step=0.1, format="%f")/100.0)



new_strain = (st.sidebar.number_input(
"New Strain Increased Transmission w.r.t. Old Strain (%)", 0.0, 1000.0, value=40.0 ,step=5.0, format="%f")/100.0)

phi = (st.sidebar.number_input(
"Vaccination Rate (%)", 0.0, 100.0, value=0.1 ,step=0.5, format="%f")/100.0)

fracNS = (st.sidebar.number_input(
"Percent of Population with new strain (%)", 0.0, 100.0, value=60.0 ,step=5.0, format="%f")/100.0)

# NYS 1/27/21: 8.1 doses per 100,000 population
# NYS 1/27/21: 7% of population w/ at least 1 shot
# NYS 1/27/21: 1.1% are fully vaccinated
# https://www.mathworks.com/matlabcentral/fileexchange/85103-covid-19-vaccination
# https://www.youtube.com/watch?v=Q6AI2nq3cPY

delta_p = 1/(st.sidebar.number_input(
"Days a person is pre-symptomatic", 0.0, 10.0, value=1.7 ,step=1.0, format="%f"))
hosp_los = st.sidebar.number_input("Hospital Length of Stay", value=8, step=1, format="%i")
icu_los = st.sidebar.number_input("ICU Length of Stay", value=11, step=1, format="%i")
vent_los = st.sidebar.number_input("Ventilator Length of Stay", value=10, step=1, format="%i")

# regional_hosp_share = (
# st.sidebar.number_input(
    # "Hospital Bed Share (%)", 0.0, 100.0, value=100.0, step=1.0, format="%f")
# / 100.0
# )

S = st.sidebar.number_input(
"Regional Population", value=S_default, step=100000, format="%i")

initial_infections = st.sidebar.number_input(
    "Currently Known Regional Infections (only used to compute detection rate - does not change projections)", value=known_infections, step=10.0, format="%f")

total_infections = current_hosp / regional_hosp_share / hosp_rate
detection_prob = initial_infections / total_infections


#S, I, R = S, initial_infections / detection_prob, 0

intrinsic_growth_rate = 2 ** (1 / doubling_time) - 1
# (0.12 + 0.07)/

recovered = 0.0

# mean recovery rate, gamma, (in 1/days).
gamma = 1 / recovery_days

# Contact rate, beta
beta = (intrinsic_growth_rate + gamma) / S * (1-relative_contact_rate)

r_t = beta / gamma * S # r_t is r_0 after distancing
r_naught = (intrinsic_growth_rate + gamma) / gamma
doubling_time_t = 1/np.log2(beta*S - gamma +1) # doubling time after distancing

# Contact rate,  beta for SEIR
beta2 = (
    intrinsic_growth_rate + (1/infectious_period)
) / S * (1-relative_contact_rate)
alpha = 1/incubation_period

# Contact rate,  beta for SEIR with phase adjusted R0
beta3 = (
(alpha+intrinsic_growth_rate)*(intrinsic_growth_rate + (1/infectious_period))
) / (alpha*S) *(1-relative_contact_rate)

## converting beta to intrinsic growth rate calculation
# https://www.sciencedirect.com/science/article/pii/S2468042719300491
beta4 = ((alpha+intrinsic_growth_rate)*(intrinsic_growth_rate + (1/infectious_period))) / (alpha*S)


# for SEIJRD
gamma_hosp = 1 / hosp_los
icu_days = 1 / icu_los

st.title("Great Lakes Healthcare COVID-19 Disease Model - Erie County, NY")


###################### First Graph ###################
# Erie cases Graph
erie_cases_bar = alt.Chart(erie_df).mark_bar(color='lightgray').encode(
    x='Date:T',
    y='Cases:Q',
    tooltip=[alt.Tooltip("Cases:Q", format=".0f", title="Cases")])
erie_admit_line = alt.Chart(erie_df).mark_line(color='red', point=True).encode(
    x='Date:T',
    y='Admissions:Q')
erie_icu_line = alt.Chart(erie_df).mark_line(color='orange', point=True).encode(
    x='Date:T',
    y='ICU:Q')

# New admissions in 24 hrs
#erie_admit24_line = alt.Chart(erie_df).mark_line(color='red').encode(x='Date:T',y='New_admits:Q',tooltip=[alt.Tooltip("value:Q", format=".0f"),"key:N"]).interactive()




# Slider and Date
n_days = st.slider("Number of days to project", 30, 500, 450, 1, "%i")
as_date = st.checkbox(label="Present result as dates", value=True)



#st.header("""Erie County: Reported Cases, Census and Admissions""")

# Erie Graph of Cases # Lines of cases
def erie_chart(
    projection_admits: pd.DataFrame) -> alt.Chart:
    """docstring"""

    projection_admits = projection_admits.rename(columns={"Admissions": "Census Inpatient",
                                                            "ICU":"Census Intensive",
                                                            "Ventilated":"Census Ventilated",
                                                            "New_admits":"New Admissions",
                                                            "New_discharge":"New Discharges",
                                                            })

    return(
        alt
        .Chart(projection_admits)
        .transform_fold(fold=["Census Inpatient",
                                "Census Intensive",
                                "Census Ventilated",
                                "New Admissions",
                                "New Discharges"
                                ])
        .mark_line(strokeWidth=3, point=False)
        .encode(
            x=alt.X("Date", title="Date"),
            y=alt.Y("value:Q", title="Erie County Census"),
            color="key:N",
            tooltip=[alt.Tooltip("value:Q", format=".0f"),"key:N"]
        )
        .interactive()
    )


#Erie Graph of Cases # Lines of cases # Inpatient Census
if as_date:
    #erie_df = add_date_column(erie_df)
    day_date = 'Date:T'
    def erie_inpatient(projection_admits: pd.DataFrame) -> alt.Chart:
        """docstring"""

        projection_admits = projection_admits.rename(columns={"Admissions": "Erie County Inpatient"})

        return(
            alt
            .Chart(projection_admits)
            .transform_fold(fold=["Erie County Inpatient"])
            .mark_line(strokeWidth=3, strokeDash=[2,3], point=True)
            .encode(
                x=alt.X(day_date),
                y=alt.Y("value:Q", title="Hospital Census"),
                #color="key:N",
                color=alt.value('steelblue'),
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
                y=alt.Y("value:Q", title="Hospital Census"),
                #color="key:N",
                color=alt.value('steelblue'),
                tooltip=[alt.Tooltip("value:Q", format=".0f"),"key:N"]
            )
            .interactive()
        )

# Erie Graph of Cases # Lines of cases # ICU Census
if as_date:
    #erie_df = add_date_column(erie_df)
    day_date = 'Date:T'
    def erie_icu(projection_admits: pd.DataFrame) -> alt.Chart:
        """docstring"""

        projection_admits = projection_admits.rename(columns={"ICU": "Erie County ICU"})

        return(
            alt
            .Chart(projection_admits)
            .transform_fold(fold=["Erie County ICU"])
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
    def erie_icu(projection_admits: pd.DataFrame) -> alt.Chart:
        """docstring"""

        projection_admits = projection_admits.rename(columns={"ICU": "Erie County ICU"})

        return(
            alt
            .Chart(projection_admits)
            .transform_fold(fold=["Erie County ICU"])
            .mark_line(strokeWidth=3, strokeDash=[2,3], point=True)
            .encode(
                x=alt.X(day_date),
                y=alt.Y("value:Q", title="Census"),
                color="key:N",
                tooltip=[alt.Tooltip("value:Q", format=".0f"),"key:N"]
            )
            .interactive()
        )

# Erie Graph of Cases # Lines of cases # Ventilator Census
if as_date:
    #erie_df = add_date_column(erie_df)
    day_date = 'Date:T'
    def erie_vent(projection_admits: pd.DataFrame) -> alt.Chart:
        """docstring"""

        projection_admits = projection_admits.rename(columns={"Ventilated": "Erie County Ventilated"})

        return(
            alt
            .Chart(projection_admits)
            .transform_fold(fold=["Erie County Ventilated"])
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
    def erie_vent(projection_admits: pd.DataFrame) -> alt.Chart:
        """docstring"""

        projection_admits = projection_admits.rename(columns={"Ventilated": "Erie County Ventilated"})

        return(
            alt
            .Chart(projection_admits)
            .transform_fold(fold=["Erie County Ventilated"])
            .mark_line(strokeWidth=3, strokeDash=[2,3], point=True)
            .encode(
                x=alt.X(day_date),
                y=alt.Y("value:Q", title="Census"),
                color="key:N",
                tooltip=[alt.Tooltip("value:Q", format=".0f"),"key:N"]
            )
            .interactive()
        )



#########


#Erie Graph of Cases # Lines of cases # Inpatient Census
if as_date:
    #erie_df = add_date_column(erie_df)
    day_date = 'Date:T'
    def erie_admit(projection_admits: pd.DataFrame) -> alt.Chart:
        """docstring"""

        projection_admits = projection_admits.rename(columns={"Admits_24h": "Admissions 24 hr"})

        return(
            alt
            .Chart(projection_admits)
            .transform_fold(fold=["Admissions 24 hr"])
            .mark_line(strokeWidth=1, point=True)
            .encode(
                x=alt.X(day_date),
                y=alt.Y("value:Q", title="Daily admissions"),
                #color="key:N",
                color=alt.value('red'),
                tooltip=[alt.Tooltip("value:Q", format=".0f"),"key:N"]
            )
            .interactive()
        )
else:
    day_date = 'day'
    def erie_admit(
        projection_admits: pd.DataFrame) -> alt.Chart:
        """docstring"""

        projection_admits = projection_admits.rename(columns={"Admits_24h": "Admissions 24 hr"})

        return(
            alt
            .Chart(projection_admits)
            .transform_fold(fold=["Admissions 24 hr"])
            .mark_line(strokeWidth=1, point=True)
            .encode(
                x=alt.X(day_date),
                y=alt.Y("value:Q", title="Daily admissions"),
                #color="key:N",
                color=alt.value('red'),
                tooltip=[alt.Tooltip("value:Q", format=".0f"),"key:N"]
            )
            .interactive()
        )


############





erie_lines = erie_chart(erie_df)
erie_lines_ip = erie_inpatient(erie_df)
erie_lines_icu = erie_icu(erie_df)
erie_lines_vent = erie_vent(erie_df)
erie_admit24_line = erie_admit(erie_df)

# Bar chart of Erie cases with layer of HERDS DAta Erie
#st.altair_chart(erie_cases_bar + erie_lines, use_container_width=True)

beta_decay = 0.0

RateLos = namedtuple("RateLos", ("rate", "length_of_stay"))
hospitalized=RateLos(hosp_rate, hosp_los)
icu=RateLos(icu_rate, icu_los)
ventilated=RateLos(vent_rate, vent_los)


rates = tuple(each.rate for each in (hospitalized, icu, ventilated))
lengths_of_stay = tuple(each.length_of_stay for each in (hospitalized, icu, ventilated))


# #############
### SIR model
s_v, i_v, r_v = sim_sir(S-2, 1, 1 ,beta, gamma, n_days)
susceptible_v, infected_v, recovered_v = s_v, i_v, r_v

i_hospitalized_v, i_icu_v, i_ventilated_v = get_dispositions(i_v, rates, regional_hosp_share)

r_hospitalized_v, r_icu_v, r_ventilated_v = get_dispositions(r_v, rates, regional_hosp_share)

dispositions = (
            i_hospitalized_v + r_hospitalized_v,
            i_icu_v + r_icu_v,
            i_ventilated_v + r_ventilated_v)

hospitalized_v, icu_v, ventilated_v = (
            i_hospitalized_v,
            i_icu_v,
            i_ventilated_v)


##############
### SEIR model
gamma2=1/infectious_period
exposed2=beta4*S*total_infections
S2=S-exposed2-total_infections

s_e, e_e, i_e, r_e = sim_seir(S-2, 1 ,1, recovered, beta3, gamma2, alpha, n_days)

susceptible_e, exposed_e, infected_e, recovered_e = s_e, e_e, i_e, r_e

i_hospitalized_e, i_icu_e, i_ventilated_e = get_dispositions(i_e, rates, regional_hosp_share)

r_hospitalized_e, r_icu_e, r_ventilated_e = get_dispositions(r_e, rates, regional_hosp_share)

dispositions_e = (
            i_hospitalized_e + r_hospitalized_e,
            i_icu_e + r_icu_e,
            i_ventilated_e + r_ventilated_e)

hospitalized_e, icu_e, ventilated_e = (
            i_hospitalized_e,
            i_icu_e,
            i_ventilated_e)



##################################################################
## SEIR model with phase adjusted R_0 and Disease Related Fatality,
## Asymptomatic, Hospitalization, Presymptomatic, and masks
# Main Curve 1/5/20
E0=667
A0=195
I0=393
D0=322
R0=111725
J0=22
P0=357
x=0.5
#S0=S-E0-P0-A0-I0-D0-J0-R0
S0=1286318.1612
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
                                                      decay1, decay2, decay3, decay4, decay5, decay6, decay7, decay8, decay9, decay10, start_day, int1_delta, int2_delta, int3_delta, int4_delta, int5_delta, int6_delta, int7_delta, int8_delta, int9_delta,
                                                      fatal_hosp, asymptomatic, hosp_rate, q,  l, x,
                                                      p_m1, p_m2, p_m3, p_m4, p_m5, p_m6, p_m7, p_m8, p_m9, p_m10, delta_p)

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
#st.dataframe(i_hospitalized_P)
#st.dataframe(J_p)
r_hospitalized_P, r_icu_P, r_ventilated_P = get_dispositions(RH_p, rates_p, regional_hosp_share)
d_hospitalized_P, d_icu_P, d_ventilated_P = get_dispositions(D_p, rates_p, regional_hosp_share)
dispositions_P0 = (
            i_hospitalized_P + r_hospitalized_P+ d_hospitalized_P ,
            i_icu_P+r_icu_P+d_icu_P,
            i_ventilated_P+r_ventilated_P +d_ventilated_P)
#st.dataframe(i_hospitalized_P + r_hospitalized_P+ d_hospitalized_P)
#st.dataframe(dispositions_P0)
hospitalized_P0, icu_P0, ventilated_P0 = (
            i_hospitalized_P,
            i_icu_P,
            i_ventilated_P)



##################################################################
## SEIR model with phase adjusted R_0 and Disease Related Fatality,
## Asymptomatic, Hospitalization, Presymptomatic, and masks
# Higher Value of Intervention 12/1/20
E0=667
A0=195
I0=393
D0=322
R0=111725
J0=22
P0=357
x=0.5
#S0=S-E0-P0-A0-I0-D0-J0-R0
S0=1286318.1612
beta_j=0.06
q=0.583
l=0.717
gamma_hosp=1/hosp_lag
AAA=beta4*(1/gamma2)*S
beta_j=AAA*(1/(((1-asymptomatic)*1/gamma2)+(asymptomatic*q/(gamma2+hosp_rate))+(asymptomatic*hosp_rate*l/((gamma2+hosp_rate)*gamma_hosp))))

R0_n=beta_j* (((1-asymptomatic)*1/gamma2)+(asymptomatic*q/(gamma2+hosp_rate))+(asymptomatic*hosp_rate*l/((gamma2+hosp_rate)*gamma_hosp)))
beta_j=0.51
R0_n=beta_j* (((1-asymptomatic)*1/gamma2)+(asymptomatic*q/(gamma2+hosp_rate))+(asymptomatic*hosp_rate*l/((gamma2+hosp_rate)*gamma_hosp)))
S_p, E_p,P_p,A_p, I_p,J_p, R_p, D_p, RH_p=sim_sepaijrd_decay_ode(S0, E0, P0,A0,I0,J0, R0, D0, beta_j,gamma2, gamma_hosp, alpha, n_days,
                                                      decay1, decay2, decay3, decay4, decay5, decay6, decay7, decay8, decay9, decay10, start_day, int1_delta, int2_delta, int3_delta, int4_delta, int5_delta, int6_delta, int7_delta, int8_delta, int9_delta,
                                                      fatal_hosp, asymptomatic, hosp_rate, q,  l, x,
                                                      p_m1, p_m2, p_m3, p_m4, p_m5, p_m6, p_m7, p_m8, p_m9, p_m10, delta_p)

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
dispositions_P1 = (
            i_hospitalized_P + r_hospitalized_P+ d_hospitalized_P ,
            i_icu_P+r_icu_P+d_icu_P,
            i_ventilated_P+r_ventilated_P +d_ventilated_P)

hospitalized_P1, icu_P1, ventilated_P1 = (
            i_hospitalized_P,
            i_icu_P,
            i_ventilated_P)

##################################################################
## SEIR model with phase adjusted R_0 and Disease Related Fatality,
## Asymptomatic, Hospitalization, Presymptomatic, and masks
# Lower value of intervention 12/1/20
E0=667
A0=195
I0=393
D0=322
R0=111725
J0=22
P0=357
x=0.5
#S0=S-E0-P0-A0-I0-D0-J0-R0
S0=1286318.1612
beta_j=0.6
q=0.583
l=0.717
#p_m5 = p_m5*(1-new_strain)
#decay5 = decay5*(1-new_strain)
gamma_hosp=1/hosp_lag
AAA=beta4*(1/gamma2)*S
beta_j=AAA*(1/(((1-asymptomatic)*1/gamma2)+(asymptomatic*q/(gamma2+hosp_rate))+(asymptomatic*hosp_rate*l/((gamma2+hosp_rate)*gamma_hosp))))

R0_n=beta_j* (((1-asymptomatic)*1/gamma2)+(asymptomatic*q/(gamma2+hosp_rate))+(asymptomatic*hosp_rate*l/((gamma2+hosp_rate)*gamma_hosp)))
beta_j=0.51
R0_n=beta_j* (((1-asymptomatic)*1/gamma2)+(asymptomatic*q/(gamma2+hosp_rate))+(asymptomatic*hosp_rate*l/((gamma2+hosp_rate)*gamma_hosp)))
S_p, E_p,P_p,A_p, I_p,J_p, R_p, D_p, RH_p=sim_sepaijrd_decay_ode(S0, E0, P0,A0,I0,J0, R0, D0, beta_j,gamma2, gamma_hosp, alpha, n_days,
                                                      decay1, decay2, decay3, decay4, decay5, decay6, decay7, decay8, decay9, decay10, start_day, int1_delta, int2_delta, int3_delta, int4_delta, int5_delta, int6_delta, int7_delta, int8_delta, int9_delta,
                                                      fatal_hosp, asymptomatic, hosp_rate, q,  l, x,
                                                      p_m1, p_m2, p_m3, p_m4, p_m5, p_m6, p_m7, p_m8, p_m9, p_m10, delta_p)

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
#st.dataframe(i_hospitalized_P)
r_hospitalized_P, r_icu_P, r_ventilated_P = get_dispositions(RH_p, rates_p, regional_hosp_share)
d_hospitalized_P, d_icu_P, d_ventilated_P = get_dispositions(D_p, rates_p, regional_hosp_share)
dispositions_P2 = (
            i_hospitalized_P + r_hospitalized_P+ d_hospitalized_P ,
            i_icu_P+r_icu_P+d_icu_P,
            i_ventilated_P+r_ventilated_P +d_ventilated_P)

hospitalized_P2, icu_P2, ventilated_P2 = (
            i_hospitalized_P,
            i_icu_P,
            i_ventilated_P)


# V_Curve
##################################################################
## SEIR model with phase adjusted R_0 and Disease Related Fatality,
## Asymptomatic, Hospitalization, Presymptomatic, and masks
# Higher Facemask Use
# Not in use as of 11/16/20
E0=100
A0=100
I0=100
D0=0
R0=0
J0=0
P0=220
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
                                                      decay1, decay2, decay3, decay4, decay5, decay6, decay7, decay8, decay9, decay10, start_day, int1_delta, int2_delta, int3_delta, int4_delta, int5_delta, int6_delta, int7_delta, int8_delta, int9_delta,
                                                      fatal_hosp, asymptomatic, hosp_rate, q,  l, x,
                                                      p_m1, p_m2, p_m3, p_m4, p_m5, p_m6, p_m7, p_m8, p_m9, p_m10, delta_p)

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
dispositions_P3 = (
            i_hospitalized_P + r_hospitalized_P+ d_hospitalized_P ,
            i_icu_P+r_icu_P+d_icu_P,
            i_ventilated_P+r_ventilated_P +d_ventilated_P)

hospitalized_P3, icu_P3, ventilated_P3 = (
            i_hospitalized_P,
            i_icu_P,
            i_ventilated_P)


##################################################################
## SEIR model with phase adjusted R_0 and Disease Related Fatality,
## Asymptomatic, Hospitalization, Presymptomatic, and masks
# Vaccination Curve - 1 - 2/3/20
E0=667
V0=0
A0=195
I0=393
D0=322
R0=111725
J0=22
P0=357
x=0.5
S0=1286318.1612
q=0.583
l=0.717
sigma=(1-0.8) #10% of those vaccinated are still getting infected
# phi = 0.003 #how many susceptible people are fully vaccinated each day
# fracNS = 0.0
# p_m6=0.3
# decay6=0.2
gamma_hosp=1/hosp_lag
AAA=beta4*(1/gamma2)*S
beta_j=AAA*(1/(((1-asymptomatic)*1/gamma2)+(asymptomatic*q/(gamma2+hosp_rate))+(asymptomatic*hosp_rate*l/((gamma2+hosp_rate)*gamma_hosp))))

R0_n=beta_j* (((1-asymptomatic)*1/gamma2)+(asymptomatic*q/(gamma2+hosp_rate))+(asymptomatic*hosp_rate*l/((gamma2+hosp_rate)*gamma_hosp)))
beta_j=0.51
R0_n=beta_j* (((1-asymptomatic)*1/gamma2)+(asymptomatic*q/(gamma2+hosp_rate))+(asymptomatic*hosp_rate*l/((gamma2+hosp_rate)*gamma_hosp)))

######

S_v, V_v,E_v,P_v,A_v, I_v,J_v, R_v, D_v, RH_v=sim_svepaijrd_decay_ode(S0, V0,E0, P0,A0,I0,J0, R0, D0, beta_j,gamma2, gamma_hosp, alpha, n_days,
                                                      decay1, decay2, decay3, decay4, decay5, decay6, decay7, decay8, decay9, decay10, start_day, int1_delta, int2_delta, int3_delta, int4_delta, int5_delta, int6_delta, int7_delta, int8_delta, int9_delta,
                                                      fatal_hosp, asymptomatic, hosp_rate, q,  l, x,
                                                      p_m1, p_m2, p_m3, p_m4, p_m5, p_m6, p_m7, p_m8, p_m9, p_m10, delta_p, sigma, phi)

icu_curve= J_v*icu_rate
vent_curve=J_v*vent_rate

hosp_rate_v=1.0
RateLos = namedtuple("RateLos", ("rate", "length_of_stay"))
hospitalized_v=RateLos(hosp_rate_v, hosp_los)
icu_rate_v= icu_rate
vent_rate_v= vent_rate
icu=RateLos(icu_rate_v, icu_los)
ventilated=RateLos(vent_rate_v, vent_los)


rates_v = tuple(each.rate for each in (hospitalized_v, icu, ventilated))
lengths_of_stay = tuple(each.length_of_stay for each in (hospitalized_v, icu, ventilated))


i_hospitalized_V, i_icu_V, i_ventilated_V= get_dispositions(J_v, rates_v, regional_hosp_share)
r_hospitalized_V, r_icu_V, r_ventilated_V = get_dispositions(RH_v, rates_v, regional_hosp_share)
d_hospitalized_V, d_icu_V, d_ventilated_V = get_dispositions(D_v, rates_v, regional_hosp_share)
dispositions_V0 = (
            i_hospitalized_V + r_hospitalized_V+ d_hospitalized_V,
            i_icu_V+r_icu_V+d_icu_V,
            i_ventilated_V+r_ventilated_V +d_ventilated_V)
hospitalized_V0, icu_V0, ventilated_V0 = (
            i_hospitalized_V,
            i_icu_V,
            i_ventilated_V)


##################################################################
## SEIR model with phase adjusted R_0 and Disease Related Fatality,
## Asymptomatic, Hospitalization, Presymptomatic, and masks
# Vaccination Curve 2/3/20
# Extra curve 1 -
E0=667
V0=0
A0=195
I0=393
D0=322
R0=111725
J0=22
P0=357
x=0.5
S0=1286318.1612
q=0.583
l=0.717
sigma=(1-0.8) #30% of those vaccinated are still getting infected
# phi = 0.003 #how many susceptible people are fully vaccinated each day
# fracNS = 0.0 # Fraction of the population with new strain.
# p_m6=0.15
# decay6=0.1
gamma_hosp=1/hosp_lag
AAA=beta4*(1/gamma2)*S
beta_j=AAA*(1/(((1-asymptomatic)*1/gamma2)+(asymptomatic*q/(gamma2+hosp_rate))+(asymptomatic*hosp_rate*l/((gamma2+hosp_rate)*gamma_hosp))))

R0_n=beta_j* (((1-asymptomatic)*1/gamma2)+(asymptomatic*q/(gamma2+hosp_rate))+(asymptomatic*hosp_rate*l/((gamma2+hosp_rate)*gamma_hosp)))
beta_j=0.51
R0_n=beta_j* (((1-asymptomatic)*1/gamma2)+(asymptomatic*q/(gamma2+hosp_rate))+(asymptomatic*hosp_rate*l/((gamma2+hosp_rate)*gamma_hosp)))

######

S_v, V_v,E_v,P_v,A_v, I_v,J_v, R_v, D_v, RH_v=sim_svepaijrd_decay_ode(S0, V0,E0, P0,A0,I0,J0, R0, D0, beta_j,gamma2, gamma_hosp, alpha, n_days,
                                                      decay1, decay2, decay3, decay4, decay5, decay6, decay7, decay8, decay9, decay10, start_day, int1_delta, int2_delta, int3_delta, int4_delta, int5_delta, int6_delta, int7_delta, int8_delta, int9_delta,
                                                      fatal_hosp, asymptomatic, hosp_rate, q,  l, x,
                                                      p_m1, p_m2, p_m3, p_m4, p_m5, p_m6, p_m7, p_m8, p_m9, p_m10, delta_p, sigma, phi)

icu_curve= J_v*icu_rate
vent_curve=J_v*vent_rate

hosp_rate_v=1.0
RateLos = namedtuple("RateLos", ("rate", "length_of_stay"))
hospitalized_v=RateLos(hosp_rate_v, hosp_los)
icu_rate_v= icu_rate
vent_rate_v= vent_rate
icu=RateLos(icu_rate_v, icu_los)
ventilated=RateLos(vent_rate_v, vent_los)


rates_v = tuple(each.rate for each in (hospitalized_v, icu, ventilated))
lengths_of_stay = tuple(each.length_of_stay for each in (hospitalized_v, icu, ventilated))


i_hospitalized_V, i_icu_V, i_ventilated_V= get_dispositions(J_v, rates_v, regional_hosp_share)
r_hospitalized_V, r_icu_V, r_ventilated_V = get_dispositions(RH_v, rates_v, regional_hosp_share)
d_hospitalized_V, d_icu_V, d_ventilated_V = get_dispositions(D_v, rates_v, regional_hosp_share)
dispositions_V1 = (
            i_hospitalized_V + r_hospitalized_V+ d_hospitalized_V,
            i_icu_V+r_icu_V+d_icu_V,
            i_ventilated_V+r_ventilated_V +d_ventilated_V)
hospitalized_V1, icu_V1, ventilated_V1 = (
            i_hospitalized_V,
            i_icu_V,
            i_ventilated_V)


##################################################################
## SEIR model with phase adjusted R_0 and Disease Related Fatality,
## Asymptomatic, Hospitalization, Presymptomatic, and masks
# Vaccination Curve 2/3/20
# Extra curve 2 - 2/15/21
E0=667
V0=0
A0=195
I0=393
D0=322
R0=111725
J0=22
P0=357
x=0.5
S0=1286318.1612
q=0.583
l=0.717
sigma=(1-0.8) #10% of those vaccinated are still getting infected
# phi = 0.003 #how many susceptible people are fully vaccinated each day
# fracNS = 0.0
# p_m6=0.01
# decay6=0.01
gamma_hosp=1/hosp_lag
AAA=beta4*(1/gamma2)*S
beta_j=AAA*(1/(((1-asymptomatic)*1/gamma2)+(asymptomatic*q/(gamma2+hosp_rate))+(asymptomatic*hosp_rate*l/((gamma2+hosp_rate)*gamma_hosp))))

R0_n=beta_j* (((1-asymptomatic)*1/gamma2)+(asymptomatic*q/(gamma2+hosp_rate))+(asymptomatic*hosp_rate*l/((gamma2+hosp_rate)*gamma_hosp)))
beta_j=0.51
R0_n=beta_j* (((1-asymptomatic)*1/gamma2)+(asymptomatic*q/(gamma2+hosp_rate))+(asymptomatic*hosp_rate*l/((gamma2+hosp_rate)*gamma_hosp)))

######

S_v, V_v,E_v,P_v,A_v, I_v,J_v, R_v, D_v, RH_v=sim_svepaijrd_decay_ode(S0, V0,E0, P0,A0,I0,J0, R0, D0, beta_j,gamma2, gamma_hosp, alpha, n_days,
                                                      decay1, decay2, decay3, decay4, decay5, decay6, decay7, decay8, decay9, decay10, start_day, int1_delta, int2_delta, int3_delta, int4_delta, int5_delta, int6_delta, int7_delta, int8_delta, int9_delta,
                                                      fatal_hosp, asymptomatic, hosp_rate, q,  l, x,
                                                      p_m1, p_m2, p_m3, p_m4, p_m5, p_m6, p_m7, p_m8, p_m9, p_m10, delta_p, sigma, phi)

icu_curve= J_v*icu_rate
vent_curve=J_v*vent_rate

hosp_rate_v=1.0
RateLos = namedtuple("RateLos", ("rate", "length_of_stay"))
hospitalized_v=RateLos(hosp_rate_v, hosp_los)
icu_rate_v= icu_rate
vent_rate_v= vent_rate
icu=RateLos(icu_rate_v, icu_los)
ventilated=RateLos(vent_rate_v, vent_los)


rates_v = tuple(each.rate for each in (hospitalized_v, icu, ventilated))
lengths_of_stay = tuple(each.length_of_stay for each in (hospitalized_v, icu, ventilated))


i_hospitalized_V, i_icu_V, i_ventilated_V= get_dispositions(J_v, rates_v, regional_hosp_share)
r_hospitalized_V, r_icu_V, r_ventilated_V = get_dispositions(RH_v, rates_v, regional_hosp_share)
d_hospitalized_V, d_icu_V, d_ventilated_V = get_dispositions(D_v, rates_v, regional_hosp_share)
dispositions_V2 = (
            i_hospitalized_V + r_hospitalized_V+ d_hospitalized_V,
            i_icu_V+r_icu_V+d_icu_V,
            i_ventilated_V+r_ventilated_V +d_ventilated_V)

hospitalized_V2, icu_V2, ventilated_V2 = (
            i_hospitalized_V,
            i_icu_V,
            i_ventilated_V)







# S_Curve
##################################################################
## SEIR model with phase adjusted R_0 and Disease Related Fatality,
## Asymptomatic, Hospitalization, Presymptomatic, and masks
# Vaccination Curve + New strain 2/3/20
# Last modified - Gabe - 8/5/21
# Last modified - Gabe - 9/22/21
E0=667
V0=0
A0=195
I0=393
D0=322
R0=111725
J0=22
P0=357
x=0.5
S0=1286318.1612
q=0.583
l=0.717
sigma=(1-0.8) #10% of those vaccinated are still getting infected
#phi = 0.003 #how many susceptible people are fully vaccinated each day
#fracNS = 0.60
#p_m6=0.15
#decay6=0.10
gamma_hosp=1/hosp_lag
AAA=beta4*(1/gamma2)*S
beta_j=AAA*(1/(((1-asymptomatic)*1/gamma2)+(asymptomatic*q/(gamma2+hosp_rate))+(asymptomatic*hosp_rate*l/((gamma2+hosp_rate)*gamma_hosp))))

R0_n=beta_j* (((1-asymptomatic)*1/gamma2)+(asymptomatic*q/(gamma2+hosp_rate))+(asymptomatic*hosp_rate*l/((gamma2+hosp_rate)*gamma_hosp)))
beta_j=0.51
R0_n=beta_j* (((1-asymptomatic)*1/gamma2)+(asymptomatic*q/(gamma2+hosp_rate))+(asymptomatic*hosp_rate*l/((gamma2+hosp_rate)*gamma_hosp)))

######

S_vNS, V_v, NSE_vNS, P_vNS, A_vNS, I_vNS, J_vNS, R_vNS, D_vNS, RH_vNS=sim_svepaijrdNS_decay_ode(S0, V0,E0, P0,A0,I0,J0, R0, D0, beta_j,gamma2, gamma_hosp, alpha, n_days,
                                                      decay1, decay2, decay3, decay4, decay5, decay6, decay7, decay8, decay9, decay10, start_day, int1_delta, int2_delta, int3_delta, int4_delta, int5_delta, int6_delta, int7_delta, int8_delta, int9_delta,
                                                      fatal_hosp, asymptomatic, hosp_rate, q,  l, x,
                                                      p_m1, p_m2, p_m3, p_m4, p_m5, p_m6, p_m7, p_m8, p_m9, p_m10, delta_p, sigma, phi5, phi6, phi7, phi8, phi9, phi10, new_strain6, new_strain7, new_strain8, new_strain9, new_strain10, fracNS6, fracNS7, fracNS8, fracNS9, fracNS10)

icu_curve= J_vNS*icu_rate
vent_curve=J_vNS*vent_rate

hosp_rate_vNS=1.0
RateLos = namedtuple("RateLos", ("rate", "length_of_stay"))
hospitalized_vNS=RateLos(hosp_rate_vNS, hosp_los)
icu_rate_vNS= icu_rate
vent_rate_vNS= vent_rate
icu=RateLos(icu_rate_vNS, icu_los)
ventilated=RateLos(vent_rate_vNS, vent_los)


rates_vNS = tuple(each.rate for each in (hospitalized_vNS, icu, ventilated))
lengths_of_stay = tuple(each.length_of_stay for each in (hospitalized_vNS, icu, ventilated))


i_hospitalized_VNS, i_icu_VNS, i_ventilated_VNS= get_dispositions(J_vNS, rates_vNS, regional_hosp_share)
r_hospitalized_VNS, r_icu_VNS, r_ventilated_VNS = get_dispositions(RH_vNS, rates_vNS, regional_hosp_share)
d_hospitalized_VNS, d_icu_VNS, d_ventilated_VNS = get_dispositions(D_vNS, rates_vNS, regional_hosp_share)
dispositions_VNS0 = (
            i_hospitalized_VNS + r_hospitalized_VNS+ d_hospitalized_VNS,
            i_icu_VNS+r_icu_VNS+d_icu_VNS,
            i_ventilated_VNS+r_ventilated_VNS +d_ventilated_VNS)
hospitalized_VNS0, icu_VNS0, ventilated_VNS0 = (
            i_hospitalized_VNS,
            i_icu_VNS,
            i_ventilated_VNS)



# Create incidence/prevalence table for main model as other models had different parameters and this uses same A_vNS etc variables.
# 8/22/21 - Gabe

### Incidence
dispositions_inc= (P_vNS)
dispositions_inc= pd.DataFrame(dispositions_inc, columns=['newcases'])
dispositions_inc2 = dispositions_inc.iloc[:-1, :] - dispositions_inc.shift(1)
dispositions_inc2['newcases'] = np.where(dispositions_inc2 <0, dispositions_inc2*(-1.0), dispositions_inc2)
dispositions_inc2["day"] = range(dispositions_inc2.shape[0])
dispositions_inc2["TotalCases"]=S_vNS
dispositions_inc2.at[0,'newcases']=21
dispositions_inc2["incidencerate"]=(dispositions_inc2['newcases']/dispositions_inc2['TotalCases'])*100000
## total number of new cases daily/total number of people disease free at the start of the day

#st.write(dispositions_inc)
#st.write(dispositions_inc2)

## Prevalence
dispositions_prev=(P_vNS + A_vNS + I_vNS + J_vNS)
dispositions_prev=pd.DataFrame(dispositions_prev, columns=['Total Prevalent Cases'])
dispositions_prev["day"] = range(dispositions_prev.shape[0])
dispositions_prev["Total Cases"]=S_vNS
#dispositions_prev["Population"]=1400000.0
dispositions_prev["pointprevalencerate"]=(dispositions_prev['Total Prevalent Cases']/dispositions_prev['Total Cases'])*100000
#total number of infected people during that day/ total number in population

##################################################################
## SEIR model with phase adjusted R_0 and Disease Related Fatality,
## Asymptomatic, Hospitalization, Presymptomatic, and masks
# Vaccination Curve + New strain 2/3/20
# Last Modified - Gabe - 8/5/21 - Added extra curve with/without delta variant
E0=667
V0=0
A0=195
I0=393
D0=322
R0=111725
J0=22
P0=357
x=0.5
S0=1286318.1612
q=0.583
l=0.717
#sigma=(1-0.8) #30% of those vaccinated are still getting infected
#phi = 0.0015 #how many susceptible people are fully vaccinated each day
#fracNS = 0.60
#p_m6=0.15
#decay6=0.10
#p_m7=0.15
#decay7=0.10
#fracNS8 = 1
#new_strain8 = 0.8
p_m9 = 0.25
decay9 = 0.15
fracNS9 = 0.9
new_strain9 = 1
phi9 = 0.001
gamma_hosp=1/hosp_lag
AAA=beta4*(1/gamma2)*S
beta_j=AAA*(1/(((1-asymptomatic)*1/gamma2)+(asymptomatic*q/(gamma2+hosp_rate))+(asymptomatic*hosp_rate*l/((gamma2+hosp_rate)*gamma_hosp))))

R0_n=beta_j* (((1-asymptomatic)*1/gamma2)+(asymptomatic*q/(gamma2+hosp_rate))+(asymptomatic*hosp_rate*l/((gamma2+hosp_rate)*gamma_hosp)))
beta_j=0.51
R0_n=beta_j* (((1-asymptomatic)*1/gamma2)+(asymptomatic*q/(gamma2+hosp_rate))+(asymptomatic*hosp_rate*l/((gamma2+hosp_rate)*gamma_hosp)))

######

S_vNS, V_v,NSE_vNS,P_vNS,A_vNS, I_vNS,J_vNS, R_vNS, D_vNS, RH_vNS=sim_svepaijrdNS_decay_ode(S0, V0,E0, P0,A0,I0,J0, R0, D0, beta_j,gamma2, gamma_hosp, alpha, n_days,
                                                      decay1, decay2, decay3, decay4, decay5, decay6, decay7, decay8, decay9, decay10, start_day, int1_delta, int2_delta, int3_delta, int4_delta, int5_delta, int6_delta, int7_delta, int8_delta, int9_delta,
                                                      fatal_hosp, asymptomatic, hosp_rate, q,  l, x,
                                                      p_m1, p_m2, p_m3, p_m4, p_m5, p_m6, p_m7, p_m8, p_m9, p_m10, delta_p, sigma, phi5, phi6, phi7, phi8, phi9, phi10, new_strain6, new_strain7, new_strain8, new_strain9, new_strain10, fracNS6, fracNS7, fracNS8, fracNS9, fracNS10)

icu_curve= J_vNS*icu_rate
vent_curve=J_vNS*vent_rate

hosp_rate_vNS=1.0
RateLos = namedtuple("RateLos", ("rate", "length_of_stay"))
hospitalized_vNS=RateLos(hosp_rate_vNS, hosp_los)
icu_rate_vNS= icu_rate
vent_rate_vNS= vent_rate
icu=RateLos(icu_rate_vNS, icu_los)
ventilated=RateLos(vent_rate_vNS, vent_los)


rates_vNS = tuple(each.rate for each in (hospitalized_vNS, icu, ventilated))
lengths_of_stay = tuple(each.length_of_stay for each in (hospitalized_vNS, icu, ventilated))


i_hospitalized_VNS, i_icu_VNS, i_ventilated_VNS= get_dispositions(J_vNS, rates_vNS, regional_hosp_share)
r_hospitalized_VNS, r_icu_VNS, r_ventilated_VNS = get_dispositions(RH_vNS, rates_vNS, regional_hosp_share)
d_hospitalized_VNS, d_icu_VNS, d_ventilated_VNS = get_dispositions(D_vNS, rates_vNS, regional_hosp_share)
dispositions_VNS1 = (
            i_hospitalized_VNS + r_hospitalized_VNS+ d_hospitalized_VNS,
            i_icu_VNS+r_icu_VNS+d_icu_VNS,
            i_ventilated_VNS+r_ventilated_VNS +d_ventilated_VNS)
hospitalized_VNS1, icu_VNS1, ventilated_VNS1 = (
            i_hospitalized_VNS,
            i_icu_VNS,
            i_ventilated_VNS)


##################################################################
## SEIR model with phase adjusted R_0 and Disease Related Fatality,
## Asymptomatic, Hospitalization, Presymptomatic, and masks
# Vaccination Curve + New strain
# Modification - Add population and transmission to effectiveness of vaccine. 2/15/21
E0=667
V0=0
A0=195
I0=393
D0=322
R0=111725
J0=22
P0=357
x=0.5
S0=1286318.1612
q=0.583
l=0.717
#sigma=(1-0.8) #50% of those vaccinated are still getting infected
#phi = 0.000 #how many susceptible people are fully vaccinated each day
#fracNS = 0.60
#p_m6=0.15
#decay6=0.1
#p_m7=0.3
#decay7=0.2
#fracNS8 = 1
#fracNS9 = 1
#new_strain8 = 0.5
#new_strain9 = 0.5
p_m9 = 0.15
decay9 = 0.1
fracNS9 = 1
new_strain9 = 0.9
phi9 = 0.003
gamma_hosp=1/hosp_lag
AAA=beta4*(1/gamma2)*S
beta_j=AAA*(1/(((1-asymptomatic)*1/gamma2)+(asymptomatic*q/(gamma2+hosp_rate))+(asymptomatic*hosp_rate*l/((gamma2+hosp_rate)*gamma_hosp))))

R0_n=beta_j* (((1-asymptomatic)*1/gamma2)+(asymptomatic*q/(gamma2+hosp_rate))+(asymptomatic*hosp_rate*l/((gamma2+hosp_rate)*gamma_hosp)))
beta_j=0.51
R0_n=beta_j* (((1-asymptomatic)*1/gamma2)+(asymptomatic*q/(gamma2+hosp_rate))+(asymptomatic*hosp_rate*l/((gamma2+hosp_rate)*gamma_hosp)))

######

S_vNS, V_v,NSE_vNS,P_vNS,A_vNS, I_vNS,J_vNS, R_vNS, D_vNS, RH_vNS=sim_svepaijrdNS_decay_ode(S0, V0,E0, P0,A0,I0,J0, R0, D0, beta_j,gamma2, gamma_hosp, alpha, n_days,
                                                      decay1, decay2, decay3, decay4, decay5, decay6, decay7, decay8, decay9, decay10, start_day, int1_delta, int2_delta, int3_delta, int4_delta, int5_delta, int6_delta, int7_delta, int8_delta, int9_delta,
                                                      fatal_hosp, asymptomatic, hosp_rate, q,  l, x,
                                                      p_m1, p_m2, p_m3, p_m4, p_m5, p_m6, p_m7, p_m8, p_m9, p_m10, delta_p, sigma, phi5, phi6, phi7, phi8, phi9, phi10, new_strain6, new_strain7, new_strain8, new_strain9, new_strain10, fracNS6, fracNS7, fracNS8, fracNS9, fracNS10)

icu_curve= J_vNS*icu_rate
vent_curve=J_vNS*vent_rate

hosp_rate_vNS=1.0
RateLos = namedtuple("RateLos", ("rate", "length_of_stay"))
hospitalized_vNS=RateLos(hosp_rate_vNS, hosp_los)
icu_rate_vNS= icu_rate
vent_rate_vNS= vent_rate
icu=RateLos(icu_rate_vNS, icu_los)
ventilated=RateLos(vent_rate_vNS, vent_los)


rates_vNS = tuple(each.rate for each in (hospitalized_vNS, icu, ventilated))
lengths_of_stay = tuple(each.length_of_stay for each in (hospitalized_vNS, icu, ventilated))


i_hospitalized_VNS, i_icu_VNS, i_ventilated_VNS= get_dispositions(J_vNS, rates_vNS, regional_hosp_share)
r_hospitalized_VNS, r_icu_VNS, r_ventilated_VNS = get_dispositions(RH_vNS, rates_vNS, regional_hosp_share)
d_hospitalized_VNS, d_icu_VNS, d_ventilated_VNS = get_dispositions(D_vNS, rates_vNS, regional_hosp_share)
dispositions_VNS2 = (
            i_hospitalized_VNS + r_hospitalized_VNS+ d_hospitalized_VNS,
            i_icu_VNS+r_icu_VNS+d_icu_VNS,
            i_ventilated_VNS+r_ventilated_VNS +d_ventilated_VNS)
hospitalized_VNS2, icu_VNS2, ventilated_VNS2 = (
            i_hospitalized_VNS,
            i_icu_VNS,
            i_ventilated_VNS)




# Projection days
plot_projection_days = n_days - 10


#############
# # SIR Model
# # New cases
#projection_admits = build_admissions_df(dispositions)
# # Census Table
#census_table = build_census_df(projection_admits)
# ############################

############
# SEIR Model
# New cases
#projection_admits_e = build_admissions_df(dispositions_e)
# Census Table
#census_table_e = build_census_df(projection_admits_e)

#############
# SEIR Model with phase adjustment
# New cases
#projection_admits_R = build_admissions_df(dispositions_R)
# Census Table
#census_table_R = build_census_df(projection_admits_R)

#############
# SEIR Model with phase adjustment and Disease Fatality
# New cases
#projection_admits_D = build_admissions_df(dispositions_D)
# Census Table
#census_table_D = build_census_df(projection_admits_D)

#############
# SEIR Model with phase adjustment and Disease Fatality
# New cases - using high social distancing
#projection_admits_D_socialcases = build_admissions_df(dispositions_D_socialcases)
# Census Table
#census_table_D_socialcases = build_census_df(projection_admits_D_socialcases)

#############
# SEIR Model with phase adjustment and Disease Fatality
# New cases - using dynamic doubling time and social distancing
#projection_admits_D_ecases = build_admissions_df(dispositions_D_ecases)
# Census Table
#census_table_D_ecases = build_census_df(projection_admits_D_ecases)

#############
# SEAIJRD Model
# New Cases
#projection_admits_A_ecases = build_admissions_df_n(dispositions_A_ecases)
## Census Table
#census_table_A_ecases = build_census_df(projection_admits_A_ecases)

#############
# SEPAIJRD Model
# Base model
# New Cases
projection_admits_P0 = build_admissions_df_n(dispositions_P0)
#st.dataframe(dispositions_P0)
#st.dataframe(projection_admits_P0)
## Census Table
census_table_P0 = build_census_df(projection_admits_P0)
#st.dataframe(census_table_P0)
# Higher Mask Use
# New Cases
projection_admits_P1 = build_admissions_df_n(dispositions_P1)
## Census Table
census_table_P1 = build_census_df(projection_admits_P1)

# Lower Mask Use
# New Cases
projection_admits_P2 = build_admissions_df_n(dispositions_P2)
## Census Table
census_table_P2 = build_census_df(projection_admits_P2)

# Lower Mask Use
# New Cases
projection_admits_P3 = build_admissions_df_n(dispositions_P3)
## Census Table
census_table_P3 = build_census_df(projection_admits_P3)


 #SVEPAIJRD Model
# Base model
# New Cases
projection_admits_V0 = build_admissions_df_n(dispositions_V0)
projection_admits_V1 = build_admissions_df_n(dispositions_V1)
projection_admits_V2 = build_admissions_df_n(dispositions_V2)
## Census Table
census_table_V0 = build_census_df(projection_admits_V0)
census_table_V1 = build_census_df(projection_admits_V1)
census_table_V2 = build_census_df(projection_admits_V2)

 #SVEPAIJRD Model +New Strain
# Base model
# New Cases
projection_admits_VNS0 = build_admissions_df_n(dispositions_VNS0)
projection_admits_VNS1 = build_admissions_df_n(dispositions_VNS1)
projection_admits_VNS2 = build_admissions_df_n(dispositions_VNS2)
#st.dataframe(dispositions_P0)
#st.dataframe(projection_admits_P0)
## Census Table
census_table_VNS0 = build_census_df(projection_admits_VNS0)
census_table_VNS1 = build_census_df(projection_admits_VNS1)
census_table_VNS2 = build_census_df(projection_admits_VNS2)
#st.dataframe(census_table_P0)

# Erie Graph of Cases: SEIR
# Admissions Graphs
# Erie Graph of Cases
def regional_admissions_chart(
    projection_admits: pd.DataFrame,
    plot_projection_days: int,
    as_date:bool = False) -> alt.Chart:
    """docstring"""

    projection_admits = projection_admits.rename(columns={"hosp": "Hospitalized", "icu": "ICU", "vent": "Ventilated"})

    tooltip_dict = {False: "day", True: "date:T"}
    if as_date:
        projection_admits = add_date_column(projection_admits)
        x_kwargs = {"shorthand": "date:T", "title": "Date"}
    else:
        x_kwargs = {"shorthand": "day", "title": "Days from initial infection"}

    return (
        alt
        .Chart(projection_admits.head(plot_projection_days))
        .transform_fold(fold=["Hospitalized", "ICU", "Ventilated"])
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

# , scale=alt.Scale(domain=[0, 3250])


#Comparison of Single line graph - Hospitalized, ICU, Vent and All
# if model_options == "Inpatient":
    # columns_comp = {"hosp": "Hospitalized"}
    # fold_comp = ["Hospitalized"]
    # capacity_col = {"total_county_beds":"Inpatient Beds"}
    # capacity_fol = ["Inpatient Beds"]
# if model_options == "ICU":
    # columns_comp = {"icu": "ICU"}
    # fold_comp = ["ICU"]
    # capacity_col = {"total_county_icu": "ICU Beds"}
    # capacity_fol = ["ICU Beds"]
# if model_options == "Ventilated":
    # columns_comp = {"vent": "Ventilated"}
    # fold_comp = ["Ventilated"]

def ip_chart(
    projection_admits: pd.DataFrame,
    plot_projection_days: int,
    as_date:bool = False) -> alt.Chart:
    """docstring"""

    projection_admits = projection_admits.rename(columns=columns_comp|capaity_col)

    tooltip_dict = {False: "day", True: "date:T"}
    if as_date:
        projection_admits = add_date_column(projection_admits)
        x_kwargs = {"shorthand": "date:T", "title": "Date"}
    else:
        x_kwargs = {"shorthand": "day", "title": "Days from initial infection"}

    return (
        alt
        .Chart(projection_admits.head(plot_projection_days))
        .transform_fold(fold=fold_comp+capacity_fol)
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

#, scale=alt.Scale(domain=[0, 100])
# alt.value('orange')


###################### Vertical Lines Graph ###################
vertical = pd.DataFrame({'day': [int1_delta, int2_delta, int3_delta, int4_delta, int5_delta, int6_delta, int7_delta, int8_delta, int9_delta]})

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

vertical1 = vertical_chart(vertical, as_date=as_date)



##############################
#4/3/20 First Projection Graph - Admissions
##############################
#st.header("""Projected Admissions Models for Erie County""")
#st.subheader("Projected number of **daily** COVID-19 admissions for Erie County: SEIR -Phase Adjusted R_0 with Case Fatality and Asymptomatic Component")
#admits_graph_seir = regional_admissions_chart(projection_admits_e,
#        plot_projection_days,
#        as_date=as_date)
#admits_graph = regional_admissions_chart(projection_admits_D,
#        plot_projection_days,
#        as_date=as_date)
### High Social Distancing
#admits_graph_highsocial = regional_admissions_chart(projection_admits_D_socialcases,
#        plot_projection_days,
#        as_date=as_date)
### Dynamic Doubling Time
#admits_graph_ecases = regional_admissions_chart(projection_admits_D_ecases,
#        plot_projection_days,
#        as_date=as_date)
### SEAIJRD
#admits_graph_A= regional_admissions_chart(projection_admits_A_ecases,
#        plot_projection_days,
#        as_date=as_date)
### SEPAIJRD
admits_graph_P= regional_admissions_chart(projection_admits_P0,
        plot_projection_days,
        as_date=as_date)

# st.altair_chart(
    # #admits_graph_seir
    # #+
    # #admits_graph
    # #+
    # vertical1
    # #+ admits_graph_ecases
    # + admits_graph_A
    # #+ admits_graph_highsocial
    # #+ erie_admit24_line
    # , use_container_width=True)

# st.subheader("Projected number of **daily** COVID-19 admissions for Erie County: SEIR - Phase Adjusted R_0 with Case Fatality with Asymptomatic, Pre-Symptomatic, and Mask-use")
# st.altair_chart(
    # #admits_graph_seir
    # #+
    # #admits_graph
    # #+
    # vertical1
    # #+ admits_graph_ecases
    # + admits_graph_P
    # #+ admits_graph_highsocial
    # + erie_admit24_line
    # , use_container_width=True)


if st.checkbox("Show more about the assumptions and specifications of the SEIR model"):
    st.subheader(
    "[Deterministic SEIR model](https://www.tandfonline.com/doi/full/10.1080/23737867.2018.1509026)")
    st.markdown(
    """The model consists of individuals who are either _Susceptible_ ($S$), _Exposed_ ($E$), _Infected_ ($I$), _Recovered_ ($R$), or _Fatal_ ($D$).
The epidemic proceeds via a growth and decline process. This is the core model of infectious disease spread and has been in use in epidemiology for many years."""
)
    st.markdown("""The system of differential equations are given by the following 5 equations.""")

    st.latex(r'''\frac{ds}{dt}=-\rho_t \beta SI/N''')
    st.latex(r'''\frac{de}{dt}=\rho_t \beta SI/N - \alpha E''')
    st.latex(r'''\frac{di}{dt}= \alpha E - \gamma I''')
    st.latex(r'''\frac{dr}{dt}=(1-f) \gamma I''')
    st.latex(r'''\frac{dd}{dt}=f \gamma I''')

    st.markdown(
    """where $\gamma$ is $1/mean\ infectious\ rate$, $$\\alpha$$ is $1/mean\ incubation\ period$, $$\\rho$$ is the rate of social distancing at time $t$,
and $$\\beta$$ is the rate of transmission. More information, including parameter specifications and reasons for model choice can be found [here]("https://github.com/gabai/stream_KH/wiki).""")


##### with vaccinations
    ### SEPAIJRD
admits_graph_V= regional_admissions_chart(projection_admits_V0,
        plot_projection_days,
        as_date=as_date)

# st.altair_chart(
    # #admits_graph_seir
    # #+
    # #admits_graph
    # #+
    # vertical1
    # #+ admits_graph_ecases
    # + admits_graph_A
    # #+ admits_graph_highsocial
    # #+ erie_admit24_line
    # , use_container_width=True)

# st.subheader("Projected number of **daily** COVID-19 admissions for Erie County: SEIR - Phase Adjusted R_0 with Case Fatality with Asymptomatic, Pre-Symptomatic, Vaccinations, and Mask-use")
# st.altair_chart(
    # #admits_graph_seir
    # #+
    # #admits_graph
    # #+
    # vertical1
    # #+ admits_graph_ecases
    # + admits_graph_V
    # #+ admits_graph_highsocial
    # + erie_admit24_line
    # , use_container_width=True)


##### with vaccinations +New Strain
    ### SEPAIJRD
admits_graph_VNS= regional_admissions_chart(projection_admits_VNS0,
        plot_projection_days,
        as_date=as_date)

# st.altair_chart(
    # #admits_graph_seir
    # #+
    # #admits_graph
    # #+
    # vertical1
    # #+ admits_graph_ecases
    # + admits_graph_A
    # #+ admits_graph_highsocial
    # #+ erie_admit24_line
    # , use_container_width=True)

st.subheader("Projected number of **daily** COVID-19 admissions for Erie County: SEIR - Phase Adjusted R_0 with Case Fatality with Asymptomatic, Pre-Symptomatic, Vaccinations, New Strain transmissibility, and Mask-use")
st.altair_chart(
    #admits_graph_seir
    #+
    #admits_graph
    #+
    vertical1
    #+ admits_graph_ecases
    + admits_graph_VNS
    #+ admits_graph_highsocial
    + erie_admit24_line
    , use_container_width=True)



#sir = regional_admissions_chart(projection_admits, plot_projection_days, as_date=as_date)
#seir = regional_admissions_chart(projection_admits_e, plot_projection_days, as_date=as_date)
#seir_r = regional_admissions_chart(projection_admits_R, plot_projection_days, as_date=as_date)
#seir_d = regional_admissions_chart(projection_admits_D, plot_projection_days, as_date=as_date)


# if st.checkbox("Show Graph of Erie County Projected Admissions with Model Comparison of Social Distancing"):
    # st.subheader("Projected number of **daily** COVID-19 admissions for Erie County: Model Comparison (Left: 0% Social Distancing, Right: Step-Wise Social Distancing)")
    # st.altair_chart(
        # alt.layer(seir.mark_line())
        # + alt.layer(seir_d.mark_line())
        # + alt.layer(vertical1.mark_rule())
        # , use_container_width=True)

def hospital_admissions_chart(
    projection_admits: pd.DataFrame,
    plot_projection_days: int,
    as_date:bool = False) -> alt.Chart:
    """docstring"""
    projection_admits = projection_admits.rename(columns=col_name1)

    tooltip_dict = {False: "day", True: "date:T"}
    if as_date:
        projection_admits = add_date_column(projection_admits)
        x_kwargs = {"shorthand": "date:T", "title": "Date"}
    else:
        x_kwargs = {"shorthand": "day", "title": "Days from initial infection"}

    return (
        alt
        .Chart(projection_admits.head(plot_projection_days))
        .transform_fold(fold=fold_name1)
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


################################################
################################################
#############    Census Graphs        ##########
################################################
################################################
st.header("""Projected Census Models for Erie County""")


# Comparison of Census Single line graph - Hospitalized, ICU, Vent
# if model_options == "Inpatient":
    # columns_comp_census = {"hosp": "Hospital Census"}
    # fold_comp_census = ["Hospital Census"]
    # graph_selection = erie_lines_ip
# if model_options == "ICU":
    # columns_comp_census = {"icu": "ICU Census"}
    # fold_comp_census = ["ICU Census"]
    # graph_selection = erie_lines_icu
# if model_options == "Ventilated":
    # columns_comp_census = {"vent": "Ventilated Census"}
    # fold_comp_census = ["Ventilated Census"]
    # graph_selection = erie_lines_vent

def ip_census_chart(
    census: pd.DataFrame,
    plot_projection_days: int,
    as_date:bool = False) -> alt.Chart:
    """docstring"""
    census = census.rename(columns={"hosp": "Hospital Census"})

    tooltip_dict = {False: "day", True: "date:T"}
    if as_date:
        census = add_date_column(census.head(plot_projection_days))
        x_kwargs = {"shorthand": "date:T", "title": "Date"}
    else:
        x_kwargs = {"shorthand": "day", "title": "Days from initial infection"}

    return (
        alt
        .Chart(census)
        .transform_fold(fold=["Hospital Census"])
        .mark_line(point=False)
        .encode(
            x=alt.X(**x_kwargs),
            y=alt.Y("value:Q", title="Hospital Census"),
            #y=alt.Y("value:Q", title="Hospital Census", scale=alt.Scale(domain=[0, 1600])),
            #color="key:N",

            color=alt.value('orange'),
            tooltip=[
                tooltip_dict[as_date],
                alt.Tooltip("value:Q", format=".0f", title="Census"),
                "key:N",
            ],
        )
        .interactive()
    )

# on y axi
# , scale=alt.Scale(domain=[0, 250])


    #, scale=alt.Scale(domain=[0, 40000])
    # scale=alt.Scale(domain=[-5, 9000])



#sir_ip_c = ip_census_chart(census_table, plot_projection_days, as_date=as_date)
#seir_ip_c = ip_census_chart(census_table_e, plot_projection_days, as_date=as_date)
#seir_r_ip_c = ip_census_chart(census_table_R, plot_projection_days, as_date=as_date)
#seir_d_ip_c = ip_census_chart(census_table_D, plot_projection_days, as_date=as_date)
###

### 4/20/20 for high social distancing model
#seir_d_ip_highsocial = ip_census_chart(census_table_D_socialcases, plot_projection_days, as_date=as_date)
### 4/17/20 for stepwise SD/DT model
#seir_d_ip_ecases = ip_census_chart(census_table_D_ecases, plot_projection_days, as_date=as_date)
### 4/22/20 seaijrd
#seir_A_ip_ecases = ip_census_chart(census_table_A_ecases, plot_projection_days, as_date=as_date)
### 4/22/20 sepaijrd
#seir_P_ip_ecases = ip_census_chart(census_table_P_ecases, plot_projection_days, as_date=as_date)
### 6/22/20 sepaijrd
seir_P0 = ip_census_chart(census_table_P0[9:len(census_table_P0)], plot_projection_days, as_date=as_date)
seir_P1 = ip_census_chart(census_table_P1[9:len(census_table_P1)], plot_projection_days, as_date=as_date)
seir_P2 = ip_census_chart(census_table_P2[9:len(census_table_P2)], plot_projection_days, as_date=as_date)
seir_P3 = ip_census_chart(census_table_P3, plot_projection_days, as_date=as_date)

# Vaccine Curve
seir_V0 = ip_census_chart(census_table_V0[9:len(census_table_V0)], plot_projection_days, as_date=as_date)
seir_V1 = ip_census_chart(census_table_V1[9:len(census_table_V0)], plot_projection_days, as_date=as_date)
seir_V2 = ip_census_chart(census_table_V2[9:len(census_table_V0)], plot_projection_days, as_date=as_date)
# Vaccine w/ Strain
seir_VNS0 = ip_census_chart(census_table_VNS0[9:len(census_table_VNS0)], plot_projection_days, as_date=as_date)
seir_VNS1 = ip_census_chart(census_table_VNS1[9:len(census_table_VNS0)], plot_projection_days, as_date=as_date)
seir_VNS2 = ip_census_chart(census_table_VNS2[9:len(census_table_VNS0)], plot_projection_days, as_date=as_date)

# Chart of Model Comparison for SEIR and Adjusted with Erie County Data
#st.subheader("Comparison of COVID-19 admissions for Erie County: Data vs Model (SEAIJRD)")
#st.altair_chart(
    #alt.layer(seir_ip_c.mark_line())
    #+ alt.layer(seir_d_ip_c.mark_line())
    #+ alt.layer(seir_d_ip_ecases.mark_line())
    #+
    #alt.layer(seir_A_ip_ecases.mark_line())
    #+ alt.layer(seir_d_ip_highsocial.mark_line())
    #+ alt.layer(graph_selection)
    #+ alt.layer(vertical1)
    #, use_container_width=True)

# Main Graph - GA
# Active as of 10/5/20
# st.subheader("Comparison of COVID-19 admissions for Erie County: Data vs Model (SEPAIJRD)")
# st.altair_chart(
    # #alt.layer(seir_ip_c.mark_line())
    # #+ alt.layer(seir_d_ip_c.mark_line())
    # #+ alt.layer(seir_d_ip_ecases.mark_line())
    # #+
    # alt.layer(seir_P0.mark_line())
    # #+ alt.layer(seir_d_ip_highsocial.mark_line())
    # + alt.layer(erie_lines_ip)
    # + alt.layer(vertical1)
    # , use_container_width=True)

# V_Graph
# Main Graph - VACCINATIONS + strain
# Active as of 2/3/21
# st.subheader("Comparison of COVID-19 hospital admissions for Erie County: Model Comparison - Vaccine (SVEPAIJRD)")
# st.altair_chart(
    # #alt.layer(seir_ip_c.mark_line())
    # #+ alt.layer(seir_d_ip_c.mark_line())
    # #+ alt.layer(seir_d_ip_ecases.mark_line())
    # #+
    # alt.layer(seir_P0.mark_line())
    # +
    # seir_V0
    # #+alt.layer(seir_VNS0.mark_line())
    # #+ alt.layer(seir_d_ip_highsocial.mark_line())
    # + alt.layer(erie_lines_ip)
    # + alt.layer(vertical1)
    # , use_container_width=True)

# Main Graph - VACCINATIONS
# Active as of 2/3/21
# st.subheader("Comparison of COVID-19 hospital admissions for Erie County: Model Comparison - Vaccine Only")
# st.altair_chart(
    # #alt.layer(seir_ip_c.mark_line())
    # #+ alt.layer(seir_d_ip_c.mark_line())
    # #+ alt.layer(seir_d_ip_ecases.mark_line())
    # #+
    # #alt.layer(seir_P0.mark_line())
    # #+
    # #alt.layer(seir_VNS0.mark_line())
    # #+
    # seir_V0
    # #+
    # #alt.layer(seir_VNS1.mark_line())
    # #+ seir_V1
    # #+ seir_V2
    # + alt.layer(erie_lines_ip)
    # + alt.layer(vertical1)
    # , use_container_width=True)


# Main Graph - VACCINATIONS
# Active as of 2/3/21
# st.subheader("Comparison of COVID-19 hospital admissions for Erie County: Model Comparison - Vaccine and Vaccine with half FM/SD")
# st.altair_chart(
    # #alt.layer(seir_ip_c.mark_line())
    # #+ alt.layer(seir_d_ip_c.mark_line())
    # #+ alt.layer(seir_d_ip_ecases.mark_line())
    # #+
    # #alt.layer(seir_P0.mark_line())
    # #+
    # #alt.layer(seir_VNS0.mark_line())
    # #+
    # #seir_V0
    # #+
    # #alt.layer(seir_VNS1.mark_line())
    # #+ seir_V1
    # #+
    # seir_V2
    # + alt.layer(erie_lines_ip)
    # + alt.layer(vertical1)
    # , use_container_width=True)

# Main Graph - VACCINATIONS
# Active as of 2/3/21
# st.subheader("Comparison of COVID-19 hospital admissions for Erie County: Model Comparison - Vaccine with Variant and half FM/SD")
# st.altair_chart(
    # #alt.layer(seir_ip_c.mark_line())
    # #+ alt.layer(seir_d_ip_c.mark_line())
    # #+ alt.layer(seir_d_ip_ecases.mark_line())
    # #+
    # #alt.layer(seir_P0.mark_line())
    # #+
    # #alt.layer(seir_VNS0.mark_line())
    # #+
    # seir_V0
    # #+
    # #+alt.layer(seir_VNS1.mark_line())
    # + seir_V1
    # + seir_V2
    # + alt.layer(erie_lines_ip)
    # + alt.layer(vertical1)
    # , use_container_width=True)

# S_Graph
# Main Graph - VACCINATIONS + strain
# Active as of 2/15/21
st.subheader("Comparison of COVID-19 hospital admissions for Erie County: Model Comparison - Efficacy of Vaccine with New COVID Variant (SVEPAIJRD)")
st.altair_chart(
    #alt.layer(seir_ip_c.mark_line())
    #+ alt.layer(seir_d_ip_c.mark_line())
    #+ alt.layer(seir_d_ip_ecases.mark_line())
    #+
    #alt.layer(seir_P0.mark_line())
    #+
    alt.layer(seir_VNS0.mark_line())
    #+
    #alt.layer(seir_VNS1.mark_line())
    #+
    #alt.layer(seir_VNS2.mark_line())
    + alt.layer(erie_lines_ip)
    + alt.layer(vertical1)
    , use_container_width=True)



#############################################################################
# Changes 7/21/21




############################### prevalence and incidence ###########################
# https://www.tandfonline.com/doi/full/10.1057/hs.2015.2
####################################################################################
st.subheader("Prevalence and Incidence Across Time")

st.markdown("""Incidence is measured as the number of new cases daily, (from compartment P) prevalence is measured
as the population infected with the disease (A,I,J) daily.""")

#st.dataframe(census_table_P0)

#S_vNS, V_v,NSE_vNS,P_vNS,A_vNS, I_vNS,J_vNS, R_vNS, D_vNS, RH_vNS

## incidence
#frame = [A_vNS, I_vNS, J_vNS, R_vNS, D_vNS]
#dispositions_inc = np.array(frame)
#dispositions_inc = pd.DataFrame(data=dispositions_inc).T
#st.write(dispositions_inc)
#st.dataframe(dispositions_inc)
#dispositions_inc = pd.DataFrame(dispositions_inc, columns=['newcases'])
# dispositions_inc2 = dispositions_inc.iloc[:-1, :] - dispositions_inc.shift(1)
# dispositions_inc2["day"] = range(dispositions_inc2.shape[0])
# dispositions_inc2["TotalCases"]=S_vNS
# dispositions_inc2.at[0,'newcases']=0
# dispositions_inc2["incidencerate"]=dispositions_inc2['newcases']/dispositions_inc2['TotalCases']


def additional_projections_chart2(i, p)  -> alt.Chart:
    dat = pd.DataFrame({"Incidence Rate":i,"Prevalence Rate":p})

    return (
        alt
        .Chart(dat.reset_index())
        .transform_fold(fold=["Incidence Rate", "Prevalence Rate"])
        .mark_line(point=False)
        .encode(
            x=alt.X("index", title="Days from initial infection"),
            y=alt.Y("value:Q", title="Case Volume"),
            tooltip=["key:N", "value:Q"],
            color="key:N"
        )
        .interactive()
    )

st.altair_chart(additional_projections_chart2(dispositions_inc2["incidencerate"], dispositions_prev["pointprevalencerate"]), use_container_width=True)




st.dataframe(dispositions_inc2)

st.dataframe(dispositions_prev)
