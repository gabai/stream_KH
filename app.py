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
secret = os.environ['SECRET_KEY']

# Prompt the user for the secret
password = st.text_input("Secret Handshake:", value="", type="password")

# If the secrete provided matches the ENV, proceeed with the app
if password == secret:
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
        
        counter = 0
        for i in hosp_list:
            projection[groups[0]+"_"+i] = projection.hosp*bed_share.iloc[3,counter]
            projection[groups[1]+"_"+i] = projection.icu*bed_share.iloc[3,counter]
            projection[groups[2]+"_"+i] = projection.vent*bed_share.iloc[3,counter]
            counter +=1
            if counter == 4: break
        
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
        
        counter = 0
        for i in hosp_list:
            projection[groups[0]+"_"+i] = projection.hosp*bed_share.iloc[3,counter]
            projection[groups[1]+"_"+i] = projection.icu*bed_share.iloc[3,counter]
            projection[groups[2]+"_"+i] = projection.vent*bed_share.iloc[3,counter]
            counter +=1
            if counter == 4: break
        
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
        
        counter = 0
        for i in hosp_list:
            projection[groups[0]+"_"+i] = projection.hosp*bed_share.iloc[3,counter]
            projection[groups[1]+"_"+i] = projection.icu*bed_share.iloc[3,counter]
            projection[groups[2]+"_"+i] = projection.vent*bed_share.iloc[3,counter]
            counter +=1
            if counter == 4: break
        
        # New cases
        projection_admits = projection.iloc[:-1, :] - projection.shift(1)
        projection_admits["day"] = range(projection_admits.shape[0])
        return projection_admits


    def build_census_df(
        projection_admits: pd.DataFrame) -> pd.DataFrame:
        """ALOS for each category of COVID-19 case (total guesses)"""
        n_days = np.shape(projection_admits)[0]
        los_dict = {
        "hosp": hosp_los, "icu": icu_los, "vent": vent_los,
        "hosp_kh": hosp_los, "icu_kh": icu_los, "vent_kh": vent_los,
        "hosp_ecmc": hosp_los, "icu_ecmc": icu_los, "vent_ecmc": vent_los,
        "hosp_chs": hosp_los, "icu_chs": icu_los, "vent_chs": vent_los,
        "hosp_rpci": hosp_los, "icu_rpci": icu_los, "vent_rpci": vent_los
        }

        census_dict = dict()
        for k, los in los_dict.items():
            census = (
                projection_admits.cumsum().iloc[:-los, :]
                - projection_admits.cumsum().shift(los).fillna(0)
            ).apply(np.ceil)
            census_dict[k] = census[k]

        census_df = pd.DataFrame(census_dict)
        census_df["day"] = census_df.index
        census_df = census_df[["day", "hosp", "icu", "vent", 
        "hosp_kh", "icu_kh", "vent_kh", 
        "hosp_ecmc", "icu_ecmc", "vent_ecmc",
        "hosp_chs", "icu_chs", "vent_chs",
        "hosp_rpci", "icu_rpci", "vent_rpci"
        ]]
        
        census_df['total_county_icu'] = icu_county
        census_df['total_county_beds'] = beds_county
        census_df['expanded_icu_county'] = expanded_icu_county_05
        census_df['expanded_beds_county'] = expanded_beds_county_05
        census_df['expanded_icu_county2'] = expanded_icu_county_1
        census_df['expanded_beds_county2'] = expanded_beds_county_1
        census_df['icu_beds'] = icu_val
        census_df['total_beds'] = total_beds_val
        census_df['total_vents'] = vent_val
        census_df['expanded_beds'] = expanded_beds_val
        census_df['expanded_icu_beds'] = expanded_icu_val
        census_df['expanded_vent_beds'] = expanded_vent_val
        census_df['expanded_beds2'] = expanded_beds2_val
        census_df['expanded_icu_beds2'] = expanded_icu2_val
        census_df['expanded_vent_beds2'] = expanded_vent2_val
        
        # PPE for hosp/icu
        census_df['ppe_mild_d'] = census_df['hosp'] * ppe_mild_val_lower
        census_df['ppe_mild_u'] = census_df['hosp'] * ppe_mild_val_upper
        census_df['ppe_severe_d'] = census_df['icu'] * ppe_severe_val_lower
        census_df['ppe_severe_u'] = census_df['icu'] * ppe_severe_val_upper
        census_df['ppe_mean_mild'] = census_df[["ppe_mild_d","ppe_mild_u"]].mean(axis=1)
        census_df['ppe_mean_severe'] = census_df[["ppe_severe_d","ppe_severe_u"]].mean(axis=1)
        
        for hosp in hosp_list:
            census_df['ppe_mild_d_'+hosp] = census_df['hosp_'+hosp] * ppe_mild_val_lower
            census_df['ppe_mild_u_'+hosp] = census_df['hosp_'+hosp] * ppe_mild_val_upper
            census_df['ppe_severe_d_'+hosp] = census_df['icu_'+hosp] * ppe_severe_val_lower
            census_df['ppe_severe_u_'+hosp] = census_df['icu_'+hosp] * ppe_severe_val_upper
            census_df['ppe_mean_mild_'+hosp] = census_df[["ppe_mild_d_"+hosp,"ppe_mild_u_"+hosp]].mean(axis=1)
            census_df['ppe_mean_severe_'+hosp] = census_df[["ppe_severe_d_"+hosp,"ppe_severe_u_"+hosp]].mean(axis=1)
        
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
            elif int2_delta<=day<=end_delta:
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
            decay1:float, decay2:float, decay3: float, decay4: float, end_delta: int, fatal: float
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
                elif int2_delta<=day<=end_delta:
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


    # Model with high social distancing
    def sim_seird_decay_social(
        s: float, e:float, i: float, r: float, d: float, beta: float, gamma: float, alpha: float, n_days: int,
        decay1:float, decay2:float, decay3: float, decay4: float, end_delta: int, fatal: float
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulate the SIR model forward in time."""
        s, e, i, r, d= (float(v) for v in (s, e, i, r, d))
        n = s + e + i + r + d
        s_v, e_v, i_v, r_v, d_v = [s], [e], [i], [r], [d]
        for day in range(n_days):
            if start_day<=day<=int1_delta:
                beta = (alpha+(2 ** (1 / 2) - 1))*((2 ** (1 / 2) - 1) + (1/infectious_period)) / (alpha*S)
                beta_decay=beta*(1-.02)
            elif int1_delta<=day<=int2_delta:
                beta = (alpha+(2 ** (1 / 2) - 1))*((2 ** (1 / 2) - 1)+ (1/infectious_period)) / (alpha*S)
                beta_decay=beta*(1-.52)
            elif int2_delta<=day<=end_delta:
                beta = (alpha+(2 ** (1 / 2) - 1))*((2 ** (1 / 2) - 1)+ (1/infectious_period)) / (alpha*S)
                beta_decay=beta*(1-.83)
            else:
                beta = (alpha+(2 ** (1 / 2) - 1))*((2 ** (1 / 2) - 1)+ (1/infectious_period)) / (alpha*S)
                beta_decay=beta*(1-.73)
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

    # Model with dynamic doubling time
    def sim_seird_decay_erie(
        s: float, e:float, i: float, r: float, d: float, beta: float, gamma: float, alpha: float, n_days: int,
        decay1:float, decay2:float, decay3: float, decay4: float, end_delta: int, fatal: float
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulate the SIR model forward in time."""
        s, e, i, r, d= (float(v) for v in (s, e, i, r, d))
        n = s + e + i + r + d
        s_v, e_v, i_v, r_v, d_v = [s], [e], [i], [r], [d]
        for day in range(n_days):
            if start_day<=day<=int1_delta:
                beta = (alpha+(2 ** (1 / 1.61) - 1))*((2 ** (1 / 1.61) - 1) + (1/infectious_period)) / (alpha*S)
                beta_decay=beta*(1-.3)
            elif int1_delta<=day<=int2_delta:
                beta = (alpha+(2 ** (1 / 2.65) - 1))*((2 ** (1 / 2.65) - 1)+ (1/infectious_period)) / (alpha*S)
                beta_decay=beta*(1-.3)
            elif int2_delta<=day<=end_delta:
                beta = (alpha+(2 ** (1 / 5.32) - 1))*((2 ** (1 / 5.32) - 1)+ (1/infectious_period)) / (alpha*S)
                beta_decay=beta*(1-.5)
            else:
                beta = (alpha+(2 ** (1 / 9.70) - 1))*((2 ** (1 / 9.70) - 1)+ (1/infectious_period)) / (alpha*S)
                beta_decay=beta*(1-.30)
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

    def sim_seijcrd_decay(
        s: float, e:float, i: float, j:float, c: float, r: float, d: float, beta: float, gamma: float, alpha: float, n_days: int,
        decay1:float, decay2:float, decay3: float, decay4: float, end_delta: int, fatal_hosp: float, hosp_rate: float, icu_rate: float, icu_days:float, crit_lag: float, death_days:float
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulate the SIR model forward in time."""
        s, e, i, j, c, r, d= (float(v) for v in (s, e, i, c, j, r, d))
        n = s + e + i + j+r + d
        s_v, e_v, i_v, j_v, c_v, r_v, d_v = [s], [e], [i], [j], [c], [r], [d]
        for day in range(n_days):
            if 0<=day<=21:
                beta = (alpha+(2 ** (1 / 1.61) - 1))*((2 ** (1 / 1.61) - 1) + (1/infectious_period)) / (alpha*S)
                beta_decay=beta*(1-decay1)
            elif 22<=day<=28:
                beta = (alpha+(2 ** (1 / 2.65) - 1))*((2 ** (1 / 2.65) - 1)+ (1/infectious_period)) / (alpha*S)
                beta_decay=beta*(1-decay2)
            elif 29<=day<=end_delta: 
                beta = (alpha+(2 ** (1 / 5.32) - 1))*((2 ** (1 / 5.32) - 1)+ (1/infectious_period)) / (alpha*S)
                beta_decay=beta*(1-decay3)
            else:
                beta = (alpha+(2 ** (1 / 9.70) - 1))*((2 ** (1 / 9.70) - 1)+ (1/infectious_period)) / (alpha*S)
                beta_decay=beta*(1-decay4)
            s, e, i,j, c, r,d = seijcrd(s, e, i,j, c, r, d, beta_decay, gamma, alpha, n, fatal_hosp, hosp_rate, icu_rate, icu_days, crit_lag, death_days)
            s_v.append(s)
            e_v.append(e)
            i_v.append(i)
            j_v.append(j)
            c_v.append(c)
            r_v.append(r)
            d_v.append(d)

        return (
            np.array(s_v),
            np.array(e_v),
            np.array(i_v),
            np.array(j_v),
            np.array(c_v),
            np.array(r_v),
            np.array(d_v)
        )


    def betanew(t,beta):
        if start_day<= t <= int1_delta:
            beta_decay=beta*(1-decay1)
        elif int1_delta<=t<int2_delta:
            beta_decay=beta*(1-decay2)
        elif int2_delta<=t<=end_delta:
            beta_decay=beta*(1-decay3)
        elif end_delta<=t<=step2_delta:
            beta_decay=beta*(1-decay4)
        else:
            beta_decay=beta*(1-decay5)    
        return beta_decay

    #The SIR model differential equations with ODE solver.
    def derivdecay(y, t, N, beta, gamma1, gamma2, alpha, p, hosp,q,l,n_days, decay1, decay2, decay3, decay4, decay5, start_day, int1_delta, int2_delta, end_delta, step2_delta, fatal_hosp ):
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
        s, e,a,i, j,r, d, beta, gamma1, gamma2, alpha, n_days,decay1,decay2,decay3, decay4, decay5, start_day, int1_delta, int2_delta,end_delta, step2_delta, fatal_hosp, p, hosp, q,
        l):
        n = s + e + a + i + j+ r + d
        rh=0
        y0= s,e,a,i,j,r,d, rh
        
        t=np.arange(0, n_days, step=1)
        ret = odeint(derivdecay, y0, t, args=(n, beta, gamma1, gamma2, alpha, p, hosp,q,l, n_days, decay1, decay2, decay3, decay4, decay5, start_day, int1_delta, int2_delta, end_delta, step2_delta, fatal_hosp))
        S_n, E_n,A_n, I_n,J_n, R_n, D_n ,RH_n= ret.T
        
        return (S_n, E_n,A_n, I_n,J_n, R_n, D_n, RH_n)


    ####The SIR model differential equations with ODE solver. Presymptomatic and masks
    def betanew2(t,beta,x,p_m1, pm_2 ):
        if start_day<= t <= int1_delta:
            beta_decay=beta*(1-decay1)
        elif int1_delta<=t<int2_delta:
            beta_decay=beta*(1-decay2)
        elif int2_delta<=t<=end_delta:
            beta_decay=beta*(1-decay3)*(1-(x*p_m1))**2
        elif end_delta<=t<=step2_delta:
            beta_decay=beta*(1-decay4)*(1-(x*p_m1))**2 
        else:
            beta_decay=beta*(1-0.4)*(1-(x*p_m2))**2    
        return beta_decay

    def derivdecayP(y, t, beta, gamma1, gamma2, alpha, sym, hosp,q,l,n_days, decay1,decay2, decay3, decay4,decay5,start_day, int1_delta, int2_delta, end_delta,
                    step2_delta,fatal_hosp, x, p_m1, p_m2, delta_p ):
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
        int2_delta,end_delta, step2_delta, fatal_hosp, sym, hosp, q,
        l,x, p_m1, p_m2, delta_p):
        n = s + e + p+a + i + j+ r + d
        rh=0
        y0= s,e,p,a,i,j,r,d, rh
        
        t=np.arange(0, n_days, step=1)
        ret = odeint(derivdecayP, y0, t, args=(beta, gamma1, gamma2, alpha, sym, hosp,q,l, n_days, decay1, decay2, decay3, decay4, decay5, start_day, int1_delta,
                                               int2_delta, end_delta, step2_delta, fatal_hosp, x, p_m1, p_m2, delta_p))
        S_n, E_n,P_n,A_n, I_n,J_n, R_n, D_n ,RH_n= ret.T
        
        return (S_n, E_n,P_n,A_n, I_n,J_n, R_n, D_n, RH_n)
    
        
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

        

    # Adding ICU bed for Erie county
    # 3/25/20
    icu_county = 468
    beds_county = 2762
    # Bed expansion at 50%
    expanded_icu_county_05 = 369
    expanded_beds_county_05 = 3570
    # Bed expansion at 100%
    expanded_icu_county_1 = 492
    expanded_beds_county_1 = 4760

    # PPE Values
    ppe_mild_val_lower = 14
    ppe_mild_val_upper = 15
    ppe_severe_val_lower = 15
    ppe_severe_val_upper = 24

    # List of Hospitals
    hosp_list = ['kh', 'ecmc', 'chs', 'rpci']
    groups = ['hosp', 'icu', 'vent']

    # Hospital Bed Sharing Percentage
    # ignore the first 3 numbers
    data = {
        'Kaleida' : [0.34, 0.34, 0.26, 0.38],
        'ECMC': [0.14, 0.20, 0.17, 0.23], 
        'CHS': [0.21, 0.17, 0.18, 0.33],
        'RPCI': [0.0, 0.09, 0.06, 0.05]
    }
    bed_share = pd.DataFrame(data)

    url = 'https://raw.githubusercontent.com/gabai/stream_KH/master/Cases_Erie.csv'
    erie_df = pd.read_csv(url)
    erie_df['Date'] = pd.to_datetime(erie_df['Date'])

    # Populations and Infections
    erie = 1400000
    monroe = 741770
    cases_erie = erie_df['Cases'].iloc[-1]
    S_default = erie
    known_infections = erie_df['Cases'].iloc[-1]
    known_cases = erie_df['Admissions'].iloc[-1]
    regional_hosp_share = 1.0
    S = erie


    # Widgets
    hosp_options = st.sidebar.radio(
        "Hospitals Systems", ('Kaleida', 'ECMC', 'CHS', 'RPCI'))
        
    model_options = st.sidebar.radio(
        "Service", ('Inpatient', 'ICU', 'Ventilated'))

    current_hosp = st.sidebar.number_input(
        "Total Hospitalized Cases", value=known_cases, step=1.0, format="%f")

    doubling_time = st.sidebar.number_input(
        "Doubling Time (days)", value=3.0, step=1.0, format="%f")

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
        "Social distancing (% reduction in social contact) in Week 3 - School Closure", 0, 100, value=10, step=5, format="%i")/100.0

    intervention2 = st.sidebar.date_input(
        "Date of change in Social Distancing - Closure Businesses, Shelter in Place", datetime(2020,3,28))
    int2_delta = (intervention2 - start_date).days

    decay3 = st.sidebar.number_input(
        "Social distancing (% reduction in social contact) from Week 3 to change in SD - After Business Closure%", 0, 100, value=25 ,step=5, format="%i")/100.0

    end_date = st.sidebar.date_input(
        "Step 1 reduction in social distancing", datetime(2020,5,15))
    # Delta from start and end date for decay4
    end_delta = (end_date - start_date).days

    decay4 = st.sidebar.number_input(
        "Step 1 reduction in social distancing %", 0, 100, value=25 ,step=5, format="%i")/100.0

    step2 = st.sidebar.date_input(
        "Step 2 reduction in social distancing", datetime(2020,6,15))
    # Delta from start and end date for decay4
    step2_delta = (step2 - start_date).days

    decay5 = st.sidebar.number_input(
        "Step 2 reduction in social distancing %", 0, 100, value=25 ,step=5, format="%i")/100.0


    hosp_rate = (
        st.sidebar.number_input("Hospitalization %", 0.0, 100.0, value=1.2, step=0.50, format="%f")/ 100.0)

    icu_rate = (
        st.sidebar.number_input("ICU %", 0.0, 100.0, value=25.0, step=5.0, format="%f") / 100.0)

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

    p_m1 = (st.sidebar.number_input(
    "Percent of people adhering to mask-wearing after April 22,2020", 0.0, 100.0, value=42.0 ,step=0.1, format="%f")/100.0)
    p_m2 = (st.sidebar.number_input(
    "Percent of people adhering to mask-wearing during Phased transitioning", 0.0, 100.0, value=45.0 ,step=0.1, format="%f")/100.0)

    delta_p = 1/(st.sidebar.number_input(
    "Days a person is pre-symptomatic", 0.0, 10.0, value=1.7 ,step=1.0, format="%f"))
    hosp_los = st.sidebar.number_input("Hospital Length of Stay", value=6, step=1, format="%i")
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
    beta = (
        intrinsic_growth_rate + gamma
    ) / S * (1-relative_contact_rate) # {rate based on doubling time} / {initial S}

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
    beta4 = (
        (alpha+intrinsic_growth_rate)*(intrinsic_growth_rate + (1/infectious_period))
    ) / (alpha*S) 


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
    erie_admit24_line = alt.Chart(erie_df).mark_line(color='red', point=True).encode(
        x='day',
        y='New_admits:Q')

    # Slider and Date
    n_days = st.slider("Number of days to project", 30, 200, 120, 1, "%i")
    as_date = st.checkbox(label="Present result as dates", value=False)


    st.header("""Erie County: Reported Cases, Census and Admissions""")

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
            .mark_line(strokeWidth=3, point=True)
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


    erie_lines = erie_chart(erie_df)
    erie_lines_ip = erie_inpatient(erie_df)
    erie_lines_icu = erie_icu(erie_df)
    erie_lines_vent = erie_vent(erie_df)

    # Bar chart of Erie cases with layer of HERDS DAta Erie
    st.altair_chart(erie_cases_bar + erie_lines, use_container_width=True)

    beta_decay = 0.0

    RateLos = namedtuple("RateLos", ("rate", "length_of_stay"))
    hospitalized=RateLos(hosp_rate, hosp_los)
    icu=RateLos(icu_rate, icu_los)
    ventilated=RateLos(vent_rate, vent_los)


    rates = tuple(each.rate for each in (hospitalized, icu, ventilated))
    lengths_of_stay = tuple(each.length_of_stay for each in (hospitalized, icu, ventilated))


    #############
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


    #####################################
    ## SEIR model with phase adjusted R_0

    s_R, e_R, i_R, r_R = sim_seir_decay(S-2, 1 ,1, 0.0, beta4, gamma2,alpha, n_days, decay1, decay2, decay3, decay4, end_delta)

    susceptible_R, exposed_R, infected_R, recovered_R = s_R, e_R, i_R, r_R

    i_hospitalized_R, i_icu_R, i_ventilated_R = get_dispositions(i_R, rates, regional_hosp_share)

    r_hospitalized_R, r_icu_R, r_ventilated_R = get_dispositions(r_R, rates, regional_hosp_share)

    dispositions_R = (
                i_hospitalized_R + r_hospitalized_R,
                i_icu_R + r_icu_R,
                i_ventilated_R + r_ventilated_R)

    hospitalized_R, icu_R, ventilated_R = (
                i_hospitalized_R,
                i_icu_R,
                i_ventilated_R)


    ##################################################################
    ## SEIR model with phase adjusted R_0 and Disease Related Fatality

    s_D, e_D, i_D, r_D, d_D = sim_seird_decay(S-2, 1, 1 , 0.0, 0.0, beta4, gamma2,alpha, n_days, decay1, decay2, decay3, decay4, end_delta, fatal)

    susceptible_D, exposed_D, infected_D, recovered_D = s_D, e_D, i_D, r_D

    i_hospitalized_D, i_icu_D, i_ventilated_D = get_dispositions(i_D, rates, regional_hosp_share)

    r_hospitalized_D, r_icu_D, r_ventilated_D = get_dispositions(r_D, rates, regional_hosp_share)

    dispositions_D = (
                i_hospitalized_D + r_hospitalized_D,
                i_icu_D + r_icu_D,
                i_ventilated_D + r_ventilated_D)

    hospitalized_D, icu_D, ventilated_D = (
                i_hospitalized_D,
                i_icu_D,
                i_ventilated_D)


    ##################################################################
    ## SEIR model with phase adjusted R_0 and Disease Related Fatality
    ## (Dr. W) Model based on Erie cases with set parameters of extreme social distancing

    s_D, e_D, i_D, r_D, d_D = sim_seird_decay_social(S-2, 1, 1 , 0.0, 0.0, beta4, gamma2,alpha, n_days, decay1, decay2, decay3, decay4, end_delta, fatal)

    susceptible_D, exposed_D, infected_D, recovered_D = s_D, e_D, i_D, r_D

    i_hospitalized_D, i_icu_D, i_ventilated_D = get_dispositions(i_D, rates, regional_hosp_share)

    r_hospitalized_D, r_icu_D, r_ventilated_D = get_dispositions(r_D, rates, regional_hosp_share)

    dispositions_D_socialcases = (
                i_hospitalized_D + r_hospitalized_D,
                i_icu_D + r_icu_D,
                i_ventilated_D + r_ventilated_D)

    hospitalized_D_socialcases, icu_D, ventilated_D = (
                i_hospitalized_D,
                i_icu_D,
                i_ventilated_D)


    ##################################################################
    ## SEIR model with phase adjusted R_0 and Disease Related Fatality
    ## (Gabe) Model based on Erie cases with set parameters of doubling time and social distancing

    s_D, e_D, i_D, r_D, d_D = sim_seird_decay_erie(S-2, 1, 1 , 0.0, 0.0, beta4, gamma2,alpha, n_days, decay1, decay2, decay3, decay4, end_delta, fatal)

    susceptible_D, exposed_D, infected_D, recovered_D = s_D, e_D, i_D, r_D

    i_hospitalized_D, i_icu_D, i_ventilated_D = get_dispositions(i_D, rates, regional_hosp_share)

    r_hospitalized_D, r_icu_D, r_ventilated_D = get_dispositions(r_D, rates, regional_hosp_share)

    d_hospitalized_D, d_icu_D, d_ventilated_D = get_dispositions(d_D, rates, regional_hosp_share)

    dispositions_D_ecases = (
                i_hospitalized_D + r_hospitalized_D+d_hospitalized_D,
                i_icu_D + r_icu_D+d_icu_D,
                i_ventilated_D + r_ventilated_D+d_ventilated_D)

    hospitalized_D_ecases, icu_D, ventilated_D = (
                i_hospitalized_D,
                i_icu_D,
                i_ventilated_D)


    ##################################################################
    ## SEIR model with phase adjusted R_0 and Disease Related Fatality,
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
                                                          end_delta, step2_delta, fatal_hosp,asymptomatic, hosp_rate, q,  l)


    icu_curve= J_n*icu_rate
    vent_curve=J_n*vent_rate

    hosp_rate_n=1.0
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
    dispositions_A_ecases = (
                i_hospitalized_A + r_hospitalized_A+ d_hospitalized_A ,
                i_icu_A+r_icu_A+d_icu_A,
                i_ventilated_A+r_ventilated_A +d_ventilated_A)

    hospitalized_A_ecases, icu_A, ventilated_A = (
                i_hospitalized_A,
                i_icu_A,
                i_ventilated_A)
    ##################################################################
    ## SEIR model with phase adjusted R_0 and Disease Related Fatality,
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
                                                          end_delta, step2_delta, fatal_hosp,asymptomatic, hosp_rate, q,  l,x, p_m1, p_m2, delta_p)

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
    dispositions_P_ecases = (
                i_hospitalized_P + r_hospitalized_P+ d_hospitalized_P ,
                i_icu_P+r_icu_P+d_icu_P,
                i_ventilated_P+r_ventilated_P +d_ventilated_P)

    hospitalized_P_ecases, icu_P, ventilated_P = (
                i_hospitalized_P,
                i_icu_P,
                i_ventilated_P)



    # Individual hospitals selection
    if hosp_options == 'Kaleida':
        col_name1 = {"hosp_kh": "Hospitalized - Kaleida", "icu_kh": "ICU - Kaleida", "vent_kh": "Ventilated - Kaleida"}
        fold_name1 = ["Hospitalized - Kaleida", "ICU - Kaleida", "Ventilated - Kaleida"]
        # Added expanded beds
        col_name2 = {"hosp_kh": "Hospitalized - Kaleida", "icu_kh": "ICU - Kaleida", "vent_kh": "Ventilated - Kaleida", "total_beds":"Total Beds", "icu_beds": "Total ICU Beds"}
        fold_name2 = ["Hospitalized - Kaleida", "ICU - Kaleida", "Ventilated - Kaleida", "Total Beds", "Total ICU Beds"]
        icu_val = 245
        total_beds_val = 1224
        vent_val = 206
        expanded_beds_val = 1319
        expanded_icu_val = 183
        expanded_vent_val = 309
        expanded_beds2_val = 1758
        expanded_icu2_val = 244
        expanded_vent2_val = 412
    if hosp_options == 'ECMC':
        col_name1 = {"hosp_ecmc": "Hospitalized - ECMC", "icu_ecmc": "ICU - ECMC", "vent_ecmc": "Ventilated - ECMC"}
        fold_name1 = ["Hospitalized - ECMC", "ICU - ECMC", "Ventilated - ECMC"]
        col_name2 = {"hosp_ecmc": "Hospitalized - ECMC", "icu_ecmc": "ICU - ECMC", "vent_ecmc": "Ventilated - ECMC", "total_beds":"Total Beds", "icu_beds": "Total ICU Beds"}
        fold_name2 = ["Hospitalized - ECMC", "ICU - ECMC", "Ventilated - ECMC", "Total Beds", "Total ICU Beds"]
        icu_val = 46
        total_beds_val = 518
        vent_val = 0
        expanded_beds_val = 860
        expanded_icu_val = 54
        expanded_vent_val = 0
        expanded_beds2_val = 1146
        expanded_icu2_val = 72
        expanded_vent2_val = 0
    if hosp_options == 'CHS':
        col_name1 = {"hosp_chs": "Hospitalized - CHS", "icu_chs": "ICU - CHS", "vent_chs": "Ventilated - CHS"}
        fold_name1 = ["Hospitalized - CHS", "ICU - CHS", "Ventilated - CHS"]
        col_name2 = {"hosp_chs": "Hospitalized - CHS", "icu_chs": "ICU - CHS", "vent_chs": "Ventilated - CHS", "total_beds":"Total Beds", "icu_beds": "Total ICU Beds"}
        fold_name2 = ["Hospitalized - CHS", "ICU - CHS", "Ventilated - CHS", "Total Beds", "Total ICU Beds"]
        icu_val = 163
        total_beds_val = 887
        vent_val = 0
        expanded_beds_val = 1193
        expanded_icu_val = 111
        expanded_vent_val = 0
        expanded_beds2_val = 1590 
        expanded_icu2_val = 148
        expanded_vent2_val = 0
    if hosp_options == 'RPCI':
        col_name1 = {"hosp_rpci": "Hospitalized - Roswell", "icu_rpci": "ICU - Roswell", "vent_rpci": "Ventilated - Roswell"}
        fold_name1 = ["Hospitalized - Roswell", "ICU - Roswell", "Ventilated - Roswell"]
        col_name2 = {"hosp_rpci": "Hospitalized - Roswell", "icu_rpci": "ICU - Roswell", "vent_rpci": "Ventilated - Roswell", "total_beds":"Total Beds", "icu_beds": "Total ICU Beds"}
        fold_name2 = ["Hospitalized - Roswell", "ICU - Roswell", "Ventilated - Roswell", "Total Beds", "Total ICU Beds"]
        icu_val = 14
        total_beds_val = 133
        vent_val = 0
        expanded_beds_val = 200
        expanded_icu_val = 24
        expanded_vent_val = 0
        expanded_beds2_val = 266
        expanded_icu2_val = 28
        expanded_vent2_val = 0
        

    # Projection days
    plot_projection_days = n_days - 10


    #############
    # # SIR Model
    # # New cases
    projection_admits = build_admissions_df(dispositions)
    # # Census Table
    census_table = build_census_df(projection_admits)
    # ############################

    ############
    # SEIR Model
    # New cases
    projection_admits_e = build_admissions_df(dispositions_e)
    # Census Table
    census_table_e = build_census_df(projection_admits_e)

    #############
    # SEIR Model with phase adjustment
    # New cases
    projection_admits_R = build_admissions_df(dispositions_R)
    # Census Table
    census_table_R = build_census_df(projection_admits_R)

    #############
    # SEIR Model with phase adjustment and Disease Fatality
    # New cases
    projection_admits_D = build_admissions_df(dispositions_D)
    # Census Table
    census_table_D = build_census_df(projection_admits_D)

    #############
    # SEIR Model with phase adjustment and Disease Fatality
    # New cases - using high social distancing
    projection_admits_D_socialcases = build_admissions_df(dispositions_D_socialcases)
    # Census Table
    census_table_D_socialcases = build_census_df(projection_admits_D_socialcases)

    #############
    # SEIR Model with phase adjustment and Disease Fatality
    # New cases - using dynamic doubling time and social distancing
    projection_admits_D_ecases = build_admissions_df(dispositions_D_ecases)
    # Census Table
    census_table_D_ecases = build_census_df(projection_admits_D_ecases)

    #############
    # SEAIJRD Model 
    # New Cases
    projection_admits_A_ecases = build_admissions_df_n(dispositions_A_ecases)
    ## Census Table
    census_table_A_ecases = build_census_df(projection_admits_A_ecases)

        #############
    # SEPAIJRD Model 
    # New Cases
    projection_admits_P_ecases = build_admissions_df_n(dispositions_P_ecases)
    ## Census Table
    census_table_P_ecases = build_census_df(projection_admits_P_ecases)

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


    # Comparison of Single line graph - Hospitalized, ICU, Vent and All
    if model_options == "Inpatient":
        columns_comp = {"hosp": "Hospitalized"}
        fold_comp = ["Hospitalized"]
        capacity_col = {"total_county_beds":"Inpatient Beds"}
        capacity_fol = ["Inpatient Beds"]
    if model_options == "ICU":
        columns_comp = {"icu": "ICU"}
        fold_comp = ["ICU"]
        capacity_col = {"total_county_icu": "ICU Beds"}
        capacity_fol = ["ICU Beds"]
    if model_options == "Ventilated":
        columns_comp = {"vent": "Ventilated"}
        fold_comp = ["Ventilated"]

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
    # Schools 18th
    # Non-essential business 22nd
    vertical = pd.DataFrame({'day': [int1_delta, int2_delta, end_delta, step2_delta]})

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
    #############
    st.header("""Projected Admissions Models for Erie County""")
    st.subheader("Projected number of **daily** COVID-19 admissions for Erie County: SEIR -Phase Adjusted R_0 with Case Fatality and Asymptomatic Component")
    admits_graph_seir = regional_admissions_chart(projection_admits_e, 
            plot_projection_days, 
            as_date=as_date)
    admits_graph = regional_admissions_chart(projection_admits_D, 
            plot_projection_days, 
            as_date=as_date)
    ### High Social Distancing
    admits_graph_highsocial = regional_admissions_chart(projection_admits_D_socialcases, 
            plot_projection_days, 
            as_date=as_date)
    ### Dynamic Doubling Time
    admits_graph_ecases = regional_admissions_chart(projection_admits_D_ecases, 
            plot_projection_days, 
            as_date=as_date)
    ### SEAIJRD
    admits_graph_A= regional_admissions_chart(projection_admits_A_ecases, 
            plot_projection_days, 
            as_date=as_date)
    ### SEPAIJRD
    admits_graph_P= regional_admissions_chart(projection_admits_P_ecases, 
            plot_projection_days, 
            as_date=as_date)
            
    st.altair_chart(
        #admits_graph_seir
        #+ 
        #admits_graph 
        #+ 
        vertical1
        #+ admits_graph_ecases
        + admits_graph_A
        #+ admits_graph_highsocial
        #+ erie_admit24_line
        , use_container_width=True)
    st.subheader("Projected number of **daily** COVID-19 admissions for Erie County: SEIR - Phase Adjusted R_0 with Case Fatality with Asymptomatic, Pre-Symptomatic, and Mask-use")
 
    st.altair_chart(
        #admits_graph_seir
        #+ 
        #admits_graph 
        #+ 
        vertical1
        #+ admits_graph_ecases
        + admits_graph_P
        #+ admits_graph_highsocial
        #+ erie_admit24_line
        , use_container_width=True)


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


    sir = regional_admissions_chart(projection_admits, plot_projection_days, as_date=as_date)
    seir = regional_admissions_chart(projection_admits_e, plot_projection_days, as_date=as_date)
    seir_r = regional_admissions_chart(projection_admits_R, plot_projection_days, as_date=as_date)
    seir_d = regional_admissions_chart(projection_admits_D, plot_projection_days, as_date=as_date)


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
    if model_options == "Inpatient":
        columns_comp_census = {"hosp": "Hospital Census", "total_county_beds":"Inpatient Beds"}
        fold_comp_census = ["Hospital Census", "Inpatient Beds"]
        graph_selection = erie_lines_ip
    if model_options == "ICU":
        columns_comp_census = {"icu": "ICU Census", "total_county_icu": "ICU Beds"}
        fold_comp_census = ["ICU Census", "ICU Beds"]
        graph_selection = erie_lines_icu
    if model_options == "Ventilated":
        columns_comp_census = {"vent": "Ventilated Census"}
        fold_comp_census = ["Ventilated Census"]
        graph_selection = erie_lines_vent

    def ip_census_chart(
        census: pd.DataFrame,
        plot_projection_days: int,
        as_date:bool = False) -> alt.Chart:
        """docstring"""
        census = census.rename(columns=columns_comp_census)

        tooltip_dict = {False: "day", True: "date:T"}
        if as_date:
            census = add_date_column(census.head(plot_projection_days))
            x_kwargs = {"shorthand": "date:T", "title": "Date"}
        else:
            x_kwargs = {"shorthand": "day", "title": "Days from initial infection"}

        return (
            alt
            .Chart(census)
            .transform_fold(fold=fold_comp_census)
            .mark_line(point=False)
            .encode(
                x=alt.X(**x_kwargs),
                y=alt.Y("value:Q", title="Census"),
                color="key:N",
                tooltip=[
                    tooltip_dict[as_date],
                    alt.Tooltip("value:Q", format=".0f", title="Census"),
                    "key:N",
                ],
            )
            .interactive()
        )

    def ip_census_upper(
        census: pd.DataFrame,
        plot_projection_days: int,
        as_date:bool = False) -> alt.Chart:
        """docstring"""
        census = census.rename(columns=columns_comp_census)

        tooltip_dict = {False: "day", True: "date:T"}
        if as_date:
            census = add_date_column(census.head(plot_projection_days))
            x_kwargs = {"shorthand": "date:T", "title": "Date"}
        else:
            x_kwargs = {"shorthand": "day", "title": "Days from initial infection"}

        return (
            alt
            .Chart(census)
            .transform_fold(fold=fold_comp_census)
            .mark_line(point=False)
            .encode(
                x=alt.X(**x_kwargs),
                y=alt.Y("value:Q", title="Census"),
                color="key:N",
                tooltip=[
                    tooltip_dict[as_date],
                    alt.Tooltip("value:Q", format=".0f", title="Census"),
                    "key:N",
                ],
            )
            .interactive()
        )
    ################# Add 0% 10% 20% SD graph of SEIR MODEL ###################

        #, scale=alt.Scale(domain=[0, 40000])
        # scale=alt.Scale(domain=[-5, 9000])
        


    #sir_ip_c = ip_census_chart(census_table, plot_projection_days, as_date=as_date)
    seir_ip_c = ip_census_chart(census_table_e, plot_projection_days, as_date=as_date)
    #seir_r_ip_c = ip_census_chart(census_table_R, plot_projection_days, as_date=as_date)
    seir_d_ip_c = ip_census_chart(census_table_D, plot_projection_days, as_date=as_date)
    ###

    ### 4/20/20 for high social distancing model
    seir_d_ip_highsocial = ip_census_chart(census_table_D_socialcases, plot_projection_days, as_date=as_date)
    ### 4/17/20 for stepwise SD/DT model
    seir_d_ip_ecases = ip_census_chart(census_table_D_ecases, plot_projection_days, as_date=as_date)
    ### 4/22/20 seaijrd
    seir_A_ip_ecases = ip_census_chart(census_table_A_ecases, plot_projection_days, as_date=as_date)
    ### 4/22/20 sepaijrd
    seir_P_ip_ecases = ip_census_chart(census_table_P_ecases, plot_projection_days, as_date=as_date)



    # Chart of Model Comparison for SEIR and Adjusted with Erie County Data
    st.subheader("Comparison of COVID-19 admissions for Erie County: Data vs Model (SEAIJRD)")
    st.altair_chart(
        #alt.layer(seir_ip_c.mark_line())
        #+ alt.layer(seir_d_ip_c.mark_line())
        #+ alt.layer(seir_d_ip_ecases.mark_line())
        #+ 
        alt.layer(seir_A_ip_ecases.mark_line())
        #+ alt.layer(seir_d_ip_highsocial.mark_line())
        + alt.layer(graph_selection)
        + alt.layer(vertical1)
        , use_container_width=True)

    st.subheader("Comparison of COVID-19 admissions for Erie County: Data vs Model (SEPAIJRD)")
    st.altair_chart(
        #alt.layer(seir_ip_c.mark_line())
        #+ alt.layer(seir_d_ip_c.mark_line())
        #+ alt.layer(seir_d_ip_ecases.mark_line())
        #+ 
        alt.layer(seir_P_ip_ecases.mark_line())
        #+ alt.layer(seir_d_ip_highsocial.mark_line())
        + alt.layer(graph_selection)
        + alt.layer(vertical1)
        , use_container_width=True)


    st.header("""Hospital Specific Projected Admissions and Census""")
    # By Hospital Admissions Chart - SEIR model with Phase Adjusted R_0 and Case Fatality
    st.subheader("Projected number of **daily** COVID-19 admissions by Hospital: SEIR model with Phase Adjusted R_0 and Case Fatality")
    st.markdown("Distribution of regional cases based on total bed percentage (CCU/ICU/MedSurg).")

    st.altair_chart(
        hospital_admissions_chart(
            projection_admits_A_ecases, plot_projection_days, as_date=as_date), 
        use_container_width=True)

    ##########################################
    ##########################################
    # Census by hospital
    def hosp_admitted_patients_chart(
        census: pd.DataFrame, 
        as_date:bool = False) -> alt.Chart:
        """docstring"""
        census = census.rename(columns=dict(col_name2))

        tooltip_dict = {False: "day", True: "date:T"}
        if as_date:
            census = add_date_column(census)
            x_kwargs = {"shorthand": "date:T", "title": "Date"}
        else:
            x_kwargs = {"shorthand": "day", "title": "Days from initial infection"}

        return (
            alt
            .Chart(census)
            .transform_fold(fold=fold_name2)
            .mark_line(point=False)
            .encode(
                x=alt.X(**x_kwargs),
                y=alt.Y("value:Q", title="Census"),
                color="key:N",
                tooltip=[
                    tooltip_dict[as_date],
                    alt.Tooltip("value:Q", format=".0f"),
                    "key:N",
                ],
            )
            .interactive()
        )


    # Projected Hospital census SEIR Model with adjusted R_0 and Case Mortality    
    st.subheader("Projected **census** of COVID-19 patients by Hospital, accounting for arrivals and discharges: SEIR Model with Adjusted R_0 and Case Fatality")
    st.altair_chart(
        hosp_admitted_patients_chart(
            census_table_A_ecases, 
            as_date=as_date), 
        use_container_width=True)


    # Erie Graph of Beds
    def bed_lines(
        projection_admits: pd.DataFrame) -> alt.Chart:
        """docstring"""
        
        projection_admits = projection_admits.rename(columns={"total_county_icu": "ICU Beds", 
                                                                "total_county_beds":"Inpatient Beds", 
                                                                "expanded_icu_county":"Expanded ICU 50%",
                                                                "expanded_beds_county":"Expanded Inpatient 50%",
                                                                "expanded_icu_county2":"Expanded ICU 100%",
                                                                "expanded_beds_county2":"Expanded Inpatient 100%"
                                                                })
        
        return(
            alt
            .Chart(projection_admits)
            .transform_fold(fold=["ICU Beds", 
                                    "Inpatient Beds", 
                                    "Expanded ICU 50%",
                                    "Expanded Inpatient 50%",
                                    "Expanded ICU 100%",
                                    "Expanded Inpatient 100%"
                                    ])
            .mark_line(point=False)
            .encode(
                x=alt.X("day", title="Date"),
                y=alt.Y("value:Q", title="Erie County Bed Census"),
                color="key:N",
                tooltip=[alt.Tooltip("value:Q", format=".0f"),"key:N"]
            )
            .interactive()
        )


    ##########################################################
    ##########################################################
    ###########            PPE            ####################
    ##########################################################
    ##########################################################
    st.header("Projected PPE Needs for Erie County")
    def ppe_chart(
        census: pd.DataFrame,
        as_date:bool = False) -> alt.Chart:
        """docstring"""
        census = census.rename(columns={'ppe_mean_mild': 'Mean PPE needs - mild cases', 'ppe_mean_severe': 'Mean PPE needs - severe cases'})
        tooltip_dict = {False: "day", True: "date:T"}
        if as_date:
            census = add_date_column(census)
            x_kwargs = {"shorthand": "date:T", "title": "Date"}
        else:
            x_kwargs = {"shorthand": "day", "title": "Days from initial infection"}

        return (
            alt
            .Chart(census)
            .transform_fold(fold=['Mean PPE needs - mild cases', 'Mean PPE needs - severe cases'])
            .mark_line(point=False)
            .encode(
                x=alt.X(**x_kwargs),
                y=alt.Y("value:Q", title="Projected PPE needs per day"),
                color="key:N",
                tooltip=[
                    tooltip_dict[as_date],
                    alt.Tooltip("value:Q", format=".0f", title="PPE Needs"),
                    "key:N",
                ],
            )
            .interactive()
        )

    # , scale=alt.Scale(domain=[0, 450000])
        
    # SEIR Model with adjusted R_0 with Case Fatality - PPE predictions
    st.subheader("Projected personal protective equipment needs for mild and severe cases of COVID-19: SEIR Model with Adjutes R_0 and Case Fatality")

    ppe_graph = ppe_chart(census_table_A_ecases, as_date=as_date)

    st.altair_chart(alt.layer(ppe_graph.mark_line()) + alt.layer(vertical1), use_container_width=True)

    # Recovered/Infected/Fatality table
    st.subheader("Infected,recovered,and fatal individuals in the **region** across time")

    def additional_projections_chart(i: np.ndarray, r: np.ndarray, d: np.ndarray) -> alt.Chart:
        dat = pd.DataFrame({"Infected": i, "Recovered": r, "Fatal":d})

        return (
            alt
            .Chart(dat.reset_index())
            .transform_fold(fold=["Infected"])
            .mark_line(point=False)
            .encode(
                x=alt.X("index", title="Days from initial infection"),
                y=alt.Y("value:Q", title="Case Volume"),
                tooltip=["key:N", "value:Q"], 
                color="key:N"
            )
            .interactive()
        )

    recov_infec = additional_projections_chart(i_D, r_D, d_D)

    


    def death_chart(i: np.ndarray, r: np.ndarray, d: np.ndarray) -> alt.Chart:
        dat = pd.DataFrame({"Infected": i, "Recovered": r, "Fatal":d})

        return (
            alt
            .Chart(dat.reset_index())
            .transform_fold(fold=["Fatal"])
            .mark_bar()
            .encode(
                x=alt.X("index", title="Days from initial infection"),
                y=alt.Y("value:Q", title="Case Volume"),
                tooltip=["key:N", "value:Q"], 
                color=alt.value('red')
            )
            .interactive()
        )

    deaths = death_chart(i_D, r_D, d_D)

    st.altair_chart(deaths + recov_infec, use_container_width=True)



    total_fatalities=max(d_D)
    infection_total_t=max(d_D)+max(r_D)
    st.markdown(
        """There is a projected number of **{total_fatalities:.0f}** fatalities due to COVID-19.""".format(
            total_fatalities=total_fatalities 
        ))

    st.markdown("""There is a projected number of **{infection_total_t:.0f}** infections due to COVID-19.""".format(
            infection_total_t=infection_total_t
        )
                )

    AAA=beta4*(1/gamma2)*S
    R2=AAA*(1-decay2)
    R3=AAA*(1-decay3)
    R4=AAA*(1-decay4)

    st.markdown("""The initial $R_0$ is **{AAA:.1f}** with a $$\\beta$$ of **{beta4:.2f}**, the $R_e$ after 2 weeks is **{R2:.1f}** and the $R_e$ after 3 weeks to end of social distancing is **{R3:.1f}**.
    After reducing social distancing the $R_e$ is **{R4:.1f}**
                This is based on a doubling rate of **{doubling_time:.0f}**
                and the calculation of the [basic reproduction number](https://www.sciencedirect.com/science/article/pii/S2468042719300491).""".format(
            AAA=AAA,
            beta4=beta4*S,
            R2=R2,
            R3=R3,
            R4=R4,
            doubling_time=doubling_time
        )
                )


    st.subheader("Extension of the SEIR model to include asymptomatic and direct hospitalization components")
    if st.checkbox("Show more about the assumptions and specifications of the SEAIJRD model"):
        st.subheader(
        "[Deterministic SEIR model with asymptomatic, hospitalizations, and fatality components](https://www.tandfonline.com/doi/full/10.1080/23737867.2018.1509026)")
        st.markdown(
        """The model consists of individuals who are either _Susceptible_ ($S$), _Exposed_ ($E$),
    _Asymptomatic_ ($A$),_Infected_ ($I$),
     _Hospitalized_ ($J$), _Recovered_ ($R$), or _Fatal_ ($D$).
    The epidemic proceeds via a growth and decline process."""
    )
        st.markdown("""The system of differential equations are given by the following 7 equations.""")

        st.latex(r'''\frac{dS}{dt}=-\rho_t \beta S[qI+lJ+A]/N''')
        st.latex(r'''\frac{dE}{dt}=\rho_t \beta S[qI+lJ+A]/N - \alpha E''')
        st.latex(r'''\frac{dA}{dt}= (1-z)\alpha E - \gamma_1 A''')
        st.latex(r'''\frac{dI}{dt}= z\alpha E - \gamma_1 I-h I''')
        st.latex(r'''\frac{dJ}{dt}= h I - \gamma_2 J''')
        st.latex(r'''\frac{dR}{dt}=\gamma_1(A+I) + (1-f)\gamma_2 J''')
        st.latex(r'''\frac{dD}{dt}=f \gamma_2 J''')

        st.markdown(
        """where $\gamma_1$ is $1/mean\ infectious\ rate$,$\gamma_2$ is $1/mean\ hospital\ day\ rate$, $$\\alpha$$ is $1/mean\ incubation\ period$, $$\\rho$$ is the rate of social distancing at time $t$,
    $$\\beta$$ is the rate of transmission, $f$ is the hospital fatality rate, $h$ is the hospitalization rate, $z$ is the symptomatic rate, $q$ is the isolation rate for the symptomati, $l$ is the isolation rate for hospitalized, and $z$ is the symptomatic rate (where $(1-z)$ is the asymptomatic rate). More information, including parameter specifications and reasons for model choice can be found
    [here]("https://github.com/gabai/stream_KH/wiki).  $R_0$ was calculated using the [next generation matrix method](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2871801/).""")
        st.latex(r'''R_0=\beta [ \frac{(1-z)}{\gamma_1 }+ \frac{zq}{\gamma_1 + \alpha} + \frac{z \alpha l}{( \gamma_1 + \alpha )\gamma_2}]''')

        st.markdown("""Note that a number of assumptions are made with deterministic compartmental models. First, we are assuming a large, closed population with no births or deaths.
    Second, within the time period, immunity to the disease is acquired. Third, the susceptible and infected subpopulations are dispersed homogeneously in geographic space.
    In addition to the model assumptions noted here, the model is limited by uncertainty related to parameter choice.
    Parameters are measured independently from the model, which is hard to do in the midst of an outbreak.
    Early reports from other geographic locations have allowed us to estimate this model.
    However, parameters can be different depending on population characteristics and can vary over periods of the outbreak.
    Therefore, interpreting the results can be difficult.""")


    st.subheader("Asymptomatic, Symptomatic,Hospitalized,and Fatal individuals in the **region** across time")

    def additional_projections_chart(a:np.ndarray, i:np.ndarray, j:np.ndarray,d:np.ndarray)  -> alt.Chart:
        dat = pd.DataFrame({"Asymptomatic":a,"Infected":i, "Hospitalized":j,"Fatal":d})

        return (
            alt
            .Chart(dat.reset_index())
            .transform_fold(fold=["Asymptomatic","Symptomatic", "Hospitalized","Fatal"])
            .mark_line(point=False)
            .encode(
                x=alt.X("index", title="Days from initial infection"),
                y=alt.Y("value:Q", title="Case Volume"),
                tooltip=["key:N", "value:Q"], 
                color="key:N"
            )
            .interactive()
        )

    st.altair_chart(additional_projections_chart(A_n, I_n, J_n, D_n), use_container_width=True)

    st.subheader("Presymptomatic, Asymptomatic, Symptomatic,Hospitalized,and Fatal individuals in the **region** across time")

    def additional_projections_chart(p:np.ndarray,a:np.ndarray, i:np.ndarray, j:np.ndarray,d:np.ndarray)  -> alt.Chart:
        dat = pd.DataFrame({"Presymptomatic":p,"Asymptomatic":a,"Symptomatic":i, "Hospitalized":j,"Fatal":d})

        return (
            alt
            .Chart(dat.reset_index())
            .transform_fold(fold=["Presymptomatic","Asymptomatic","Symptomatic", "Hospitalized","Fatal"])
            .mark_line(point=False)
            .encode(
                x=alt.X("index", title="Days from initial infection"),
                y=alt.Y("value:Q", title="Case Volume"),
                tooltip=["key:N", "value:Q"], 
                color="key:N"
            )
            .interactive()
        )

    st.altair_chart(additional_projections_chart(P_p,A_p, I_p, J_p, D_p), use_container_width=True)


    ############################### prevalence and incidence ###########################
    # https://www.tandfonline.com/doi/full/10.1057/hs.2015.2
    ####################################################################################
    st.subheader("Prevalence and Incidence Across Time")

    st.markdown("""Incidence is measured as the number of new cases daily, (from compartments A, I) prevalence is measured
as the population infected with the disease (A,I,J) daily.""")

    ### incidence
    dispositions_inc= (A_n+I_n+J_n+R_n+D_n)
    dispositions_inc=pd.DataFrame(dispositions_inc, columns=['newcases']) 
    dispositions_inc2 = dispositions_inc.iloc[:-1, :] - dispositions_inc.shift(1)
    dispositions_inc2["day"] = range(dispositions_inc2.shape[0])
    dispositions_inc2["TotalCases"]=S_n
    dispositions_inc2.at[0,'newcases']=0
    dispositions_inc2["incidencerate"]=dispositions_inc2['newcases']/dispositions_inc2['TotalCases']

    # total number of new cases daily/total number of people disease free at the start of the day

    ### prevalence

    dispositions_prev=(A_n+I_n+J_n)
    dispositions_prev=pd.DataFrame(dispositions_prev, columns=['cumucases']) 
    dispositions_prev["day"] = range(dispositions_prev.shape[0])    
    dispositions_prev["TotalCases"]=1400000.0
    dispositions_prev["pointprevalencerate"]=dispositions_prev['cumucases']/dispositions_prev['TotalCases']
    #total number of infected people during that day/ total number in population

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



          

    
    
