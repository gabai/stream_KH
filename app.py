# Modified version for Erie County, New York
# Contact: ganaya@buffalo.edu

from functools import reduce
from typing import Generator, Tuple, Dict, Any, Optional
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)


# Models and base functions
########
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
    
def sim_sir_df(p) -> pd.DataFrame:
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

def build_admissions_df(dispositions) -> pd.DataFrame:
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
        if counter == 8: break
    
    
    # New cases
    projection_admits = projection.iloc[:-1, :] - projection.shift(1)
    projection_admits["day"] = range(projection_admits.shape[0])
    return projection_admits

def build_census_df(projection_admits: pd.DataFrame) -> pd.DataFrame:
    """ALOS for each category of COVID-19 case (total guesses)"""
    #n_days = np.shape(projection_admits)[0]
    los_dict = {
    "hosp": hosp_los, "icu": icu_los, "vent": vent_los,
    "hosp_bgh": hosp_los, "icu_bgh": icu_los, "vent_bgh": vent_los,
    "hosp_ecmc": hosp_los, "icu_ecmc": icu_los, "vent_ecmc": vent_los,
    "hosp_mercy": hosp_los, "icu_mercy": icu_los, "vent_mercy": vent_los,
    "hosp_mfsh": hosp_los, "icu_mfsh": icu_los, "vent_mfsh": vent_los,
    "hosp_och": hosp_los, "icu_och": icu_los, "vent_och": vent_los,
    "hosp_rpci": hosp_los, "icu_rpci": icu_los, "vent_rpci": vent_los,
    "hosp_sch": hosp_los, "icu_sch": icu_los, "vent_sch": vent_los,
    "hosp_scsjh": hosp_los, "icu_scsjh": icu_los, "vent_scsjh": vent_los
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
    "hosp_bgh", "icu_bgh", "vent_bgh", 
    "hosp_ecmc", "icu_ecmc", "vent_ecmc",
    "hosp_mercy", "icu_mercy", "vent_mercy",
    "hosp_mfsh", "icu_mfsh", "vent_mfsh",
    "hosp_och", "icu_och", "vent_och",
    "hosp_rpci", "icu_rpci", "vent_rpci",
    "hosp_sch", "icu_sch", "vent_sch",
    "hosp_scsjh", "icu_scsjh", "vent_scsjh"
    ]]
    
    census_df['total_county_icu'] = icu_county
    census_df['expanded_icu_county'] = expanded_icu_county
    census_df['total_county_beds'] = beds_county
    census_df['expanded_beds_county'] = expanded_beds_county
    census_df['icu_beds'] = icu_val
    census_df['total_beds'] = total_beds_val
    census_df['expanded_beds'] = expanded_beds_val
    census_df['expanded_icu_beds'] = expanded_icu_val
    
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
    
    # census_df = census_df.rename(
        # columns={
            # disposition: f"{disposition}"
            # for disposition in ("hosp", "icu", "vent")
        # }
    # )
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
    start = datetime.now()
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

    
# General Variables
today = date.today()
fdate = date.today().strftime("%m-%d-%Y")
time = time.strftime("%H:%M:%S")

### Extract data for US
# URL
# 1 Request URL
#url = 'https://www.cdc.gov/coronavirus/2019-ncov/cases-updates/cases-in-us.html'
#page = requests.get(url)
# 2 Parse HTML content
#soup = BeautifulSoup(page.text, 'html.parser')
# 3 Extract cases data
#cdc_data = soup.find_all(attrs={"class": "card-body bg-white"})
# Create dataset of extracted data
#df = []
#for ul in cdc_data:
#    for li in ul.find_all('li'):
#        df.append(li.text.replace('\n', ' ').strip())
### US specific cases - CDC
#cases_us = df[0].split(': ')
# Replace + and , for numeric values
#cases_us = int(cases_us[1].replace(',', ''))
# Total US deaths - CDC
#deaths_us = df[1].split(': ')
#deaths_us = pd.to_numeric(deaths_us[1])
# Calculate mortality rate
#us_MR = round((deaths_us/cases_us)*100,2)
# Create table
#data = {'Cases': [cases_us],
#       'Deaths': [deaths_us],
#       'Calculated Mortality Rate': [us_MR]}
#us_data = pd.DataFrame(data)

# Extract data for NY State cases
# URL
# 1 Request URL
#url = 'https://coronavirus.health.ny.gov/county-county-breakdown-positive-cases'
#page = requests.get(url)
# 2 Parse HTML content
#soup = BeautifulSoup(page.text, 'html.parser')
# 3 Get the table having the class country table
#table = soup.find("div", attrs={'class':"wysiwyg--field-webny-wysiwyg-body"})
#table_data = table.find_all("td")
# Get all the headings of Lists
#df = []
#for i in range(0,len(table_data)):
#    for td in table_data[i]:
#        df.append(table_data[i].text.replace('\n', ' ').strip())
        


#counties = pd.DataFrame([])
#for i in range(0, len(df), 2):
#    counties = counties.append(pd.DataFrame({'County': df[i], 'Cases': df[i+1]},
#                                              index =[0]), ignore_index=True)


# NY state Modification for Counties and State Tables
#NYC = counties[counties['County']=='New York City'].reset_index()
#NYS = counties[counties['County']=='Total Number of Positive Cases'].reset_index()
#erie = counties[counties['County']=='Erie'].reset_index()
#counties_cases = counties[~(counties['County']=='New York City') & ~(counties['County']=='Total Number of Positive Cases')]
# Remove comma
#NYC['Cases'] = pd.to_numeric(NYC['Cases'].str.replace(',', ''))
#NYS['Cases'] = pd.to_numeric(NYS['Cases'].str.replace(',', ''))
# Extract value
#cases_nys = NYC.Cases[0]
#cases_nyc = NYS.Cases[0]
#cases_erie = pd.to_numeric(erie.Cases[0])

# Create table
#data = {'County': ['Erie', 'New York City', 'New York State'],
#       'Cases': [cases_erie, cases_nyc, cases_nys]}
#ny_data = pd.DataFrame(data)

# Adding ICU bed for county
icu_county = 246
expanded_icu_county = 369
beds_county = 2380
expanded_beds_county = 3570
# PPE Values
ppe_mild_val_lower = 14
ppe_mild_val_upper = 15
ppe_severe_val_lower = 15
ppe_severe_val_upper = 24

# List of Hospitals
hosp_list = ['bgh', 'ecmc', 'mercy', 'mfsh', 'och', 'rpci', 'sch', 'scsjh']
groups = ['hosp', 'icu', 'vent']

# Hospital Bed Sharing Percentage
data = {
    #'Bed Type': ['CCU', 'ICU', 'MedSurg', 'Hosp_share'],
    'BGH' : [0.34, 0.34, 0.26, 0.27],
    'ECMC': [0.14, 0.20, 0.17, 0.17], 
    'Mercy': [0.21, 0.17, 0.18, 0.18],
    'MFSH' : [0.12, 0.6, 0.15, 0.14],
    'OCH' : [0.0, 0.12, 0.05, 0.05],
    'RPCI': [0.0, 0.09, 0.06, 0.06],
    'SCH' : [0.12, 0.10, 0.13, 0.13],
    'SCSJH': [0.08, 0.04, 0.06, 0.06]
}
bed_share = pd.DataFrame(data)

# Erie's admission for comparison with current curve
data_dict = {
    "Admissions": [0,0,0,0,0,0,0,0,0,0,
                   0,0,0,0,0,0,0,0,1,4,
                   6,6,8,21,29],
    "Cases": [0,0,0,0,0,0,0,0,0,0,
              0,0,0,0,0,3,7,20,34,47,
              61,96,114,121,146],
    "Deaths": [0,0,0,0,0,0,0,0,0,0,
               0,0,0,0,0,0,0,0,0,0,
               0,0,0,0,2],
    "Date": ['3/1/20', '3/2/20', '3/3/20', '3/4/20', '3/5/20', '3/6/20', '3/7/20', '3/8/20', '3/9/20', '3/10/20',
        '3/11/20', '3/12/20', '3/13/20', '3/14/20', '3/15/20', '3/16/20', '3/17/20', '3/18/20', '3/19/20', '3/20/20', 
        '3/21/20', '3/22/20', '3/23/20', '3/24/20', '3/25/20'],
    "day": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    }
erie_df = pd.DataFrame.from_dict(data_dict)
erie_df['Date'] = pd.to_datetime(erie_df['Date'])


# Populations and Infections
buffalo = 258612
tonawanda = 14904
cheektowaga = 87018
amherst = 126082
erie = 1500000
cases_erie = 146
S_default = erie
known_infections = 146
known_cases = 29
#initial_infections = 47
regional_hosp_share = 1.0
S = erie


# Widgets
hosp_options = st.sidebar.radio(
    "Hospitals", ('BGH', 'ECMC', 'Mercy', 'MFSH', 'OCH', 'RPCI', 'SCH', 'SCSJH'))

current_hosp = st.sidebar.number_input(
    "Currently Hospitalized COVID-19 Patients", value=known_cases, step=1, format="%i"
)

doubling_time = st.sidebar.number_input(
    "Doubling Time (days)", value=5, step=1, format="%i"
)

relative_contact_rate = st.sidebar.number_input(
    "Social distancing (% reduction in social contact)", 0, 100, value=30, step=5, format="%i"
)/100.0

hosp_rate = (
    st.sidebar.number_input("Hospitalization %", 0.0, 100.0, value=10.0, step=1.0, format="%f")
    / 100.0
)

icu_rate = (
    st.sidebar.number_input("ICU %", 0.0, 100.0, value=4.0, step=0.5, format="%f") / 100.0
)

vent_rate = (
    st.sidebar.number_input("Ventilated %", 0.0, 100.0, value=2.0, step=0.5, format="%f")
    / 100.0
)

incubation_period =(
    st.sidebar.number_input("Incubation Period", 0.0, 12.0, value=5.8, step=0.1, format="%f")
)

recovery_days =(
    st.sidebar.number_input("Recovery Period", 0.0, 21.0, value=11.0 ,step=0.1, format="%f")
)


hosp_los = st.sidebar.number_input("Hospital Length of Stay", value=10, step=1, format="%i")
icu_los = st.sidebar.number_input("ICU Length of Stay", value=9, step=1, format="%i")
vent_los = st.sidebar.number_input("Ventilator Length of Stay", value=7, step=1, format="%i")

# regional_hosp_share = (
   # st.sidebar.number_input(
       # "Hospital Bed Share (%)", 0.0, 100.0, value=100.0, step=1.0, format="%f")
   # / 100.0
# )

# S = st.sidebar.number_input(
   # "Regional Population", value=S_default, step=100000, format="%i"
# )

initial_infections = st.sidebar.number_input(
    "Currently Known Regional Infections (only used to compute detection rate - does not change projections)", value=known_infections, step=10, format="%i"
)

total_infections = current_hosp / regional_hosp_share / hosp_rate
detection_prob = initial_infections / total_infections

S, I, R = S, initial_infections / detection_prob, 0

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

st.title("COVID-19 Hospital Impact Model for Epidemics - Modified for Erie County")
st.markdown(
    """*This tool was initially developed by the [Predictive Healthcare team](http://predictivehealthcare.pennmedicine.org/) at
Penn Medicine and has been modified for our community. 

We have modified the model to include:
- "Exposure", creating a SEIR model. 
- We present distribution of cases by each hospital based on bed-share percentage.
- Protective Equipment Equipment needs for Erie County, NY.

For questions about this page, contact ganaya@buffalo.edu. """)


st.markdown(
    """
The first graph includes the COVID-19 cases in Erie County, NY. A county wide and hospital based analysis using both SIR and SEIR models is presented afterwards. 
Each Hospital is represented as a percent from the total bed-share distribution (CCU, ICU, MedSurg - Pending update of 'expanded' per hospital beds).

An initial doubling time of **{doubling_time}** days and a recovery time of **{recovery_days}** days imply an $R_0$ of 
**{r_naught:.2f}**.

**Mitigation**: A **{relative_contact_rate:.0%}** reduction in social contact after the onset of the 
outbreak reduces the doubling time to **{doubling_time_t:.1f}** days, implying an effective $R_t$ of **${r_t:.2f}$**.""".format(
        total_infections=total_infections,
        current_hosp=current_hosp,
        hosp_rate=hosp_rate,
        S=S,
        regional_hosp_share=regional_hosp_share,
        initial_infections=initial_infections,
        detection_prob=detection_prob,
        recovery_days=recovery_days,
        r_naught=r_naught,
        doubling_time=doubling_time,
        relative_contact_rate=relative_contact_rate,
        r_t=r_t,
        doubling_time_t=doubling_time_t
    )
)

# The estimated number of currently infected individuals in Erie County is **{total_infections:.0f}**. The **{initial_infections}** 
# confirmed cases in the region imply a **{detection_prob:.0%}** rate of detection. This is based on current inputs for 
# Hospitalizations (**{current_hosp}**), Hospitalization rate (**{hosp_rate:.0%}**) and Region size (**{S}**). 
# All credit goes to the PH team at Penn Medicine. We have adapted the code based on our current regional cases, county population and hospitals.
# The **{initial_infections}** confirmed cases in the region imply a **{detection_prob:.0%}** rate of detection.

#st.subheader("Cases of COVID-19 in the United States")
# Table of cases in the US
#st.table(us_data)
# Table of cases in NYS
#st.subheader("Cases of COVID-19 in New York State")
#counties.sort_values(by=['Cases'], ascending=False)
#st.table(ny_data)
#st.subheader("Cases of COVID-19 in Erie County")



st.subheader("""Reported cases and admissions in Erie County""")


st.markdown(
    """Erie county has reported **{cases_erie:.0f}** cases of COVID-19.""".format(
        cases_erie=cases_erie
    )
)
#st.markdown(""" """)
#st.markdown(""" """)




erie_cases_bar = alt.Chart(erie_df).mark_bar().encode(
    x='Date:T',
    y='Cases:Q')
erie_admit_line = alt.Chart(erie_df).mark_line(color='red').encode(
    x='Date:T',
    y='Admissions:Q')

st.altair_chart(alt.layer(erie_cases_bar + erie_admit_line), use_container_width=True)

#fig = go.Figure(data=[go.Table(header=dict(values=['Total Cases', 'Total Deaths', 'Mortality Rate %']),
#                 cells=dict(values=[cases_us, deaths_us, us_MR]))
#                     ])
#st.plotly_chart(fig)


if st.checkbox("Show more info about this tool"):
    st.subheader(
        "[Discrete-time SIR modeling](https://mathworld.wolfram.com/SIRModel.html) of infections/recovery"
    )
    st.markdown(
        """The model consists of individuals who are either _Susceptible_ ($S$), _Infected_ ($I$), or _Recovered_ ($R$).
The epidemic proceeds via a growth and decline process. This is the core model of infectious disease spread and has been in use in epidemiology for many years."""
    )
    st.markdown("""The dynamics are given by the following 3 equations.""")

    st.latex("S_{t+1} = (-\\beta S_t I_t) + S_t")
    st.latex("I_{t+1} = (\\beta S_t I_t - \\gamma I_t) + I_t")
    st.latex("R_{t+1} = (\\gamma I_t) + R_t")

    st.markdown(
        """To project the expected impact to Erie County Hospitals, we estimate the terms of the model.
To do this, we use a combination of estimates from other locations, informed estimates based on logical reasoning, and best guesses from the American Hospital Association.
### Parameters
The model's parameters, $\\beta$ and $\\gamma$, determine the virulence of the epidemic.
$$\\beta$$ can be interpreted as the _effective contact rate_:
""")
    st.latex("\\beta = \\tau \\times c")

    st.markdown(
"""which is the transmissibility ($\\tau$) multiplied by the average number of people exposed ($$c$$).  The transmissibility is the basic virulence of the pathogen.  The number of people exposed $c$ is the parameter that can be changed through social distancing.
$\\gamma$ is the inverse of the mean recovery time, in days.  I.e.: if $\\gamma = 1/{recovery_days}$, then the average infection will clear in {recovery_days} days.
An important descriptive parameter is the _basic reproduction number_, or $R_0$.  This represents the average number of people who will be infected by any given infected person.  When $R_0$ is greater than 1, it means that a disease will grow.  Higher $R_0$'s imply more rapid growth.  It is defined as """.format(recovery_days=int(recovery_days)    , c='c'))
    st.latex("R_0 = \\beta /\\gamma")

    st.markdown("""
$R_0$ gets bigger when
- there are more contacts between people
- when the pathogen is more virulent
- when people have the pathogen for longer periods of time
A doubling time of {doubling_time} days and a recovery time of {recovery_days} days imply an $R_0$ of {r_naught:.2f}.
#### Effect of social distancing
After the beginning of the outbreak, actions to reduce social contact will lower the parameter $c$.  If this happens at 
time $t$, then the number of people infected by any given infected person is $R_t$, which will be lower than $R_0$.  
A {relative_contact_rate:.0%} reduction in social contact would increase the time it takes for the outbreak to double, 
to {doubling_time_t:.2f} days from {doubling_time:.2f} days, with a $R_t$ of {r_t:.2f}.
#### Using the model
We need to express the two parameters $\\beta$ and $\\gamma$ in terms of quantities we can estimate.
- $\\gamma$:  the CDC is recommending 14 days of self-quarantine, we'll use $\\gamma = 1/{recovery_days}$.
- To estimate $$\\beta$$ directly, we'd need to know transmissibility and social contact rates.  since we don't know these things, we can extract it from known _doubling times_.  The AHA says to expect a doubling time $T_d$ of 7-10 days. That means an early-phase rate of growth can be computed by using the doubling time formula:
""".format(doubling_time=doubling_time,
           recovery_days=recovery_days,
           r_naught=r_naught,
           relative_contact_rate=relative_contact_rate,
           doubling_time_t=doubling_time_t,
           r_t=r_t)
    )
    st.latex("g = 2^{1/T_d} - 1")

    st.markdown(
        """
- Since the rate of new infections in the SIR model is $g = \\beta S - \\gamma$, and we've already computed $\\gamma$, $\\beta$ becomes a function of the initial population size of susceptible individuals.
$$\\beta = (g + \\gamma)$$.

### Initial Conditions

- The total size of the susceptible population will be the entire catchment area for Erie County.
- Erie = {erie}
- Buffalo General Hospital with 25% of bed share, calculated based on 456 CCU/ICU/MedSurg beds. Excluded MRU beds. 
- Erie County Medical Center with 16% of beds share, calculated based on 285 CCU/ICU/MedSurg beds. Excluded burns care, chemical dependence, pediatric, MRU, prisoner and psychiatric beds. 
- Mercy Hospital with 17% of bed share, calculated based on 306 CCU/ICU/MedSurg beds. Excluded maternity beds, neonatal, pediatric and MRU beds. 
- Millard Fillmore Suburban Hospital with 13% of bed share, calculated based on 227 CCU/ICU/MedSurg beds. Excluded maternity and neonatal beds. 
- Oishei Hospital Children's with 5% of bed share, calculated based on 89 ICU/MedSurg beds. Excluded bone marrow transplant beds and neonatal beds. 
- Roswell Park Cancer Institute with 6% of bed share, calculated based on 110 ICU/MedSurg beds. Excluded bone marrow transplant beds and pediatric beds. 
- Sisters of Charity Hospital with 12% of bed share, calculated based on 215 CCU/ICU/MedSurg beds. Excluded maternity, neonatal and MRU beds.
- Sisters of Charity St. Joeseph Hospital with 6% of beds share, calculated based on 103 CCU/ICU/MedSurg beds. """.format(
            erie=erie))


n_days = st.slider("Number of days to project", 30, 300, 120, 1, "%i")
as_date = st.checkbox(label="Present result as dates", value=False)

beta_decay = 0.0
s, i, r = sim_sir(S, I, R, beta, gamma, n_days)

RateLos = namedtuple("RateLos", ("rate", "length_of_stay"))
hospitalized=RateLos(hosp_rate, hosp_los)
icu=RateLos(icu_rate, icu_los)
ventilated=RateLos(vent_rate, vent_los)


rates = tuple(each.rate for each in (hospitalized, icu, ventilated))
lengths_of_stay = tuple(each.length_of_stay for each in (hospitalized, icu, ventilated))

### SIR model

s_v, i_v, r_v = sim_sir(S, total_infections, recovered, beta, gamma, n_days)
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

### SEIR model
alpha = 1 / incubation_period
exposed_start=beta*S*total_infections
s_e, e_e, i_e, r_e = sim_seir(S, exposed_start, total_infections , recovered, beta, gamma,alpha, n_days)
### Issue here
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

# Individual hospitals selection

if hosp_options == 'BGH':
    col_name1 = {"hosp_bgh": "Hospitalized - BGH", "icu_bgh": "ICU - BGH", "vent_bgh": "Ventilated - BGH"}
    fold_name1 = ["Hospitalized - BGH", "ICU - BGH", "Ventilated - BGH"]
    # Added expanded beds
    #col_name2 = {"hosp_bgh": "Hospitalized - BGH", "icu_bgh": "ICU - BGH", "vent_bgh": "Ventilated - BGH", "total_beds":"Total Beds", "icu_beds": "Total ICU Beds"}
    #fold_name2 = ["Hospitalized - BGH", "ICU - BGH", "Ventilated - BGH", "Total Beds", "Total ICU Beds"]
    col_name2 = {"hosp_bgh": "Hospitalized - BGH", "icu_bgh": "ICU - BGH", "vent_bgh": "Ventilated - BGH",
        "expanded_beds":"Expanded IP Beds", "expanded_icu_beds": "Expanded ICU Beds"}
    fold_name2 = ["Hospitalized - BGH", "ICU - BGH", "Ventilated - BGH", "Expanded IP Beds", "Expanded ICU Beds"]
    #col_name3 = {"ppe_mild_d_bgh": "PPE Mild Cases - Lower Range", "ppe_mild_u_bgh": "PPE Mild Cases - Upper Range", 
    #"ppe_severe_d_bgh": "PPE Severe Cases - Lower Range", "ppe_severe_u_bgh": "PPE Severe Cases - Upper Range"}
    #fold_name3 = ["PPE Mild Cases - Lower Range", "PPE Mild Cases - Upper Range", "PPE Severe Cases - Lower Range", "PPE Severe Cases - Upper Range"]
    icu_val = 53
    total_beds_val = 456
    expanded_beds_val = 684
    expanded_icu_val = 80
if hosp_options == 'ECMC':
    col_name1 = {"hosp_ecmc": "Hospitalized - ECMC", "icu_ecmc": "ICU - ECMC", "vent_ecmc": "Ventilated - ECMC"}
    fold_name1 = ["Hospitalized - ECMC", "ICU - ECMC", "Ventilated - ECMC"]
    #col_name2 = {"hosp_ecmc": "Hospitalized - ECMC", "icu_ecmc": "ICU - ECMC", "vent_ecmc": "Ventilated - ECMC", "total_beds":"Total Beds", "icu_beds": "Total ICU Beds"}
    #fold_name2 = ["Hospitalized - ECMC", "ICU - ECMC", "Ventilated - ECMC", "Total Beds", "Total ICU Beds"]
    col_name2 = {"hosp_ecmc": "Hospitalized - ECMC", "icu_ecmc": "ICU - ECMC", "vent_ecmc": "Ventilated - ECMC", 
        "expanded_beds":"Expanded IP Beds", "expanded_icu_beds": "Expanded ICU Beds"}
    fold_name2 = ["Hospitalized - ECMC", "ICU - ECMC", "Ventilated - ECMC", "Expanded IP Beds", "Expanded ICU Beds"]
    col_name3 ={"ppe_mild_d_ecmc": "PPE Mild Cases - Lower Range", "ppe_mild_u_ecmc": "PPE Mild Cases - Upper Range", 
    "ppe_severe_d_ecmc": "PPE Severe Cases - Lower Range", "ppe_severe_u_ecmc": "PPE Severe Cases - Upper Range"}
    fold_name3 = ["PPE Mild Cases - Lower Range", "PPE Mild Cases - Upper Range", "PPE Severe Cases - Lower Range", "PPE Severe Cases - Upper Range"]
    icu_val = 34
    total_beds_val = 285
    expanded_beds_val = 428
    expanded_icu_val = 51
if hosp_options == 'Mercy':
    col_name1 = {"hosp_mercy": "Hospitalized - Mercy", "icu_mercy": "ICU - Mercy", "vent_mercy": "Ventilated - Mercy"}
    fold_name1 = ["Hospitalized - Mercy", "ICU - Mercy", "Ventilated - Mercy"]
    #col_name2 = {"hosp_mercy": "Hospitalized - Mercy", "icu_mercy": "ICU - Mercy", "vent_mercy": "Ventilated - Mercy", "total_beds":"Total Beds", "icu_beds": "Total ICU Beds"}
    #fold_name2 = ["Hospitalized - Mercy", "ICU - Mercy", "Ventilated - Mercy", "Total Beds", "Total ICU Beds"]
    col_name2 = {"hosp_mercy": "Hospitalized - Mercy", "icu_mercy": "ICU - Mercy", "vent_mercy": "Ventilated - Mercy",
        "expanded_beds":"Expanded IP Beds", "expanded_icu_beds": "Expanded ICU Beds"}
    fold_name2 = ["Hospitalized - Mercy", "ICU - Mercy", "Ventilated - Mercy", "Expanded IP Beds", "Expanded ICU Beds"]
    col_name3 ={"ppe_mild_d_mercy": "PPE Mild Cases - Lower Range", "ppe_mild_u_mercy": "PPE Mild Cases - Upper Range", 
    "ppe_severe_d_mercy": "PPE Severe Cases - Lower Range", "ppe_severe_u_mercy": "PPE Severe Cases - Upper Range"}
    fold_name3 = ["PPE Mild Cases - Lower Range", "PPE Mild Cases - Upper Range", "PPE Severe Cases - Lower Range", "PPE Severe Cases - Upper Range"]
    icu_val = 28
    total_beds_val = 306
    expanded_beds_val = 459
    expanded_icu_val = 42
if hosp_options == 'MFSH':
    col_name1 = {"hosp_mfsh": "Hospitalized - MFSH", "icu_mfsh": "ICU - MFSH", "vent_mfsh": "Ventilated - MFSH"}
    fold_name1 = ["Hospitalized - MFSH", "ICU - MFSH", "Ventilated - MFSH"]
    #col_name2 = {"hosp_mfsh": "Hospitalized - MFSH", "icu_mfsh": "ICU - MFSH", "vent_mfsh": "Ventilated - MFSH", "total_beds":"Total Beds", "icu_beds": "Total ICU Beds"}
    #fold_name2 = ["Hospitalized - MFSH", "ICU - MFSH", "Ventilated - MFSH", "Total Beds", "Total ICU Beds"]
    col_name2 = {"hosp_mfsh": "Hospitalized - MFSH", "icu_mfsh": "ICU - MFSH", "vent_mfsh": "Ventilated - MFSH",
        "expanded_beds":"Expanded IP Beds", "expanded_icu_beds": "Expanded ICU Beds"}
    fold_name2 = ["Hospitalized - MFSH", "ICU - MFSH", "Ventilated - MFSH", "Expanded IP Beds", "Expanded ICU Beds"]
    col_name3 ={"ppe_mild_d_mfsh": "PPE Mild Cases - Lower Range", "ppe_mild_u_mfsh": "PPE Mild Cases - Upper Range", 
    "ppe_severe_d_mfsh": "PPE Severe Cases - Lower Range", "ppe_severe_u_mfsh": "PPE Severe Cases - Upper Range"}
    fold_name3 = ["PPE Mild Cases - Lower Range", "PPE Mild Cases - Upper Range", "PPE Severe Cases - Lower Range", "PPE Severe Cases - Upper Range"]
    icu_val = 10
    total_beds_val = 227
    expanded_beds_val = 341
    expanded_icu_val = 15
if hosp_options == 'OCH':
    col_name1 = {"hosp_och": "Hospitalized - Oishei", "icu_och": "ICU - Oishei", "vent_och": "Ventilated - Oishei"}
    fold_name1 = ["Hospitalized - Oishei", "ICU - Oishei", "Ventilated - Oishei"]
    #col_name2 = {"hosp_och": "Hospitalized - Oishei", "icu_och": "ICU - Oishei", "vent_och": "Ventilated - Oishei", "total_beds":"Total Beds", "icu_beds": "Total ICU Beds"}
    #fold_name2 = ["Hospitalized - Oishei", "ICU - Oishei", "Ventilated - Oishei", "Total Beds", "Total ICU Beds"]
    col_name2 = {"hosp_och": "Hospitalized - Oishei", "icu_och": "ICU - Oishei", "vent_och": "Ventilated - Oishei", 
        "expanded_beds":"Expanded IP Beds", "expanded_icu_beds": "Expanded ICU Beds"}
    fold_name2 = ["Hospitalized - Oishei", "ICU - Oishei", "Ventilated - Oishei", "Expanded IP Beds", "Expanded ICU Beds"]
    col_name3 ={"ppe_mild_d_och": "PPE Mild Cases - Lower Range", "ppe_mild_u_och": "PPE Mild Cases - Upper Range", 
    "ppe_severe_d_och": "PPE Severe Cases - Lower Range", "ppe_severe_u_och": "PPE Severe Cases - Upper Range"}
    fold_name3 = ["PPE Mild Cases - Lower Range", "PPE Mild Cases - Upper Range", "PPE Severe Cases - Lower Range", "PPE Severe Cases - Upper Range"]
    icu_val = 20
    total_beds_val = 89
    expanded_beds_val = 134
    expanded_icu_val = 30
if hosp_options == 'RPCI':
    col_name1 = {"hosp_rpci": "Hospitalized - Roswell", "icu_rpci": "ICU - Roswell", "vent_rpci": "Ventilated - Roswell"}
    fold_name1 = ["Hospitalized - Roswell", "ICU - Roswell", "Ventilated - Roswell"]
    #col_name2 = {"hosp_rpci": "Hospitalized - Roswell", "icu_rpci": "ICU - Roswell", "vent_rpci": "Ventilated - Roswell", "total_beds":"Total Beds", "icu_beds": "Total ICU Beds"}
    #fold_name2 = ["Hospitalized - Roswell", "ICU - Roswell", "Ventilated - Roswell", "Total Beds", "Total ICU Beds"]
    col_name2 = {"hosp_rpci": "Hospitalized - Roswell", "icu_rpci": "ICU - Roswell", "vent_rpci": "Ventilated - Roswell", 
        "expanded_beds":"Expanded IP Beds", "expanded_icu_beds": "Expanded ICU Beds"}
    fold_name2 = ["Hospitalized - Roswell", "ICU - Roswell", "Ventilated - Roswell", "Expanded IP Beds", "Expanded ICU Beds"]
    col_name3 ={"ppe_mild_d_rpci": "PPE Mild Cases - Lower Range", "ppe_mild_u_rpci": "PPE Mild Cases - Upper Range", 
    "ppe_severe_d_rpci": "PPE Severe Cases - Lower Range", "ppe_severe_u_rpci": "PPE Severe Cases - Upper Range"}
    fold_name3 = ["PPE Mild Cases - Lower Range", "PPE Mild Cases - Upper Range", "PPE Severe Cases - Lower Range", "PPE Severe Cases - Upper Range"]
    icu_val = 16
    total_beds_val = 110
    expanded_beds_val = 165
    expanded_icu_val = 24
if hosp_options == 'SCH':
    col_name1 = {"hosp_sch": "Hospitalized - Sisters", "icu_sch": "ICU - Sisters", "vent_sch": "Ventilated - Sisters"}
    fold_name1 = ["Hospitalized - Sisters", "ICU - Sisters", "Ventilated - Sisters"]
    #col_name2 = {"hosp_sch": "Hospitalized - Sisters", "icu_sch": "ICU - Sisters", "vent_sch": "Ventilated - Sisters", "total_beds":"Total Beds", "icu_beds": "Total ICU Beds"}
    #fold_name2 = ["Hospitalized - Sisters", "ICU - Sisters", "Ventilated - Sisters", "Total Beds", "Total ICU Beds"]
    col_name2 = {"hosp_sch": "Hospitalized - Sisters", "icu_sch": "ICU - Sisters", "vent_sch": "Ventilated - Sisters",
        "expanded_beds":"Expanded IP Beds", "expanded_icu_beds": "Expanded ICU Beds"}
    fold_name2 = ["Hospitalized - Sisters", "ICU - Sisters", "Ventilated - Sisters", "Expanded IP Beds", "Expanded ICU Beds"]
    col_name3 ={"ppe_mild_d_sch": "PPE Mild Cases - Lower Range", "ppe_mild_u_sch": "PPE Mild Cases - Upper Range", 
    "ppe_severe_d_sch": "PPE Severe Cases - Lower Range", "ppe_severe_u_sch": "PPE Severe Cases - Upper Range"}
    fold_name3 = ["PPE Mild Cases - Lower Range", "PPE Mild Cases - Upper Range", "PPE Severe Cases - Lower Range", "PPE Severe Cases - Upper Range"]
    icu_val = 16
    total_beds_val = 215
    expanded_beds_val = 323
    expanded_icu_val = 24
if hosp_options == 'SCSJH':
    col_name1 = {"hosp_scsjh": "Hospitalized - StJoseph", "icu_scsjh": "ICU - StJoseph", "vent_scsjh": "Ventilated - StJoseph"}
    fold_name1 = ["Hospitalized - StJoseph", "ICU - StJoseph", "Ventilated - StJoseph"]
    col_name2 = {"hosp_scsjh": "Hospitalized - StJoseph", "icu_scsjh": "ICU - StJoseph", "vent_scsjh": "Ventilated - StJoseph", 
        "expanded_beds":"Expanded IP Beds", "expanded_icu_beds": "Expanded ICU Beds"}
    fold_name2 = ["Hospitalized - StJoseph", "ICU - StJoseph", "Ventilated - StJoseph", "Expanded IP Beds", "Expanded ICU Beds"]
    #col_name2 = {"hosp_scsjh": "Hospitalized - StJoseph", "icu_scsjh": "ICU - StJoseph", "vent_scsjh": "Ventilated - StJoseph", "total_beds":"Total Beds", "icu_beds": "Total ICU Beds"}
    #fold_name2 = ["Hospitalized - StJoseph", "ICU - StJoseph", "Ventilated - StJoseph", "Total Beds", "Total ICU Beds"]
    col_name3 ={"ppe_mild_d_scsjh": "PPE Mild Cases - Lower Range", "ppe_mild_u_scsjh": "PPE Mild Cases - Upper Range", 
    "ppe_severe_d_scsjh": "PPE Severe Cases - Lower Range", "ppe_severe_u_scsjh": "PPE Severe Cases - Upper Range"}
    fold_name3 = ["PPE Mild Cases - Lower Range", "PPE Mild Cases - Upper Range", "PPE Severe Cases - Lower Range", "PPE Severe Cases - Upper Range"]
    icu_val = 7
    total_beds_val = 103
    expanded_beds_val = 155
    expanded_icu_val = 11
    
    

# Graphs of new admissions for Erie
st.subheader("New Admissions: SIR Model")
st.markdown("Projected number of **daily** COVID-19 admissions for Erie County")


# New cases SIR
###########################
# New cases
projection_admits = build_admissions_df(dispositions)
# Census Table
census_table = build_census_df(projection_admits)
############################

# New cases SEIR
###########################
# New cases
projection_admits_e = build_admissions_df(dispositions_e)
# Census Table
census_table_e = build_census_df(projection_admits_e)

# Projection days
plot_projection_days = n_days - 10
############################



# Erie Graph of Cases: SIR
#def regional_admissions_chart(projection_admits: pd.DataFrame, plot_projection_days: int) -> alt.Chart:
###
### Admissions Graphs
###
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
        x_kwargs = {"shorthand": "day", "title": "Days from today"}
    
    return (
        alt
        .Chart(projection_admits.head(plot_projection_days))
        .transform_fold(fold=["Hospitalized", "ICU", "Ventilated"])
        .mark_line(point=True)
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

#st.altair_chart(alt.layer(altair_chart(regional_admissions_chart(projection_admits).mark_line()))
 #+ alt.layer(regional_admissions_chart(projection_admits_e).mark_line()), use_container_width=True)    

# Erie County Admissions Chart
st.altair_chart(
    regional_admissions_chart(projection_admits, 
        plot_projection_days, 
        as_date=as_date), 
    use_container_width=True)

st.subheader("New Admissions: SEIR Model")
st.markdown("Projected number of **daily** COVID-19 admissions for Erie County")


# Erie Graph of Cases: SEIR

st.altair_chart(
    regional_admissions_chart(projection_admits_e, 
        plot_projection_days, 
        as_date=as_date), 
    use_container_width=True)


st.subheader("Projected number of **daily** COVID-19 admissions by Hospital: SIR model")
st.markdown("Distribution of regional cases based on total bed percentage (CCU/ICU/MedSurg).")

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
        x_kwargs = {"shorthand": "day", "title": "Days from today"}
    
    return (
        alt
        .Chart(projection_admits.head(plot_projection_days))
        .transform_fold(fold=fold_name1)
        .mark_line(point=True)
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
    


# By Hospital Admissions Chart - SIR
st.altair_chart(
    hospital_admissions_chart(
        projection_admits, plot_projection_days, as_date=as_date), 
    use_container_width=True)


if st.checkbox("Show Projected Admissions in tabular form:SIR"):
    admits_table = projection_admits[np.mod(projection_admits.index, 7) == 0].copy()
    admits_table["day"] = admits_table.index
    admits_table.index = range(admits_table.shape[0])
    admits_table = admits_table.fillna(0).astype(int)
    
    st.dataframe(admits_table)

st.subheader("Projected number of **daily** COVID-19 admissions by Hospital: SEIR model")
st.markdown("Distribution of regional cases based on total bed percentage (CCU/ICU/MedSurg).")

# By Hospital Admissions Chart - SEIR
st.altair_chart(
    hospital_admissions_chart(
        projection_admits_e, plot_projection_days, as_date=as_date), 
    use_container_width=True)
    
if st.checkbox("Show Projected Admissions in tabular form:SEIR"):
    admits_table_e = projection_admits_e[np.mod(projection_admits_e.index, 7) == 0].copy()
    admits_table_e["day"] = admits_table_e.index
    admits_table_e.index = range(admits_table_e.shape[0])
    admits_table_e = admits_table_e.fillna(0).astype(int)
    
    st.dataframe(admits_table_e)

st.subheader("Admitted Patients (Census)")
st.subheader("Projected **census** of COVID-19 patients for Erie County, accounting for arrivals and discharges: **SIR Model**")

###################
# Census Graphs
####################

def admitted_patients_chart(
    census: pd.DataFrame,
    plot_projection_days: int,
    as_date:bool = False) -> alt.Chart:
    """docstring"""
    census = census.rename(columns={"hosp": "Hospital Census", "icu": "ICU Census", "vent": "Ventilated Census", 
    "expanded_beds_county":"Expanded IP Beds", "expanded_icu_county": "Expanded ICU Beds"})

    tooltip_dict = {False: "day", True: "date:T"}
    if as_date:
        census = add_date_column(census.head(plot_projection_days))
        x_kwargs = {"shorthand": "date:T", "title": "Date"}
    else:
        x_kwargs = {"shorthand": "day", "title": "Days from today"}

    return (
        alt
        .Chart(census)
        .transform_fold(fold=["Hospital Census", "ICU Census", "Ventilated Census", "Expanded IP Beds", "Expanded ICU Beds"])
        .mark_line(point=False)
        .encode(
            x=alt.X(**x_kwargs),
            y=alt.Y("value:Q", title="Census"),
            color="key:N",
            tooltip=[
                tooltip_dict[as_date],
                alt.Tooltip("value:Q", format=".0f", title="Admissions"),
                "key:N",
            ],
        )
        .interactive()
    )

# Erie County Census Graph - SIR
st.altair_chart(
    admitted_patients_chart(
        census_table,
        plot_projection_days,
        as_date=as_date),
    use_container_width=True)

st.subheader("Projected **census** of COVID-19 patients for Erie County, accounting for arrivals and discharges: **SEIR Model**")
    
# Erie County Census Graph - SEIR
st.altair_chart(
    admitted_patients_chart(
        census_table_e,
        plot_projection_days,
        as_date=as_date),
    use_container_width=True)

# , scale=alt.Scale(domain=[0, 30000])

# st.altair_chart(alt.layer(admitted_patients_chart(census_table).mark_line() + alt.layer(erie_chart(erie_admits).mark_line())), use_container_width=True)

# Census by hospital
def hosp_admitted_patients_chart(
    census: pd.DataFrame, 
    as_date:bool = False) -> alt.Chart:
    """docstring"""
    census = census.rename(columns=col_name2)

    tooltip_dict = {False: "day", True: "date:T"}
    if as_date:
        census = add_date_column(census)
        x_kwargs = {"shorthand": "date:T", "title": "Date"}
    else:
        x_kwargs = {"shorthand": "day", "title": "Days from today"}

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
                alt.Tooltip("value:Q", format=".0f", title="Admissions"),
                "key:N",
            ],
        )
        .interactive()
    )


st.markdown("The following two graphs still need adjustment of the bed distribution (horizontal lines) due to recent changes in bed expansion")
st.subheader("Projected **census** of COVID-19 patients by Hospital, accounting for arrivals and discharges: SIR Model")

st.altair_chart(
    hosp_admitted_patients_chart(
        census_table, 
        as_date=as_date), 
    use_container_width=True)

st.subheader("Projected **census** of COVID-19 patients by Hospital, accounting for arrivals and discharges: SEIR Model")

st.altair_chart(
    hosp_admitted_patients_chart(
        census_table_e, 
        as_date=as_date), 
    use_container_width=True)

# , scale=alt.Scale(domain=[0, 30000])

if st.checkbox("Show Projected Census in tabular form"):
    st.dataframe(census_table)

#st.markdown(
#    """**Click the checkbox below to view additional data generated by this simulation**"""
#)

st.subheader("Projected personal protective equipment needs for mild and severe cases of COVID-19: SIR Model")

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
        x_kwargs = {"shorthand": "day", "title": "Days from today"}

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

#SIR
st.altair_chart(
    ppe_chart(
    census_table,
    as_date=as_date),
    use_container_width=True)
    
st.subheader("Projected personal protective equipment needs for mild and severe cases of COVID-19: SEIR Model")
# SEIR
st.altair_chart(
    ppe_chart(
    census_table_e,
    as_date=as_date),
    use_container_width=True)

# PPE needs summary variables

def get_key(dic, val): 
    for key, value in dic.items(): 
         if val == value: 
             return key 

float_formatter = "{:.2f}".format

# ppe_7d_mild_lower = census_table['ppe_mild_d'][1:8]
# ppe_7d_mild_upper = census_table['ppe_mild_u'][1:8]
# ppe_7d_severe_lower = census_table['ppe_severe_d'][1:8]
# ppe_7d_severe_upper = census_table['ppe_severe_u'][1:8]

# st.table(ppe_7d_mild_lower)
# st.table(ppe_7d_mild_upper)
# st.table(ppe_7d_severe_lower)
# st.table(ppe_7d_severe_upper)


# One day
ppe_1d_mild_lower = current_hosp * 14.0
ppe_1d_mild_upper = current_hosp * 15.0
ppe_1d_severe_lower = current_hosp *15.0
ppe_1d_severe_upper = current_hosp *24.0
# 7 Days
ppe_7d_mild_lower = ppe_1d_mild_lower+sum(census_table['ppe_mild_d'][1:8])
ppe_7d_mild_upper = ppe_1d_mild_upper+sum(census_table['ppe_mild_u'][1:8])
ppe_7d_severe_lower = ppe_1d_severe_lower+sum(census_table['ppe_severe_d'][1:8])
ppe_7d_severe_upper = ppe_1d_severe_upper+sum(census_table['ppe_severe_u'][1:8])
# 2nd week
ppe_14d_mild_lower = sum(census_table['ppe_mild_d'][1:15])
ppe_14d_mild_upper = sum(census_table['ppe_mild_u'][1:15])
ppe_14d_severe_lower = sum(census_table['ppe_severe_d'][1:15])
ppe_14d_severe_upper = sum(census_table['ppe_severe_u'][1:15])
# 3rd week
ppe_21d_mild_lower = sum(census_table['ppe_mild_d'][1:22])
ppe_21d_mild_upper = sum(census_table['ppe_mild_u'][1:22])
ppe_21d_severe_lower = sum(census_table['ppe_severe_d'][1:22])
ppe_21d_severe_upper = sum(census_table['ppe_severe_u'][1:22])
# Month one
ppe_1m_mild_lower = sum(census_table['ppe_mild_d'][1:29])
ppe_1m_mild_upper = sum(census_table['ppe_mild_u'][1:29])
ppe_1m_severe_lower = sum(census_table['ppe_severe_d'][1:29])
ppe_1m_severe_upper = sum(census_table['ppe_severe_u'][1:29])

st.markdown("""The estimated **daily** PPE needs for the currently admitted COVID-19 patients is **{ppe_1d_mild_lower:.0f}**-**{ppe_1d_mild_upper:.0f}** for mild cases of COVID-19, 
            and **{ppe_1d_severe_lower:.0f}**-**{ppe_1d_severe_upper:.0f}** for severe cases. The estimated PPE needs for the **first month** of COVID-19 patients is
            **{ppe_1m_mild_lower:.0f}**-**{ppe_1m_mild_upper:.0f}** for mild cases of COVID-19, and **{ppe_1m_severe_lower:.0f}**-**{ppe_1m_severe_upper:.0f}** for severe cases.""".format(
                ppe_1d_mild_lower = ppe_1d_mild_lower,
                ppe_1d_mild_upper = ppe_1d_mild_upper,
                ppe_1d_severe_lower = ppe_1d_severe_lower,
                ppe_1d_severe_upper = ppe_1d_severe_upper,
                ppe_1m_mild_lower = ppe_1m_mild_lower,
                ppe_1m_mild_upper = ppe_1m_mild_upper,
                ppe_1m_severe_lower = ppe_1m_severe_lower,
                ppe_1m_severe_upper = ppe_1m_severe_upper
))

# PPE Needs by day, 7d, 14, 21, 28d
data = {
    'PPE Needs': ['Today', 'Week 1', 'Week2', 'Week3', 'First Month'],
    'Mild Cases' : [
        float_formatter(np.mean([ppe_1d_mild_lower, ppe_1d_mild_upper])), 
        float_formatter(np.mean([ppe_7d_mild_lower, ppe_7d_mild_upper])), 
        float_formatter(np.mean([ppe_14d_mild_lower, ppe_14d_mild_upper])), 
        float_formatter(np.mean([ppe_21d_mild_lower, ppe_21d_mild_upper])), 
        float_formatter(np.mean([ppe_1m_mild_lower, ppe_1m_mild_upper]))
        ],
    'Severe Cases': [
        float_formatter(np.mean([ppe_1d_severe_lower, ppe_1d_severe_upper])), 
        float_formatter(np.mean([ppe_7d_severe_lower, ppe_7d_severe_upper])), 
        float_formatter(np.mean([ppe_14d_severe_lower, ppe_14d_severe_upper])),
        float_formatter(np.mean([ppe_21d_severe_lower, ppe_21d_severe_upper])), 
        float_formatter(np.mean([ppe_1m_severe_lower, ppe_1m_severe_upper]))
        ], 
}
ppe_needs = pd.DataFrame(data)
st.table(ppe_needs)

# Recovered/Infected table
st.subheader("The number of infected and recovered individuals in the region at any given moment")

def additional_projections_chart(i: np.ndarray, r: np.ndarray) -> alt.Chart:
    dat = pd.DataFrame({"Infected": i, "Recovered": r})

    return (
        alt
        .Chart(dat.reset_index())
        .transform_fold(fold=["Infected", "Recovered"])
        .mark_line(point=True)
        .encode(
            x=alt.X("index", title="Days from today"),
            y=alt.Y("value:Q", title="Case Volume"),
            tooltip=["key:N", "value:Q"], 
            color="key:N"
        )
        .interactive()
    )

st.altair_chart(additional_projections_chart(i, r), use_container_width=True)



if st.checkbox("Show Additional Information"):

    st.subheader("Guidance on Selecting Inputs")
    st.markdown(
        """* **Hospitalized COVID-19 Patients:** The number of patients currently hospitalized with COVID-19. This number is used in conjunction with Hospital Market Share and Hospitalization % to estimate the total number of infected individuals in your region.
        * **Currently Known Regional Infections**: The number of infections reported in your hospital's catchment region. This input is used to estimate the detection rate of infected individuals. 
        * **Doubling Time (days):** This parameter drives the rate of new cases during the early phases of the outbreak. The American Hospital Association currently projects doubling rates between 7 and 10 days. This is the doubling time you expect under status quo conditions. To account for reduced contact and other public health interventions, modify the _Social distancing_ input. 
        * **Social distancing (% reduction in person-to-person physical contact):** This parameter allows users to explore how reduction in interpersonal contact & transmission (hand-washing) might slow the rate of new infections. It is your estimate of how much social contact reduction is being achieved in your region relative to the status quo. While it is unclear how much any given policy might affect social contact (eg. school closures or remote work), this parameter lets you see how projections change with percentage reductions in social contact.
        * **Hospitalization %(total infections):** Percentage of **all** infected cases which will need hospitalization.
        * **ICU %(total infections):** Percentage of **all** infected cases which will need to be treated in an ICU.
        * **Ventilated %(total infections):** Percentage of **all** infected cases which will need mechanical ventilation.
        * **Hospital Length of Stay:** Average number of days of treatment needed for hospitalized COVID-19 patients. 
        * **ICU Length of Stay:** Average number of days of ICU treatment needed for ICU COVID-19 patients.
        * **Vent Length of Stay:**  Average number of days of ventilation needed for ventilated COVID-19 patients.
        * **Hospital Market Share (%):** The proportion of patients in the region that are likely to come to your hospital (as opposed to other hospitals in the region) when they get sick. One way to estimate this is to look at all of the hospitals in your region and add up all of the beds. The number of beds at your hospital divided by the total number of beds in the region times 100 will give you a reasonable starting estimate.
        * **Regional Population:** Total population size of the catchment region of your hospital(s). 
        """)

# Show data
days = np.array(range(0, n_days + 1))
data_list = [days, s, i, r]
data_dict = dict(zip(["day", "susceptible", "infections", "recovered"], data_list))
projection_area = pd.DataFrame.from_dict(data_dict)
infect_table = (projection_area.iloc[::7, :]).apply(np.floor)
infect_table.index = range(infect_table.shape[0])

if st.checkbox("Show Raw SIR Similation Data"):
    st.dataframe(infect_table)







st.subheader("References & Acknowledgements")
st.markdown(
    """
    We appreciate the great work done by Predictive Healthcare team (http://predictivehealthcare.pennmedicine.org/) at
Penn Medicine who created the predictive model used.
    https://www.worldometers.info/coronavirus/coronavirus-incubation-period/
Lauer SA, Grantz KH, Bi Q, et al. The Incubation Period of Coronavirus Disease 2019 (COVID-19) From Publicly Reported Confirmed Cases: Estimation and Application. Ann Intern Med. 2020; [Epub ahead of print 10 March 2020]. doi: https://doi.org/10.7326/M20-0504
http://www.centerforhealthsecurity.org/resources/COVID-19/index.html
http://www.centerforhealthsecurity.org/resources/fact-sheets/pdfs/coronaviruses.pdf'
https://coronavirus.jhu.edu/
https://www.who.int/emergencies/diseases/novel-coronavirus-2019/situation-reports
https://www.worldometers.info/coronavirus/coronavirus-age-sex-demographics/
    """
)
