# Modified version for Western New York
# Contact: ganaya@buffalo.edu

import pandas as pd
import streamlit as st
import numpy as np
import matplotlib
from bs4 import BeautifulSoup
#import plotly.graph_objects as go
import requests
import ipyvuetify as v
from traitlets import Unicode, List
import datetime
from datetime import date
import time

matplotlib.use("Agg")
import matplotlib.pyplot as plt

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# General Variables
today = date.today()
fdate = date.today().strftime("%m-%d-%Y")
time = time.strftime("%H:%M:%S")

### Extract data for US
# URL
# 1 Request URL
url = 'https://www.cdc.gov/coronavirus/2019-ncov/cases-updates/cases-in-us.html'
page = requests.get(url)
# 2 Parse HTML content
soup = BeautifulSoup(page.text, 'html.parser')
# 3 Extract cases data
cdc_data = soup.find_all(attrs={"class": "card-body bg-white"})

# Create dataset of extracted data
df = []
for ul in cdc_data:
    for li in ul.find_all('li'):
        df.append(li.text.replace('\n', ' ').strip())
### US specific cases - CDC
cases_us = df[0].split(': ')
# Replace + and , for numeric values
cases_us = int(cases_us[1].replace(',', ''))
# Total US deaths - CDC
deaths_us = df[1].split(': ')
deaths_us = pd.to_numeric(deaths_us[1])
# Calculate mortality rate
us_MR = round((deaths_us/cases_us)*100,2)
# Create table
data = {'Cases': [cases_us],
       'Deaths': [deaths_us],
       'Calculated Mortality Rate': [us_MR]}
us_data = pd.DataFrame(data)

### Extract data for NY State cases
# URL
# 1 Request URL
url = 'https://coronavirus.health.ny.gov/county-county-breakdown-positive-cases'
page = requests.get(url)
# 2 Parse HTML content
soup = BeautifulSoup(page.text, 'html.parser')
# 3 Get the table having the class country table
table = soup.find("div", attrs={'class':"wysiwyg--field-webny-wysiwyg-body"})
table_data = table.find_all("td")
# Get all the headings of Lists
df = []
for i in range(0,len(table_data)):
    for td in table_data[i]:
        df.append(table_data[i].text.replace('\n', ' ').strip())
        
counties = pd.DataFrame([])
for i in range(0, len(df), 2):
    counties = counties.append(pd.DataFrame({'County': df[i], 'Cases': df[i+1]},
                                              index =[0]), ignore_index=True)

# NY state Modification for Counties and State Tables
nys = counties[-3:]
counties_cases = counties[0:-3]
counties_cases['Cases'] = pd.to_numeric(counties_cases['Cases'])
# Total state cases for print
cases_nys = pd.to_numeric(counties.iloc[-1,1].replace(',', ''))
#Erie County
erie = counties[counties['County']=='Erie'].reset_index()
cases_erie = pd.to_numeric(erie.Cases[0])


# Populations and Infections
buffalo = 258612
tonawanda = 14904
cheektowaga = 87018
amherst = 126082
erie = 919794
S_default = erie
known_infections = cases_erie

# Widgets
initial_infections = st.sidebar.number_input(
    "Currently Known Regional Infections", value=known_infections, step=10, format="%i"
)
current_hosp = st.sidebar.number_input(
    "Currently Hospitalized COVID-19 Patients", value=1, step=1, format="%i"
)
doubling_time = st.sidebar.number_input(
    "Doubling Time (days)", value=6, step=1, format="%i"
)
relative_contact_rate = st.sidebar.number_input(
    "Social distancing (% reduction in social contact)", 0, 100, value=0, step=5, format="%i"
)/100.0

hosp_rate = (
    st.sidebar.number_input("Hospitalization %", 0, 100, value=5, step=1, format="%i")
    / 100.0
)
icu_rate = (
    st.sidebar.number_input("ICU %", 0, 100, value=2, step=1, format="%i") / 100.0
)
vent_rate = (
    st.sidebar.number_input("Ventilated %", 0, 100, value=1, step=1, format="%i")
    / 100.0
)
hosp_los = st.sidebar.number_input("Hospital LOS", value=7, step=1, format="%i")
icu_los = st.sidebar.number_input("ICU LOS", value=9, step=1, format="%i")
vent_los = st.sidebar.number_input("Vent LOS", value=10, step=1, format="%i")
BGH_market_share = (
    st.sidebar.number_input(
        "Hospital Market Share (%)", 0.0, 100.0, value=15.0, step=1.0, format="%f"
    )
    / 100.0
)
S = st.sidebar.number_input(
    "Regional Population", value=S_default, step=100000, format="%i"
)

total_infections = current_hosp / BGH_market_share / hosp_rate
detection_prob = initial_infections / total_infections

S, I, R = S, initial_infections / detection_prob, 0

intrinsic_growth_rate = 2 ** (1 / doubling_time) - 1

recovery_days = 14.0
# mean recovery rate, gamma, (in 1/days).
gamma = 1 / recovery_days

# Contact rate, beta
beta = (
    intrinsic_growth_rate + gamma
) / S * (1-relative_contact_rate) # {rate based on doubling time} / {initial S}

r_t = beta / gamma * S # r_t is r_0 after distancing
r_naught = r_t / (1-relative_contact_rate)
doubling_time_t = 1/np.log2(beta*S - gamma +1) # doubling time after distancing

st.title("COVID-19 Hospital Impact Model for Epidemics - Modified for Erie County")
st.markdown(
    """*This tool was developed by the [Predictive Healthcare team](http://predictivehealthcare.pennmedicine.org/) at
Penn Medicine. 

All credit goes to the PH team at Penn Medicine. We have adapted the code based on our current cases and population.

For questions about this page, contact ganaya@buffalo.edu. 

For question and comments about the model [contact page](http://predictivehealthcare.pennmedicine.org/contact/).""")

st.markdown(
    """The estimated number of currently infected individuals is **{total_infections:.0f}**. The **{initial_infections}** 
confirmed cases in the region imply a **{detection_prob:.0%}** rate of detection. This is based on current inputs for 
Hospitalizations (**{current_hosp}**), Hospitalization rate (**{hosp_rate:.0%}**), Region size (**{S}**), 
and Hospital market share (**{BGH_market_share:.0%}**).

An initial doubling time of **{doubling_time}** days and a recovery time of **{recovery_days}** days imply an $R_0$ of 
**{r_naught:.2f}**.

**Mitigation**: A **{relative_contact_rate:.0%}** reduction in social contact after the onset of the 
outbreak reduces the doubling time to **{doubling_time_t:.1f}** days, implying an effective $R_t$ of **${r_t:.2f}$**.""".format(
        total_infections=total_infections,
        current_hosp=current_hosp,
        hosp_rate=hosp_rate,
        S=S,
        BGH_market_share=BGH_market_share,
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

st.subheader("Cases of COVID-19 in the United States")
# Table of cases in the US
st.table(us_data)
# Table of cases in NYS
st.subheader("Cases of COVID-19 in New York State")
counties_cases.sort_values(by=['Cases'], ascending=False)
st.table(nys)
st.subheader("Cases of COVID-19 in Erie County")
st.markdown(
    """Erie county has reported **{cases_erie:.0f}** cases of COVID-19.""".format(
        cases_erie=cases_erie
    )
)
st.markdown(""" """)
st.markdown(""" """)
st.markdown(""" """)

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
        """To project the expected impact to Great Lakes Health System, we follow the model created by Penn Medicine, we estimate the terms of the model. 

To do this, we use a combination of estimates from other locations, informed estimates based on logical reasoning, and best guesses from the American Hospital Association.


### Parameters
First, we need to express the two parameters $\\beta$ and $\\gamma$ in terms of quantities we can estimate.

- The $\\gamma$ parameter represents 1 over the mean recovery time in days. Since the CDC is recommending 14 days of self-quarantine, we'll use $\\gamma = 1/14$. 
- Next, the AHA says to expect a doubling time $T_d$ of 7-10 days. That means an early-phase rate of growth can be computed by using the doubling time formula:
"""
    )
    st.latex("g = 2^{1/T_d} - 1")

    st.markdown(
        """
        - Since the rate of new infections in the SIR model is $g = \\beta S - \\gamma$, and we've already computed $\\gamma$, $\\beta$ becomes a function of the initial population size of susceptible individuals.
        $$\\beta = (g + \\gamma)/s$$

### Initial Conditions

- The total size of the susceptible population will be the entire catchment area for Buffalo General Hospita, Oishei Children's Hospital, Millard Fillmore Suburban Hospital, 
Erie County Medical Center, Sisters of Charity Hospital, Mercy Hospital.
- Erie = {erie}""".format(
            erie=erie))


# The SIR model, one time step
def sir(y, beta, gamma, N):
    S, I, R = y
    Sn = (-beta * S * I) + S
    In = (beta * S * I - gamma * I) + I
    Rn = gamma * I + R
    if Sn < 0:
        Sn = 0
    if In < 0:
        In = 0
    if Rn < 0:
        Rn = 0

    scale = N / (Sn + In + Rn)
    return Sn * scale, In * scale, Rn * scale


# Run the SIR model forward in time
def sim_sir(S, I, R, beta, gamma, n_days, beta_decay=None):
    N = S + I + R
    s, i, r = [S], [I], [R]
    for day in range(n_days):
        y = S, I, R
        S, I, R = sir(y, beta, gamma, N)
        if beta_decay:
            beta = beta * (1 - beta_decay)
        s.append(S)
        i.append(I)
        r.append(R)

    s, i, r = np.array(s), np.array(i), np.array(r)
    return s, i, r


## RUN THE MODEL

S, I, R = S, initial_infections / detection_prob, 0

intrinsic_growth_rate = 2 ** (1 / doubling_time) - 1

recovery_days = 14.0
# mean recovery rate, gamma, (in 1/days).
gamma = 1 / recovery_days

# Contact rate, beta
beta = (
    intrinsic_growth_rate + gamma
) / S  # {rate based on doubling time} / {initial S}


n_days = st.slider("Number of days to project", 30, 200, 90, 1, "%i")

beta_decay = 0.0
s, i, r = sim_sir(S, I, R, beta, gamma, n_days, beta_decay=beta_decay)


hosp = i * hosp_rate * BGH_market_share
icu = i * icu_rate * BGH_market_share
vent = i * vent_rate * BGH_market_share

days = np.array(range(0, n_days + 1))
data_list = [days, hosp, icu, vent]
data_dict = dict(zip(["day", "hosp", "icu", "vent"], data_list))

projection = pd.DataFrame.from_dict(data_dict)

st.subheader("New Admissions")
st.markdown("Projected number of **daily** COVID-19 admissions at Great Lakes Healthcare System")

# New cases
projection_admits = projection.iloc[:-1, :] - projection.shift(1)
projection_admits[projection_admits < 0] = 0

plot_projection_days = n_days - 10
projection_admits["day"] = range(projection_admits.shape[0])

fig, ax = plt.subplots(1, 1, figsize=(10, 4))
ax.plot(
    projection_admits.head(plot_projection_days)["hosp"], ".-", label="Hospitalized"
)
ax.plot(projection_admits.head(plot_projection_days)["icu"], ".-", label="ICU")
ax.plot(projection_admits.head(plot_projection_days)["vent"], ".-", label="Ventilated")
ax.legend(loc=0)
ax.set_xlabel("Days from today")
ax.grid("on")
ax.set_ylabel("Daily Admissions")
st.pyplot()

admits_table = projection_admits[np.mod(projection_admits.index, 7) == 0].copy()
admits_table["day"] = admits_table.index
admits_table.index = range(admits_table.shape[0])
admits_table = admits_table.fillna(0).astype(int)

if st.checkbox("Show Projected Admissions in tabular form"):
    st.dataframe(admits_table)

st.subheader("Admitted Patients (Census)")
st.markdown(
    "Projected **census** of COVID-19 patients, accounting for arrivals and discharges at Buffalo General hospitals"
)

# ALOS for each category of COVID-19 case (total guesses)

los_dict = {
    "hosp": hosp_los,
    "icu": icu_los,
    "vent": vent_los,
}

fig, ax = plt.subplots(1, 1, figsize=(10, 4))

census_dict = {}
for k, los in los_dict.items():
    census = (
        projection_admits.cumsum().iloc[:-los, :]
        - projection_admits.cumsum().shift(los).fillna(0)
    ).apply(np.ceil)
    census_dict[k] = census[k]
    ax.plot(census.head(plot_projection_days)[k], ".-", label=k + " census")
    ax.legend(loc=0)

ax.set_xlabel("Days from today")
ax.grid("on")
ax.set_ylabel("Census")
st.pyplot()

census_df = pd.DataFrame(census_dict)
census_df["day"] = census_df.index
census_df = census_df[["day", "hosp", "icu", "vent"]]

census_table = census_df[np.mod(census_df.index, 7) == 0].copy()
census_table.index = range(census_table.shape[0])
census_table.loc[0, :] = 0
census_table = census_table.dropna().astype(int)

if st.checkbox("Show Projected Census in tabular form"):
    st.dataframe(census_table)

st.markdown(
    """**Click the checkbox below to view additional data generated by this simulation**"""
)
if st.checkbox("Show Additional Projections"):
    st.subheader(
        "The number of infected and recovered individuals in the hospital catchment region at any given moment"
    )
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(i, label="Infected")
    ax.plot(r, label="Recovered")
    ax.legend(loc=0)
    ax.set_xlabel("days from today")
    ax.set_ylabel("Case Volume")
    ax.grid("on")
    st.pyplot()

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
