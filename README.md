## Model and Model Development
The SEIR model and application were developed by the University at Buffalo's with [Peter Winkelstein](http://medicine.buffalo.edu/faculty/profile.html?ubit=pwink) at Kaleida Health and [Great Lakes Healthcare](https://www.greatlakeshealth.com). The [Biomedical Informatics Department](http://medicine.buffalo.edu/departments/biomedical-informatics.html) with special help from [Matthew Bonner](http://sphhp.buffalo.edu/epidemiology-and-environmental-health/faculty-and-staff/faculty-directory/mrbonner.html) in the Department of Epidemiology and Environmental Health and [Greg Wilding](http://sphhp.buffalo.edu/biostatistics/faculty-and-staff/faculty-directory/gwilding.html) in the Department of Biostatistics. 

Building off of the core application from the [CHIME model](https://github.com/CodeForPhilly/chime/), our model adds compartments for _Exposed_ and _Death_ and fine-tunes the model for Erie County and hospital specific estimates.

Documentation of parameter choices and model choices can be found in the github Wiki.  For questions, please email [Gabriel Anaya](ganaya@buffalo.edu) or [Sarah Mullin](sarahmul@buffalo.edu).  

The application can be found [here](https://khcovid19.herokuapp.com).

## Development
To test the app locally just run:

`streamlit run app.py`

This will open a browser window with the app running.

### With `pipenv`
```bash
pipenv shell
pipenv install
streamlit run app.py
```

### With `conda` 
```bash
conda create -f environment.yml
source activate chime
streamlit run app.py
```

### Developing with `docker`

Copy `.env.example` to be `.env` and run the container.

```bash
cp .env.example .env
docker-compose up
```

You should be able to view the app via `localhost:8000`. If you want to change the
port, then set `PORT` in the `.env` file.

**NOTE** this is just for usage, not for development--- you would have to restart and possibly rebuild the app every time you change the code. 

## Deployment
**Before you push your changes to master make sure that everything works in development mode.**

## Acknowledgements

This work has been supported in part by grants from NIH NLM T15LM012495, NIAA R21AA026954, and NCATS UL1TR001412.
