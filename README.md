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
