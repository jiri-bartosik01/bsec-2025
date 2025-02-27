import importlib
import os

import toml
from api.app.services.middleware import OptionsMiddleware
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

app = FastAPI()

directory = "app/routers"
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f) and f.endswith(".py") and "__init__" not in f:
        module_name = filename[:-3]  # remove .py from filename
        module = importlib.import_module(directory.replace("/", ".") + "." + module_name, package=None)
        app.include_router(module.router)


app.add_middleware(BaseHTTPMiddleware, dispatch=OptionsMiddleware())

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    config = {}
    with open("pyproject.toml", "r") as f:
        config = toml.load(f)
    return {"version": config["project"]["version"]}
