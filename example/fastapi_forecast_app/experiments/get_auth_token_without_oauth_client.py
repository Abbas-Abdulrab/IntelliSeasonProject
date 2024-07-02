import os

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
import requests


# Load all the entries from .env file as environment variables
# The .env file should have the values for GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET
load_dotenv(dotenv_path="../.env")

app = FastAPI()

client_id = os.getenv('GOOGLE_CLIENT_ID')
client_secret = os.getenv('GOOGLE_CLIENT_SECRET')
redirect_uri = "http://localhost/callback"
authorization_url = "https://accounts.google.com/o/oauth2/auth"
token_url = "https://oauth2.googleapis.com/token"


@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI OAuth2 example. Visit /login to authenticate."}


@app.get("/login")
def login():
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "https://www.googleapis.com/auth/cloud-platform",
        "access_type": "offline",
        "prompt": "consent"
    }
    # Form the URL using requests' request object to handle parameters properly
    url = requests.Request('GET', authorization_url, params=params).prepare().url
    return RedirectResponse(url)


@app.get("/callback")
def callback(request: Request):
    code = request.query_params.get("code")
    if not code:
        return {"error": "No code provided"}

    response = requests.post(
        token_url,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    token_response_json = response.json()
    access_token = token_response_json.get("access_token")
    if not access_token:
        return {"error": "Failed to obtain access token"}

    print(f"Access Token: {access_token}")
    return {"access_token": access_token}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=80)
