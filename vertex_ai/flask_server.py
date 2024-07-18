import os
from flask import Flask, request, redirect, session, jsonify, render_template, url_for
from requests_oauthlib import OAuth2Session
from google.oauth2.credentials import Credentials
from google.cloud import bigquery
from google.auth.transport.requests import Request
from google.cloud import storage, aiplatform
import subprocess
import webbrowser
import utils
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your_default_secret_key')
client_id = ''
client_secret = ''
authorization_base_url = "https://accounts.google.com/o/oauth2/auth"
token_url = "https://oauth2.googleapis.com/token"
redirect_uri = "http://localhost:5000/callback"
bucket_name = ''
user_info_url = "https://www.googleapis.com/oauth2/v1/userinfo"

scope = [
    'https://www.googleapis.com/auth/userinfo.profile', 
    'https://www.googleapis.com/auth/userinfo.email',
    'https://www.googleapis.com/auth/cloud-platform',
    'https://www.googleapis.com/auth/devstorage.full_control',
    'https://www.googleapis.com/auth/bigquery'
]

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# Store the token in memory
global_token = None
global user_info
@app.route('/')
def index():
    # global global_token
    # if global_token:
    #     return 'You are already logged in. <a href="/logout">Logout</a>'
    # else:
    return redirect(url_for(login))

@app.route('/login')
def login():
    
    google = OAuth2Session(client_id, redirect_uri=redirect_uri, scope=scope)
    authorization_url, state = google.authorization_url(authorization_base_url, access_type="offline")
    session['oauth_state'] = state
    return redirect(authorization_url)



@app.route('/streamlit')
def streamlit_app():
    return redirect("http://localhost:8501")

@app.route('/callback')
def callback():
    global global_token
    global user_info
    google = OAuth2Session(client_id, redirect_uri=redirect_uri, state=session['oauth_state'])
    token = google.fetch_token(token_url, client_secret=client_secret, authorization_response=request.url)
    global_token = token

    user_info = google.get(user_info_url).json()

    # Save user ID in session
    session['user_info'] = {
        'id': user_info.get('id'),
        'name': user_info.get('name'),
        'email': user_info.get('email')
    }
    subprocess.Popen(["streamlit", "run", "streamlit_app.py"])
    return redirect(url_for('streamlit_app'))

@app.route('/data')
def get_data():
    global global_token
    if not global_token:
        return redirect('/login')

    token = global_token
    credentials = Credentials(
        token=token['access_token'],
        refresh_token=token['refresh_token'],
        token_uri=token_url,
        client_id=client_id,
        client_secret=client_secret
    )
    credentials.refresh(Request())
    client = bigquery.Client(credentials=credentials, project="")

    query = """
    SELECT *
    FROM `` LIMIT 1000
    """
    df = client.query(query).to_dataframe()
    data_json = df.to_json(orient='records')

    return jsonify(data_json)


@app.route('/automl',methods=['POST'])
def automl():
    global user_info
    global global_token
    if not global_token:
        return 'no global token', redirect('/login')

    file_path = request.form.get('file_path')
    target_column = request.form.get('target_column')
    date_column = request.form.get('date_column')
    time_series_identifier = request.form.get('time_series_identifier')

    if not file_path or not os.path.exists(file_path):
        return "Invalid file path", 400

    token = global_token
    credentials = Credentials(
        token=token['access_token'],
        refresh_token=token['refresh_token'],
        token_uri=token_url,
        client_id=client_id,
        client_secret=client_secret,
    )
    credentials.refresh(Request())
    storage_client = storage.Client(credentials=credentials, project='')
    bucket = storage_client.bucket(bucket_name)

    # Upload the CSV file to the bucket
    blob_name = f"{os.path.basename(file_path).split('.')[0]}_{user_info['id']}.csv"
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(file_path)

    # Initialize Vertex AI client
    aiplatform.init(project='', location='', credentials=credentials)

    # Create dataset
    dataset = aiplatform.TabularDataset.create(
        display_name=f"{os.path.basename(file_path).split('.')[0]}_{user_info['id']}",
        gcs_source=[f"gs://{bucket_name}/{blob_name}"]
    )

    job = aiplatform.AutoMLTabularTrainingJob(
            display_name=f"training_job_{os.path.basename(file_path).split('.')[0]}_{user_info['id']}",
            optimization_prediction_type="regression",
            optimization_objective="minimize-rmse"
        )

    # Train the model
    model = job.run(
        dataset=dataset,
        target_column=target_column,
        budget_milli_node_hours=1000,
        model_display_name=f"model_{os.path.basename(file_path).split('.')[0]}_{user_info['id']}",
        disable_early_stopping=False
    )


    return f"Successfully uploaded {os.path.basename(file_path)} to Google Cloud Storage and started AutoML training."
    

if __name__ == '__main__':
    app.debug = True
    app.run()