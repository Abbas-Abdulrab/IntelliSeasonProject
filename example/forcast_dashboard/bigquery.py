import cherrypy
from requests_oauthlib import OAuth2Session
import requests
import json
import os
import pandas as pd
from google.oauth2.credentials import Credentials
from google.cloud import bigquery
from google.api_core.exceptions import NotFound, Forbidden

# Allow OAuthlib to use HTTP for local testing
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# OAuth2 client details
client_id = ''
client_secret = ''
redirect_uri = 'http://localhost:8080/callback'
authorization_base_url = 'https://accounts.google.com/o/oauth2/auth'
token_url = 'https://oauth2.googleapis.com/token'
user_info_url = 'https://www.googleapis.com/oauth2/v1/userinfo'

# BigQuery setup
project_id = ''
dataset_id = ''
table_id = ''

class OAuth2App:
    def __init__(self):
        self.upload_dir = os.path.join(os.getcwd(), 'uploads')
        if not os.path.exists(self.upload_dir):
            os.makedirs(self.upload_dir)

    @cherrypy.expose
    def index(self):
        """Step 1: User Authorization."""
        scope = [
            'https://www.googleapis.com/auth/userinfo.profile', 
            'https://www.googleapis.com/auth/userinfo.email',
            'https://www.googleapis.com/auth/bigquery',
            'https://www.googleapis.com/auth/cloud-platform'
        ]
        google = OAuth2Session(client_id, redirect_uri=redirect_uri, scope=scope)
        authorization_url, state = google.authorization_url(authorization_base_url, access_type="offline")

        # Save the state in the session
        cherrypy.session['oauth_state'] = state
        raise cherrypy.HTTPRedirect(authorization_url)

    @cherrypy.expose
    def callback(self, state=None, code=None, scope=None, prompt=None, authuser=None):
        """Step 2: Retrieving an access token."""
        saved_state = cherrypy.session.get('oauth_state')
        if state != saved_state:
            return "State mismatch error. Possible CSRF attack."

        if code:
            google = OAuth2Session(client_id, redirect_uri=redirect_uri, state=saved_state)
            token = google.fetch_token(token_url, client_secret=client_secret, code=code)

            # Save the token in the session
            cherrypy.session['oauth_token'] = token

            # Retrieve user information
            google = OAuth2Session(client_id, token=token)
            user_info = google.get(user_info_url).json()
            cherrypy.session['user_info'] = user_info

            return f"Authorization successful. You can now upload CSV files. <a href='/upload'>Upload CSV</a>"
        else:
            return f"No authorization code found. Parameters received: state={state}, code={code}, scope={scope}"

    @cherrypy.expose
    def upload(self):
        """Upload endpoint to allow the user to upload a CSV file."""
        return """
            <html>
                <body>
                    <form action="upload_file" method="post" enctype="multipart/form-data">
                        <input type="file" name="csv_file" />
                        <input type="submit" value="Upload" />
                    </form>
                </body>
            </html>
        """

   @cherrypy.expose
    def upload_file(self, csv_file):
        """Process the uploaded CSV file and upload it to BigQuery."""
        oauth_token = cherrypy.session.get('oauth_token')
        user_info = cherrypy.session.get('user_info')
        if not oauth_token or not user_info:
            return "No access token or user information available. Please authenticate first."

        upload_path = os.path.join(self.upload_dir, csv_file.filename)
        with open(upload_path, 'wb') as out:
            while True:
                data = csv_file.file.read(8192)
                if not data:
                    break
                out.write(data)

        # Try reading the CSV file with different encodings and handle bad lines
        try:
            df = pd.read_csv(upload_path, encoding='utf-8', on_bad_lines='skip')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(upload_path, encoding='latin1', on_bad_lines='skip')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(upload_path, encoding='iso-8859-1', on_bad_lines='skip')
                except UnicodeDecodeError:
                    return "Unable to read the file with utf-8, latin1, or iso-8859-1 encoding."

        # Ensure correct data types
        for column in df.select_dtypes(include=['float64']).columns:
            df[column] = df[column].astype('float64')
        for column in df.select_dtypes(include=['int64']).columns:
            df[column] = df[column].astype('int64')
        for column in df.select_dtypes(include=['object']).columns:
            df[column] = df[column].astype('str')

        # Add user information to the DataFrame
        df['user_email'] = user_info['email']
        df['user_id'] = user_info['id']

        # Dynamically create schema based on the DataFrame columns
        schema = []
        for column in df.columns:
            if df[column].dtype == 'float64':
                schema.append(bigquery.SchemaField(column, bigquery.enums.SqlTypeNames.FLOAT))
            elif df[column].dtype == 'int64':
                schema.append(bigquery.SchemaField(column, bigquery.enums.SqlTypeNames.INTEGER))
            else:
                schema.append(bigquery.SchemaField(column, bigquery.enums.SqlTypeNames.STRING))

        credentials = Credentials(
            token=oauth_token['access_token'],
            refresh_token=oauth_token['refresh_token'],
            token_uri=token_url,
            client_id=client_id,
            client_secret=client_secret
        )
        bigquery_client = bigquery.Client(credentials=credentials, project=project_id)

        # Check if the dataset exists, create it if it doesn't
        try:
            dataset_ref = bigquery_client.dataset(dataset_id)
            dataset = bigquery_client.get_dataset(dataset_ref)
        except NotFound:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = "US"
            bigquery_client.create_dataset(dataset)
            print(f"Created dataset {dataset_id}")
        except Forbidden as e:
            return f"Access Denied: {str(e)}"

        # Create a new table with name filename_userid
        table_name = f"{csv_file.filename.split('.')[0]}_{user_info['id']}"
        table_ref = dataset_ref.table(table_name)

        table = bigquery.Table(table_ref, schema=schema)

        try:
            table = bigquery_client.create_table(table)
            print(f"Created table {table.table_id}")
        except Exception as e:
            return f"Failed to create table: {str(e)}"

        job_config = bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_APPEND)

        load_job = bigquery_client.load_table_from_dataframe(df, table_ref, job_config=job_config)
        load_job.result()

        return f"Successfully uploaded {csv_file.filename} to BigQuery for user {user_info['email']} in table {table_name}."


if __name__ == '__main__':
    cherrypy.quickstart(OAuth2App(), '/', config={
        '/': {
            'tools.sessions.on': True,
            'tools.sessions.timeout': 60
        }
    })