import os
import cherrypy
import pandas as pd
from requests_oauthlib import OAuth2Session
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google.cloud import storage, aiplatform
from google.auth import default

client_id = ''
client_secret = ''
authorization_base_url = "https://accounts.google.com/o/oauth2/auth"
token_url = "https://accounts.google.com/o/oauth2/token"
redirect_uri = "http://localhost:8080/callback"
user_info_url = "https://www.googleapis.com/oauth2/v1/userinfo"
project_id = ''
bucket_name = ''
location = ''

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
            'https://www.googleapis.com/auth/cloud-platform',
            'https://www.googleapis.com/auth/devstorage.full_control'
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
                        <input type="file" name="csv_file" /><br><br>
                        <label for="target_column">Target Column Name:</label>
                        <input type="text" id="target_column" name="target_column" /><br><br>
                        <input type="submit" value="Upload" />
                    </form>
                </body>
            </html>
        """

    @cherrypy.expose
    def upload_file(self, csv_file, target_column):
        """Process the uploaded CSV file, sanitize column names, upload it to Google Cloud Storage, and start AutoML training."""
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

        # Read and sanitize the CSV file
        df = pd.read_csv(upload_path)
        df.columns = [col.replace(' ', '_').replace('.', '_') for col in df.columns]
        
        sanitized_upload_path = os.path.join(self.upload_dir, f"sanitized_{csv_file.filename}")
        df.to_csv(sanitized_upload_path, index=False)

        # Use the OAuth token to initialize Google Cloud Storage client
        credentials = Credentials(
            token=oauth_token['access_token'],
            refresh_token=oauth_token.get('refresh_token'),
            token_uri=token_url,
            client_id=client_id,
            client_secret=client_secret
        )
        credentials.refresh(Request())
        storage_client = storage.Client(credentials=credentials, project=project_id)
        bucket = storage_client.bucket(bucket_name)

        # Upload the sanitized CSV file to the bucket
        blob_name = f"{csv_file.filename.split('.')[0]}_{user_info['id']}.csv"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(sanitized_upload_path)

        # Initialize Vertex AI client
        aiplatform.init(project=project_id, location=location, credentials=credentials)

        # Create dataset
        dataset = aiplatform.TabularDataset.create(
            display_name=f"{csv_file.filename.split('.')[0]}_{user_info['id']}",
            gcs_source=[f"gs://{bucket_name}/{blob_name}"]
        )

        # Define training job
        job = aiplatform.AutoMLTabularTrainingJob(
            display_name=f"training_job_{csv_file.filename.split('.')[0]}_{user_info['id']}",
            optimization_prediction_type="regression",
            optimization_objective="minimize-rmse"
        )

        # Train the model
        model = job.run(
            dataset=dataset,
            target_column=target_column.replace(' ', '_').replace('.', '_').replace('(', '').replace(')', ''),  # Use the sanitized target column
            budget_milli_node_hours=1000,
            model_display_name=f"model_{csv_file.filename.split('.')[0]}_{user_info['id']}",
            disable_early_stopping=False
        )

        return f"Successfully uploaded {csv_file.filename} to Google Cloud Storage and started AutoML training."

if __name__ == '__main__':
    cherrypy.quickstart(OAuth2App(), '/', config={
        '/': {
            'tools.sessions.on': True,
            'tools.sessions.timeout': 60
        }
    })