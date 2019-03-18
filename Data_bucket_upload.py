# METHOD ONE
#pip install google.cloud.storage

"""
destination_blob_name - what name should be called to the file uploaded eg: test.txt can be named as
t.txt where destination_blob_name is t.txt also the destination can also be set ie if we wanted to store in a
specific folder in the bucket eg : the/test/folder/t.txt
"""
from google.cloud import storage

def upload_blob(bucket_name, source_file_name, destination_blob_name,json_name):
    """Uploads a file to the bucket."""
    #verifying credintials
    storage_client = storage.Client.from_service_account_json(
                    '{}.json'.format(json_name)) #json file is generated from the google cloud for this particular project
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))

upload_blob("phinet.appspot.com","1.jpg","the/test2/1.jpg","phinet_storage_access")
print("upload done")


#
# # METHOD TWO
#
# from gcloud import storage
# from oauth2client.service_account import ServiceAccountCredentials
# import os
#
# credentials_dict = {
#     'type': 'service_account',
#     'client_id': os.environ['BACKUP_CLIENT_ID'],
#     'client_email': os.environ['BACKUP_CLIENT_EMAIL'],
#     'private_key_id': os.environ['BACKUP_PRIVATE_KEY_ID'],
#     'private_key': os.environ['BACKUP_PRIVATE_KEY'],
# }
# credentials = ServiceAccountCredentials.from_json_keyfile_dict(
#     credentials_dict
# )
# client = storage.Client(credentials=credentials, project='myproject')
# bucket = client.get_bucket('mybucket')
# blob = bucket.blob('myfile')
# blob.upload_from_filename('myfile')

