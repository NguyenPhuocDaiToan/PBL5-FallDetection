import cv2
import numpy as np
import time
import datetime
import firebase_admin
from firebase_admin import credentials, storage, db

# Initialize the Firebase app 
cred = credentials.Certificate("/home/yuu/Documents/PBL5/falldetect_new.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://falldetect-1c1b5-default-rtdb.firebaseio.com',
    'storageBucket': 'falldetect-1c1b5.appspot.com',
})
bucket = storage.bucket()

def upload_video_to_firebase(frames, count):
    filename = 'fall' + str(count) + '.mp4'
    writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 15, (640, 480))

    for frame in frames:
        writer.write(cv2.resize(frame, (640, 480)))
    writer.release()
        
    

    # Get a reference to the Firebase Storage bucket
    bucket = storage.bucket()

    # Date send video
    currentTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Upload the video to Firebase Storage
    blob = bucket.blob('Fall_detection$' + str(currentTime))
    blob.upload_from_filename(filename)

    # Get the download URL for the video
    expires_in = datetime.timedelta(hours=720)
    video_url = blob.generate_signed_url(expires_in)

    ref = db.reference('users')
    user_ref = ref.child('xiBWLpqpu3TIt7afsum57tLg2mu1')
    videos_ref = user_ref.child('videos')
    videos_ref.push().set({
        'time': currentTime,
        'url': video_url
    })
    print(f"Video {filename} uploaded to Firebase storage")