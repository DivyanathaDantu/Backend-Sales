import os

import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv
from pathlib import Path

dotenv_path = Path('../local.env')
load_dotenv(dotenv_path=dotenv_path)


def initialize_firestore():
    cred = credentials.Certificate(os.getenv('FIREBASE_ADMIN_SDK_PATH'))
    firebase_admin.initialize_app(cred)


def get_firestore_files(search_file_name):
    if firestore is None:
        initialize_firestore()
    db = firestore.client()
    docs = db.collection(u'sales_data_csv_files').where(u'file_name', u'>=', search_file_name). \
        where("file_name", "<=", search_file_name + "\uf8ff"). \
        limit(10). \
        stream()
    return docs
