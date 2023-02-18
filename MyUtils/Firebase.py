import os
import sys

import firebase_admin
from firebase_admin import credentials, firestore
import os
from dotenv import load_dotenv


def initialize_firestore():
    load_dotenv()
    cred_path = os.getenv('FIREBASE_PATH')
    bundle_dir = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
    path_to_dat = os.path.abspath(os.path.join(bundle_dir, cred_path))
    # print("divi test 2", path_to_dat)
    cred = credentials.Certificate(path_to_dat)
    firebase_admin.initialize_app(cred)


def get_firestore_files(search_file_name):
    try:
        firestore.client()
    except Exception as e:
        print(e)
        initialize_firestore()
    db = firestore.client()
    docs = db.collection(u'sales_data_csv_files').where(u'file_name', u'>=', search_file_name). \
        where("file_name", "<=", search_file_name + "\uf8ff"). \
        limit(10). \
        stream()
    return docs
