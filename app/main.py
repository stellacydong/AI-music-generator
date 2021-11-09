# run by typing python3 main.py in a terminal 
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, url_for, render_template
from flask import send_from_directory
from utils import get_base_url, allowed_file, and_syntax
from flask_cors import cross_origin



import os
import json
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Activation, Embedding


# In[2]:


data_directory = "Weights_Data/"
data_file = "Data_Tunes.txt"
charIndex_json = "char_to_index.json"
model_weights_directory = 'Weights_Data/LSTM1_Model_Weights/'

BATCH_SIZE = 16
SEQ_LENGTH = 64


# In[3]:


def make_model(unique_chars):
    model = Sequential()
    
    model.add(Embedding(input_dim = unique_chars, output_dim = 512, batch_input_shape = (1, 1))) 
  
    for i in range(3):
        model.add(LSTM(256, return_sequences = True, stateful = True))
        model.add(Dropout(0.2))

    
    model.add((Dense(unique_chars)))
    model.add(Activation("softmax"))
    
    return model


# In[4]:


with open(os.path.join(data_directory, charIndex_json)) as f:
    char_to_index = json.load(f)
index_to_char = {i:ch for ch, i in char_to_index.items()}
unique_chars = len(index_to_char)


# In[5]:


epoch_num = 100
model = make_model(unique_chars)
model.load_weights(model_weights_directory + "Weights_{}.h5".format(epoch_num))


# In[6]:


def generate_sequence(epoch_num, initial_index, seq_length):
    with open(os.path.join(data_directory, charIndex_json)) as f:
        char_to_index = json.load(f)
    index_to_char = {i:ch for ch, i in char_to_index.items()}
    unique_chars = len(index_to_char)
    
    model = make_model(unique_chars)
    model.load_weights(model_weights_directory + "Weights_{}.h5".format(epoch_num))
     
    sequence_index = [initial_index]
    
    for _ in range(seq_length):
        batch = np.zeros((1, 1))
        batch[0, 0] = sequence_index[-1]
        
        predicted_probs = model.predict_on_batch(batch).ravel()
        sample = np.random.choice(range(unique_chars), size = 1, p = predicted_probs)
        
        sequence_index.append(sample[0])
    
    seq = ''.join(index_to_char[c] for c in sequence_index)
    
    cnt = 0
    for i in seq:
        cnt += 1
        if i == "\n":
            break
    seq1 = seq[cnt:]
    
    cnt = 0
    for i in seq1:
        cnt += 1
        if i == "\n" and seq1[cnt] == "\n":
            break
    seq2 = seq1[:cnt]
    
    return seq2


# ar = int(input("Enter any number between 0 to 86 which will be given as initial charcter to model for generating sequence: "))

def music_generator(ar):
    ep = 100
    ln = 800
    music_abc = generate_sequence(ep, ar, ln)
    return music_abc

# setup the webserver
'''
    coding center code
    port may need to be changed if there are multiple flask servers running on same server
    comment out below three lines of code when ready for production deployment
'''
port = 10009
base_url = get_base_url(port)
app = Flask(__name__, static_url_path=base_url+'static')



#@app.route('/')
@app.route(base_url)
def home():
    return render_template('home.html')


@app.route(base_url + "/result",methods=["GET","POST"])
@cross_origin()
def result():
    if request.method=="POST":
        start_num = request.form.get('number', type=int)
        prediction=music_generator(start_num)
        return render_template('home.html',prediction_text=prediction)
    return render_template("home.html")


#@app.route('/information')
@app.route(base_url + "/information")
def info():
    return render_template('info.html')

#@app.route('/samples')
@app.route(base_url + "/samples")
def samples():
    return render_template('samples.html')


#@app.route('/team')
@app.route(base_url + "/team")
def team():
    return render_template('team.html')

# #@app.route('/files/<path:filename>')
# @app.route(base_url + '/files/<path:filename>')
# def files(filename):
#     return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

if __name__ == "__main__":
    '''
    coding center code
    '''
    # IMPORTANT: change the cocalcx.ai-camp.org to the site where you are editing this file.
    website_url = 'coding.ai-camp.org'
    print(f"Try to open\n\n    https://{website_url}" + base_url + '\n\n')

    # remove debug=True when deploying it
    app.run(host = '0.0.0.0', port=port, debug=True)
    import sys; sys.exit(0)

    '''
    cv scaffold code
    '''
    # Only for debugging while developing
    # app.run(port=80, debug=True)

