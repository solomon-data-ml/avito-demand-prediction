from __future__ import division, print_function
# coding=utf-8
import os
import warnings
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',filename=os.path.realpath("gs.log"), level=logging.INFO,datefmt='%Y-%m-%d %H:%M:%S');
 # Flask utils
from flask import Flask, request, render_template,send_from_directory
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import pickle
import os
from tensorflow.keras.preprocessing import image
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from numpy import zeros
import tensorflow as tf
import math
import tensorflow
from tensorflow.keras.initializers import he_normal
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense,Input, Dropout,Conv1D
import datetime

# Define a flask app
app = Flask(__name__)

max_length = 100
embed_dimension= 300

# Model saved with Keras model.save()
MODEL_PATH = 'avecto_inception.h5'
LABEL_ENCODER_PATH='lbl_encoders_fit_avito.pkl'
TOKENIZER_PATH='nlp_tokenizers.pkl'
EMBEDDING_PATH='final_embedding_matrix1'  

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

def na_handler(df,column):
    df[column].fillna(value='NA', inplace=True)
    return df

def datatype_handler(df,column,dataype):
    df[column] = df[column].astype(dataype)
    return df


def get_blur_value(imagepath,rootpath):
    value = 0
    if(imagepath is not float(np.nan)):  
        path = rootpath+imagepath+'.jpg'  
         
        try:
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            value = cv2.Laplacian(gray, cv2.CV_64F).var()
        except:
            return value
        return round(value)

def engineering_features(df,case):
    print('case is ',case)
    df['title_length'] = df['title'].apply(lambda x: len(str(x)))
    df['description_length'] = df['description'].apply(lambda x: len(str(x)))
    df['param123'] = (df['param_1']+'_'+df['param_2']+'_'+df['param_3']).astype(str)
    df['title_description']= (df['title']+" "+df['description']).astype(str).str.lower()
    del df['description'], df['title']
    df['price'] = np.log1p(df['price'])
    df['avg_days_up_user'] = np.log1p(df['avg_days_up_user'])
    df['avg_times_up_user'] = np.log1p(df['avg_times_up_user'])
    df['n_user_items'] = np.log1p(df['n_user_items'])
    df['item_seq_number'] = np.log(df['item_seq_number'])
    df["activation_date"] = pd.to_datetime(df['activation_date'])
    df["weekday"] = df['activation_date'].dt.weekday
    print('Calculating Blurness Value : Started')
    if(case=="test"):
        rootpath = "C:\\Users\\sigmoid\\Documents\\Projects\\Aaic\\Assignments\\Avito_Demand_Prediction\\test_jpg\\"
        
    else:
        rootpath = "C:\\Users\\sigmoid\\Documents\\Projects\\Aaic\\Assignments\\Avito_Demand_Prediction\\train_jpg\\"
        
    df['blurness'] = df.apply(lambda x: get_blur_value(x['image'],rootpath),axis=1)
    print('Calculating Blurness Value : Completed')
    gc.collect()
    return df

def label_encoding_fit(df,columns):
    lbl_fits = {}
    for col in columns:
        lbl_enc = LabelEncoder()
        lbl_enc.fit(df[col])
        lbl_fits[col] = lbl_enc
    pickle.dump(lbl_fits, open(LABEL_ENCODER_PATH, 'wb'))
    return lbl_fits

def label_encoding_transform_runtime(df,lbl_encoder_container,categorical_features):
    for col in categorical_features:
        lbl_encoder = lbl_encoder_container[col]
        le_dict = dict(zip(lbl_encoder.classes_, lbl_encoder.transform(lbl_encoder.classes_)))
        df[col] = df[col].apply(lambda x: le_dict.get(x, 112233445566))
    return df

def tokenize_data(df,columns,maxwords):
    nlp_tokenizers={}
    for col in columns:
        tokenizer = text.Tokenizer(num_words = maxwords)
        textdata = np.hstack([df[col]])
        tokenizer.fit_on_texts(textdata)
        nlp_tokenizers[col] = tokenizer
    pickle.dump(nlp_tokenizers, open(TOKENIZER_PATH, 'wb'))
    return nlp_tokenizers

def transform_tokens_to_sequence(df,columns,tokenizers):
    for col in columns:
        df[col] = tokenizers[col].texts_to_sequences(df[col].str.lower()) 
    return df



def process_handlers(df,case):
 
    na_handler(df,'region')
    na_handler(df,'parent_category_name')
    na_handler(df,'category_name')
    na_handler(df,'city')
    na_handler(df,'param_1')
    na_handler(df,'param_2')
    na_handler(df,'param_3')
    na_handler(df,'image_top_1')
    df['price'] = df['price'].fillna(0).astype('float32')
    df['avg_days_up_user'] = df['avg_days_up_user'].fillna(0) 
    df['avg_times_up_user'] = df['avg_times_up_user'].fillna(0) 
    df['n_user_items'] = df['n_user_items'].fillna(0)
    df['image'] = df['image'].fillna(df.groupby('parent_category_name')['image'].apply(lambda x: x.fillna(x.mode()[0])))
 
    df = engineering_features(df,case)
       
    df = datatype_handler(df,'region','category')
    df = datatype_handler(df,'parent_category_name','category')
    df = datatype_handler(df,'category_name','category')
    df = datatype_handler(df,'city','category')
    df = datatype_handler(df,'param_1','str')
    df = datatype_handler(df,'param_2','str')
    df = datatype_handler(df,'param_3','str')
    df = datatype_handler(df,'image_top_1','str')
    df = datatype_handler(df,'avg_days_up_user','uint32')
    df = datatype_handler(df,'avg_times_up_user','uint32')
    df = datatype_handler(df,'n_user_items','uint32')
    
    del df['param_2'], df['param_3']
    gc.collect()

def load_data(filename):
    datatype_mapper = {
                'price': 'float32',
                'deal probability': 'float32',
                'item_seq_number': 'uint32'
    }

    # No user_id
    columns = ['item_id', 'user_id','image', 'image_top_1', 'region', 'city', 'parent_category_name', 'category_name', 'param_1', 'param_2', 'param_3', 'title', 'description', 'price', 'item_seq_number', 'activation_date', 'deal_probability']
    df = pd.read_csv(filename, parse_dates=["activation_date"], usecols = columns, dtype = datatype_mapper)
    #train_features = pd.read_csv('data/aggregated_features.csv')
    #df = train.merge(train_features, on = ['user_id'], how = 'left')
    #del train_features
    #del train
    #gc.collect()


    y_train = np.array(df['deal_probability'])

    del df['deal_probability']
    gc.collect()

   
    del df['item_id'], df['user_id']
    gc.collect()
    
    return df,y_train

max_seq_title_description_length = 100
max_words_title_description = 200000
max_region = np.max(df.region.max())+2
max_city= np.max(df.city.max())+2
max_category_name = np.max(df.category_name.max())+2
max_parent_category_name = np.max(df.parent_category_name.max())+2
max_param_1 = np.max(df.param_1.max())+2
max_param123 = np.max(df.param123.max())+2
max_image_code = np.max(df.image_top_1.max())+2
vocab_size = 748126
EMBEDDING_DIM1 = 300

def preprocess(filename,df,isPredict=True):
    logging.info('Preprocessing started')
    if(isPredict==True):
        df = process_handlers(df,'test')
        logging.info('20% Completed')
        lblenc_columns = ['region','city','category_name','parent_category_name','param_1','param123','image_top_1']
        lbl_fits = pickle.load(open(LABEL_ENCODER_PATH, 'rb'))
        df = label_encoding_transform_runtime(df,lbl_fits,lblenc_columns)
        logging.info('50% Completed')
        image_mode = df.groupby('parent_category_name')['image'].apply(lambda x: x.fillna(x.mode()[0]))
        df['image'] = df['image'].fillna(image_mode)
        nlpcolumns=['title_description']
        tokenizers = pickle.load(open(TOKENIZER_PATH, 'rb'))
        df = transform_tokens_to_sequence(df,nlpcolumns,tokenizers)
        
    else:
        df,y = load_data(filename)
        df = process_handlers(df,'train')
        logging.info('20% Completed')
        lblenc_columns = ['region','city','category_name','parent_category_name','param_1','param123','image_top_1']
        lbl_fits = label_encoding_fit(df,lblenc_columns)
        logging.info('30% Completed')
        df = label_encoding_transform(df,lbl_fits,lblenc_columns)
        logging.info('50% Completed')
        nlpcolumns=['title_description']
        logging.info('90% Completed')
        tokenizers =    tokenize_data(df,nlpcolumns,max_words_title_description)  
        df = transform_tokens_to_sequence(df,nlpcolumns,tokenizers)
         
    logging.info('100% Completed')
    logging.info('Preprocessing Completed!')

    return df

embedding_matrix1 =  pickle.load(open(EMBEDDING_PATH, 'rb'))

 
def model_predict(filename,tokenizers,model,title,description,region,city,parent_category_name,category_name,
param_1,param_2,param_3,user_type,weekday,price,item_seq_number,image_top_1,imagefilename
):

    natural_language_features = ['title','description']
    categorical_features = ['region' , 'city', 'parent_category_name', 'category_name','param_1', 
    'param_2','param_3','user_type','weekday']
    numerical_features=['blurness','price','item_seq_number','image_top_1']

    data = [{ 'title' : title,
                'description' : description,
                'region' : region,
                'city' : city,
                'parent_category_name' : parent_category_name,
                'category_name' : category_name,
                'param_1' : param_1,
                'param_2' : param_2,
                'param_3' : param_3,
                'user_type' : user_type,
                'weekday' : weekday,
                'price' : price,
                'item_seq_number' : item_seq_number,
                'image_top_1' : image_top_1,
                'image':imagefilename
            }]

    # Creates DataFrame.
    df = pd.DataFrame(data)
    df = preprocess(None,df, True)
    logging.info("Encoding Started")
     
    logging.info("Prediction Started")
    predicts = model.predict([df.region.values, 
              df.city.values, 
              df.parent_category_name.values,
              df.category_name.values,
              df.param_1.values,
              df.price.values, 
              df.item_seq_number.values, 
              df.avg_days_up_user.values,
              df.avg_times_up_user.values,
              df.n_user_items.values, 
              df.image_top_1.values, 
              df.param123.values, 
              df.title_description.values,
              df.title_length.values,
              df.description_length.values,
              df.weekday.values,
              df.blurness.values,
              df.image.values
              ])
    logging.info("Prediction Completed")

    pred = predicts[0]

    logging.info("Predicted  - "+str(pred))

    return str(pred)

 


@app.route('/predict', methods=['GET', 'POST'])
def upload():

    logging.info('Request Parsed')

    lbl_encoder_container = pickle.load(open(LABEL_ENCODER_PATH, 'rb'))

    model = pickle.load(open(MODEL_PATH, 'rb'))

    out = model_predict('',
                        lbl_encoder_container,
                        model,
                        region=  int(request.args['region']),
                        city= int(request.args['city']),
                        parent_category_name=  int(request.args['parent_category_name']) ,
                        category_name=int(request.args['category_name']),
                        param_1=int(request.args['param_1']),
                        price = int(request.args['price']),
                        item_seq_number=str(request.args['item_seq_number']),
                        param123=str(request.args['param123']),
                        title=str(request.args['title']),
                        description=str(request.args['description']),
                        image_path=str(request.args['image_path'])
                        )
    return str(out)


@app.route('/downloadsample/<path:filename>', methods=['GET', 'POST'])
def download_results(filename):
    return send_from_directory('uploads', filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
