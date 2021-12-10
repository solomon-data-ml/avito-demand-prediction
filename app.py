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
MODEL_PATH = 'avito.h5'
ENCODER_PATH='lbl_encoders_v2.pkl'

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')



def get_blur_value(imagepath):
    value = 0
    if(imagepath is not float(np.nan)):  
        path = imagepath+'.jpg'  
        try:
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            value = cv2.Laplacian(gray, cv2.CV_64F).var()
        except:
            return value
        return round(value)

def preprocess(filename,df,isPredict=True):
    logging.info('Preprocessing started')
    if(!isPredict):
        df = pd.read_csv(filename)
        logging.info('Preprocessing : 5% Completed')
        df.drop('item_id', axis=1, inplace=True)
        df.drop('user_id', axis=1, inplace=True)
        df['param_1'].fillna("NA", inplace=True)
        df['param_2'].fillna("NA", inplace=True)
        df['param_3'].fillna("NA", inplace=True)
        logging.info('25% Completed')

    logging.info('Preprocessing : 35% Completed')
    df["activation_date"] = pd.to_datetime(df['activation_date'])
    df['title_length'] = df['title'].apply(lambda x: len(str(x)))
    df['description_length'] = df['description'].apply(lambda x: len(str(x)))
    df["weekday"] = df['activation_date'].dt.weekday
    logging.info('Preprocessing : 45% Completed')
    df['blurness'] = df.apply(lambda x: get_blur_value(x['image']),axis=1)
    df["blurness"].fillna(0, inplace=True)
    df["price"] = pd.to_numeric(df["price"])
    df["item_seq_number"] = pd.to_numeric(df["item_seq_number"])
    logging.info('Preprocessing : 80% Completed')
    df["image_top_1"] = pd.to_numeric(df["image_top_1"])
    df["deal_probability"] = pd.to_numeric(df["deal_probability"])
    df["blurness"] = pd.to_numeric(df["blurness"])
    df["weekday"] = pd.to_numeric(df["weekday"])
    df["blurness"].fillna(0, inplace=True)
    df = df[df['description'].notna()]
    logging.info('Preprocessing Completed!')

    return df


 
def model_predict(filename,tokenizers,model,title,description,region,city,parent_category_name,category_name,
param_1,param_2,param_3,user_type,weekday,price,item_seq_number,image_top_1,
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
                'image_top_1' : image_top_1
            }]

    # Creates DataFrame.
    df = pd.DataFrame(data)
    df = preprocess(None,df, True)
    logging.info("Encoding Started")
    title_padded, title_vocab_size, title_embedding                        = get_embeddings_(df["title"].values)
    description_padded, description_vocab_size, description_embedding      = get_embeddings_(df["description"].values)

    region = np.array(tokenizers["region"].texts_to_sequences(df["region"].values))
    city = np.array(tokenizers["city"].texts_to_sequences(df["city"].values))
    parent_category_name = np.array(tokenizers["parent_category_name"].texts_to_sequences(df["parent_category_name"].values))
    category_name = np.array(tokenizers["category_name"].texts_to_sequences(df["category_name"].values))
    param_1 = np.array(tokenizers["param_1"].texts_to_sequences(df["param_1"].values))
    param_2 = np.array(tokenizers["param_2"].texts_to_sequences(df["param_2"].values))
    param_3 = np.array(tokenizers["param_3"].texts_to_sequences(df["param_3"].values))
    user_type = np.array(tokenizers["user_type"].texts_to_sequences(df["user_type"].values))
    weekday = np.array(tokenizers["weekday"].texts_to_sequences(df["weekday"].map(str).values))
    logging.info("Encoding Completed")

    logging.info("Prediction Started")
    predicts = model.predict([title_padded,\
           description_padded,\
           region,\
           city,\
           parent_category_name,\
           category_name,\
           param_1,\
           param_2,\
           param_3,\
           user_type,\
           weekday,\
           df["blurness"].values,
           df["price"].values,
           df["item_seq_number"].values,
           df["image_top_1"].values,
           df["title_length"].values,
           df["description_length"].values
          ])
    logging.info("Prediction Completed")

    pred = predicts[0]

    logging.info("Predicted  - "+str(pred))

    return str(pred)

def get_encoded_documents(data):
    t = Tokenizer()
    t.fit_on_texts(data)
    vocab_size = len(t.word_index) + 1
    # integer encode the documents
    encoded_docs = t.texts_to_sequences(data)
    return t,vocab_size,encoded_docs

 def get_padded_docs(max_length,encoded_docs):
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    return padded_docs

def get_embeddings_index(filenamewithpath):
    embeddings_index = dict()
    f = open(filenamewithpath, 'r', encoding="utf-8")
    for line in tqdm(f):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    return embeddings_index

def generate_embedding_matrix(vocab_size,tokenizer,embeddings_index):
    embedding_matrix = zeros((vocab_size, 300))
    for word, i in tqdm(tokenizer.word_index.items()):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def get_embeddings_(data):
    tokenizer, vocabsize, encoded_data = get_encoded_documents(data)
    padded_data = get_padded_docs(max_length,encoded_data)
    embedding_mat = generate_embedding_matrix(vocabsize,tokenizer,embeddings_index)
    return padded_data,vocabsize,embedding_mat

tokenizer_container = {}
def tknzr_fit(col,train, test):
    tknzr = Tokenizer(filters='', lower=False, split='뷁', oov_token='oov' )
    tknzr.fit_on_texts(train)
    tokenizer_container[col] = tknzr
    return np.array(tknzr.texts_to_sequences(train)), np.array(tknzr.texts_to_sequences(test)), tknzr

image_size=126
def get_image_array_cv2(filename):
    
    try:
        img_path = r'train_jpg\\'+filename+'.jpg'         
        img = image.load_img(img_path)
        img = image.img_to_array(img)
        resized = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
     
        resized =resized /255.0
        #img_pil = array_to_img(resized)  --> To convert back to image for testing purpose
    except:
        return np.zeros((image_size, image_size, 3), dtype=np.uint8)
    
    return resized

class DataGenerator_(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size=4, dim=(1), shuffle=False):
         
        self.dim = dim
        self.batch_size = batch_size
        self.dataset = dataset
        self.shuffle = shuffle
        self.indexes =  np.arange(len(self.dataset[0]))
        self.on_epoch_end()
    def __len__(self):
         
        return math.ceil(len(self.dataset[0]) / self.batch_size)
    def __getitem__(self, index):
        # Generate indexes of the batch
        idxs = [i for i in range(index*self.batch_size,(index+1)*self.batch_size)]
        # Find list of IDs
        list_IDs_temp = [self.indexes[k] for k in idxs]

       #image_data= np.array([get_image_array(file_name) for file_name in self.dataset[17][list_IDs_temp]])
        image_data= np.array([get_image_array_cv2(file_name) for file_name in self.dataset[17][list_IDs_temp]])
    
        #image_data = image_data/255.0
        
        title = self.dataset[0][list_IDs_temp]#.reshape(-1)
        description = self.dataset[1][list_IDs_temp]#.reshape(-1)
        region = self.dataset[2][list_IDs_temp]#.reshape(-1)
        city = self.dataset[3][list_IDs_temp]#.reshape(-1)
        parent_category_name = self.dataset[4][list_IDs_temp]
        category_name = self.dataset[5][list_IDs_temp]
        param1 = self.dataset[6][list_IDs_temp]
        param2 = self.dataset[7][list_IDs_temp] 
        param3 = self.dataset[8][list_IDs_temp]
        usertype = self.dataset[9][list_IDs_temp]
        week = self.dataset[10][list_IDs_temp]
        blurness = self.dataset[11][list_IDs_temp]
        price = self.dataset[12][list_IDs_temp]
        item_seq_number = self.dataset[13][list_IDs_temp]
        image_top_1 = self.dataset[14][list_IDs_temp]
        title_length = self.dataset[15][list_IDs_temp]
        description_length = self.dataset[16][list_IDs_temp]
         
        y = self.dataset[18][list_IDs_temp]

        return  [image_data,\
           title,\
           description,\
           region,\
           city,\
           parent_category_name,\
           category_name,\
           param1,\
           param2,\
           param3,\
           usertype,\
           week,\
           blurness,
           price,
           item_seq_number,
           image_top_1,
           title_length,
           description_length
          ],\
           y
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.dataset[0]))
        #if self.shuffle == True:
            #np.random.shuffle(self.indexes)


def create_inception_model(tr_title_vocab_size,
    tr_title_embedding,
    tr_description_vocab_size,
    tr_description_embedding,
    tknzr_region,
    tknzr_city,
    tknzr_par_cat,
    tknzr_category_name,
    tknzr_param1,
    tknzr_param2,
    tknzr_param3,
    tknzr_usertype,
    tknzr_week):

    filtersize = 32
    kernelsize = 5
    
    henormal_init = tensorflow.keras.initializers.he_normal() 
    regularizer = tensorflow.keras.regularizers.l2(0.1)

    image_model = tf.keras.applications.InceptionV3(include_top=False,weights="imagenet", input_shape=(image_size, image_size,3))
    new_input = image_model.input
    input_image = image_model.layers[-1].output
    flat_image = Flatten()(input_image)
    #============================================= NLP Features =============================================#
    input_title = Input(shape=(max_length,))
    e_model_title = Embedding(tr_title_vocab_size, 300,weights=[tr_title_embedding], input_length=max_length, trainable=False)(input_title)
    
    conv1 = Conv1D(filters=filtersize, kernel_size=kernelsize, kernel_initializer=he_normal(seed=0),activation='sigmoid', padding='valid',kernel_regularizer=regularizers.l2(l=0.01))(e_model_title)
    conv2 = Conv1D(filters=filtersize, kernel_size=kernelsize,kernel_initializer=he_normal(seed=0), activation='sigmoid',padding='valid',kernel_regularizer=regularizers.l2(l=0.01))(e_model_title)
    conv3 = Conv1D(filters=filtersize, kernel_size=kernelsize, kernel_initializer=he_normal(seed=0),activation='sigmoid',padding='valid',kernel_regularizer=regularizers.l2(l=0.01))(e_model_title) 
    concat1 = tf.keras.layers.Concatenate(axis=1)([conv1,conv2,conv3])
    maxpool1 = tf.keras.layers.MaxPooling1D(pool_size=5,  padding='valid')(concat1)
    conv4 = Conv1D(filters=filtersize, kernel_size=kernelsize, kernel_initializer=he_normal(seed=0),activation='sigmoid',padding='valid',kernel_regularizer=regularizers.l2(l=0.01))(maxpool1)
    conv5 = Conv1D(filters=filtersize, kernel_size=kernelsize,kernel_initializer=he_normal(seed=0), activation='sigmoid',padding='valid',kernel_regularizer=regularizers.l2(l=0.01))(maxpool1)
    conv6 = Conv1D(filters=filtersize, kernel_size=kernelsize, kernel_initializer=he_normal(seed=0),activation='sigmoid',padding='valid',kernel_regularizer=regularizers.l2(l=0.01))(maxpool1)
    concat2 = tf.keras.layers.Concatenate(axis=1)([conv4,conv5, conv6])
    x = tf.keras.layers.MaxPooling1D(pool_size=2,  padding='valid')(concat2)
    x = Conv1D(filters=12, kernel_size=kernelsize,kernel_initializer=he_normal(seed=0), activation='sigmoid')(x)
    flat_title = Flatten()(x)
    

    input_desc = Input(shape=(max_length,))
    e_model_description = Embedding(tr_description_vocab_size, 300,weights=[tr_description_embedding], input_length=max_length, trainable=False)(input_desc)
    conv1 = Conv1D(filters=filtersize, kernel_size=kernelsize, kernel_initializer=he_normal(seed=0),activation='sigmoid', padding='valid',kernel_regularizer=regularizers.l2(l=0.01))(e_model_description)
    conv2 = Conv1D(filters=filtersize, kernel_size=kernelsize,kernel_initializer=he_normal(seed=0), activation='sigmoid',padding='valid',kernel_regularizer=regularizers.l2(l=0.01))(e_model_description)
    conv3 = Conv1D(filters=filtersize, kernel_size=kernelsize, kernel_initializer=he_normal(seed=0),activation='sigmoid',padding='valid',kernel_regularizer=regularizers.l2(l=0.01))(e_model_description) 
    concat1 = tf.keras.layers.Concatenate(axis=1)([conv1,conv2,conv3])
    maxpool1 = tf.keras.layers.MaxPooling1D(pool_size=5,  padding='valid')(concat1)
    conv4 = Conv1D(filters=filtersize, kernel_size=kernelsize, kernel_initializer=he_normal(seed=0),activation='sigmoid',padding='valid',kernel_regularizer=regularizers.l2(l=0.01))(maxpool1)
    conv5 = Conv1D(filters=filtersize, kernel_size=kernelsize,kernel_initializer=he_normal(seed=0), activation='sigmoid',padding='valid',kernel_regularizer=regularizers.l2(l=0.01))(maxpool1)
    conv6 = Conv1D(filters=filtersize, kernel_size=kernelsize, kernel_initializer=he_normal(seed=0),activation='sigmoid',padding='valid',kernel_regularizer=regularizers.l2(l=0.01))(maxpool1)
    concat2 = tf.keras.layers.Concatenate(axis=1)([conv4,conv5, conv6])
    x = tf.keras.layers.MaxPooling1D(pool_size=2,  padding='valid')(concat2)
    x = Conv1D(filters=12, kernel_size=kernelsize,kernel_initializer=he_normal(seed=0), activation='sigmoid')(x)
    flat_desc = Flatten()(x)
    #============================================= Categorical Features =============================================#
    input_reg = Input(shape=(1, ), name='input_region')
    embed_reg = Embedding(len(tknzr_region.word_index), 10, name='embed_region')(input_reg)
    flat_reg = Flatten()(embed_reg)

    input_city = Input(shape=(1, ), name='input_city')
    embed_city = Embedding(len(tknzr_city.word_index),10, name='embed_city' )(input_city)
    flat_city = Flatten()(embed_city)

    input_pcn = Input(shape=(1, ), name='input_parent_category_name')
    embed_pcn = Embedding(len(tknzr_par_cat.word_index), 10, name='embed_parent_category_name')(input_pcn)
    flat_pcn = Flatten()(embed_pcn)

    input_cn = Input(shape=(1, ), name='input_category_name')
    embed_cn = Embedding(len(tknzr_category_name.word_index), 10, name="embed_category_name" )(input_cn)
    flat_cn = Flatten()(embed_cn)

    input_param1 = Input(shape=(1, ), name='input_param1')
    embed_param1 = Embedding(len(tknzr_param1.word_index), 20, name='embed_param1')(input_param1)
    flat_param1 = Flatten()(embed_param1)

    input_param2 = Input(shape=(1, ), name='input_param2')
    embed_param2 = Embedding(len(tknzr_param2.word_index), 20, name='embed_param2')(input_param2)
    flat_param2 = Flatten()(embed_param2)

    input_param3 = Input(shape=(1, ), name='input_param3')
    embed_param3 = Embedding(len(tknzr_param3.word_index), 20, name='embed_param3')(input_param3)
    flat_param3 = Flatten()(embed_param3)

    input_ut = Input(shape=(1, ), name='input_user_type')
    embed_ut = Embedding(len(tknzr_usertype.word_index), 10, name='embed_user_type' )(input_ut)
    flat_ut = Flatten()(embed_ut)

    input_week = Input(shape=(1, ), name='input_week')
    embed_week = Embedding(len(tknzr_week.word_index), 10, name='embed_week' )(input_week)
    flat_week = Flatten()(embed_week)
    #============================================= Numerical Features =============================================#
    input_blurness = Input(shape=(1, ), name='input_blurness')
    dense_blurness = Dense(1, 
                           activation='sigmoid',
                           name='dense_blurness',
                           kernel_initializer=he_normal(seed=0),
                           kernel_regularizer=regularizers.l2(l=0.01)
                           )(input_blurness)

    input_price = Input(shape=(1, ), name='input_price')
    dense_price = Dense(1,
                        activation='sigmoid',
                        name='dense_price',
                        kernel_initializer=he_normal(seed=0),
                        kernel_regularizer=regularizers.l2(l=0.01)
                        )(input_price)

    input_item_seq_numer = Input(shape=(1, ), name='input_item_seq_numer')
    dense_item_seq_numer = Dense(1, 
                                 activation='sigmoid',
                                 name='dense_itemseq',
                                 kernel_initializer=he_normal(seed=0),
                                 kernel_regularizer=regularizers.l2(l=0.01)
                                 )(input_item_seq_numer)

    input_image_top_1 = Input(shape=(1, ), name='input_image_top_1')
    dense_image_top_1 = Dense(1,
                              activation='sigmoid',
                              name='dense_top1',
                              kernel_initializer=he_normal(seed=0),
                              kernel_regularizer=regularizers.l2(l=0.01)
                              )(input_image_top_1)

    input_title_length = Input(shape=(1, ), name='input_title_length')
    dense_title_length = Dense(1, 
                               activation='sigmoid',
                               name='dense_titlelength',
                               kernel_initializer=he_normal(seed=0),
                               kernel_regularizer=regularizers.l2(l=0.01)
                               )(input_title_length)


    input_desc_length = Input(shape=(1, ), name='input_desc_length')
    dense_desc_length = Dense(1, 
                              activation='sigmoid', 
                              name='dense_desclength',
                              kernel_initializer=he_normal(seed=0),
                              kernel_regularizer=regularizers.l2(l=0.01)
                              #kernel_regularizer=l2(0.0001)
                              )(input_desc_length)

    #============================================= Connecting Layers =============================================#
    #Concatenating
    concat = tf.keras.layers.Concatenate()([flat_image,\

                                            flat_title,\
                                            flat_desc,\

                                            flat_reg,\
                                            flat_city,\
                                            flat_pcn,\
                                            flat_cn,\
                                            flat_param1,\
                                            flat_param2,\
                                            flat_param3,\
                                            flat_ut,\
                                            flat_week,\

                                            dense_blurness,\
                                            dense_price,\
                                            dense_item_seq_numer,\
                                            dense_image_top_1,\
                                            dense_title_length,\
                                            dense_desc_length
                                           ])

    output = Dense(128,activation='sigmoid')(concat)
    output = Dense(64,activation='sigmoid')(output)
    output = Dropout(0.2)(output)
    output = Dense(32,activation='sigmoid')(output)
    output = Dense(1, activation='sigmoid', name='output')(concat)

    model = Model(inputs=[  image_model.input,

                            input_title,\
                            input_desc,\

                            input_reg,\
                            input_city,\
                            input_pcn,\
                            input_cn,\
                            input_param1,\
                            input_param2,\
                            input_param3,\
                            input_ut,\
                            input_week,\

                            input_blurness,\
                            input_price,\
                            input_item_seq_numer,\
                            input_image_top_1,\
                            input_title_length,\
                            input_desc_length
                            ],outputs=output)
    print(model.summary())

    logs_base_dir = ".\model1\logs"
    os.makedirs(logs_base_dir, exist_ok=True)
    logdir = os.path.join(logs_base_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    print("Log directory is " , logdir)

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001,decay =0.0001)

    rmse = tf.keras.metrics.RootMeanSquaredError()

    model.compile(loss='mean_squared_error', 
                   optimizer=opt,
                   metrics=rmse)

    #model.compile(optimizer=opt, loss='categorical_crossentropy',  metrics=[auroc_sklearn])

    
    return model,tensorboard_callback



def train(filename):

    df = pd.read_csv(filename)

    Y        = df['deal_probability']
    x        = df.drop(['deal_probability','activation_date'],axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, Y, test_size=0.3, shuffle=False)

    logging.info("Encoding Started")
    embeddings_index = get_embeddings_index('cc.ru.300.vec')
    max_length = 100
    embed_dimension= 300

    tr_title_padded, tr_title_vocab_size, tr_title_embedding                        = get_embeddings_(x_train["title"].values)
    tr_description_padded, tr_description_vocab_size, tr_description_embedding      = get_embeddings_(x_train["description"].values)

    te_title_padded, te_title_vocab_size, te_title_embedding                        = get_embeddings_(x_test["title"].values)
    te_description_padded, te_description_vocab_size, te_description_embedding      = get_embeddings_(x_test["description"].values)

    tr_region, te_region, tknzr_region = tknzr_fit('region',x_train['region'].values,  x_test['region'].values)
    tr_city, te_city, tknzr_city = tknzr_fit('city', x_train['city'].values, x_test['city'].values)
    tr_par_cat, te_par_cat, tknzr_par_cat = tknzr_fit('parent_category_name', x_train['parent_category_name'].values, x_test['parent_category_name'].values)
    tr_category_name, te_category_name, tknzr_category_name = tknzr_fit('category_name', x_train['category_name'].values, x_test['category_name'].values)
    tr_param1, te_param1, tknzr_param1 = tknzr_fit('param_1', x_train['param_1'].values, x_test['param_1'].values)
    tr_param2, te_param2, tknzr_param2 = tknzr_fit('param_2', x_train['param_2'].values, x_test['param_2'].values)
    tr_param3, te_param3, tknzr_param3 = tknzr_fit('param_3', x_train['param_3'].values, x_test['param_3'].values)
    tr_usertype, te_usertype, tknzr_usertype = tknzr_fit('user_type', x_train['user_type'].values, x_test['user_type'].values)
    tr_week, te_week, tknzr_week = tknzr_fit('weekday', x_train['weekday'].map(str).values, x_test['weekday'].map(str).values)

    pickle.dump(tokenizer_container, open('lbl_encoders_avito.pkl', 'wb'))

    train_prep = [tr_title_padded,tr_description_padded,tr_region, tr_city,
              tr_par_cat, tr_category_name,tr_param1,tr_param2, 
              tr_param3,tr_usertype,tr_week,x_train["blurness"].values,
              x_train["price"].values,x_train["item_seq_number"].values,x_train["image_top_1"].values,x_train["title_length"].values,
              x_train["description_length"].values,x_train["image"].values,y_train.values]

    test_prep = [te_title_padded,te_description_padded,te_region, tr_city,
             te_par_cat,te_category_name,te_param1,te_param2,
             te_param3,te_usertype,te_week,x_test["blurness"].values,
             x_test["price"].values,x_test["item_seq_number"].values,x_test["image_top_1"].values,x_test["title_length"].values,
             x_test["description_length"].values, x_test["image"].values,y_test.values]     
   
    train_dataloader  = DataGenerator_(train_prep,batch_size=32)
    test_dataloader  =  DataGenerator_(test_prep,batch_size=32)


    model,tensorboard_callback= create_inception_model();

    from tensorflow.keras.callbacks import ModelCheckpoint
    filepath = 'modelcheckpoints'
    checkpoint = ModelCheckpoint(filepath,monitor='root_mean_squared_error',mode='min',save_best_only=True,verbose=1)
    batchsize = 5000
    stepsperepoch=x_train.shape[0] // batchsize
    validationsteps = x_test.shape[0] // batchsize

    history = model.fit(train_dataloader,
                        validation_data = test_dataloader,
                        epochs = 200 ,
                        batch_size=batchsize,
                        steps_per_epoch=stepsperepoch,
                        verbose = 1,
                        validation_steps = validationsteps,
                        callbacks = [tensorboard_callback,checkpoint])


     
    logging.info("The rmse of train is:",np.mean(history.history['root_mean_squared_error']))

    logging.info("The rmse of test is:",np.mean(history.history['val_root_mean_squared_error']))


    model.save('avito.h5')
    logging.info("Model Training Completed")


@app.route('/predict', methods=['GET', 'POST'])
def upload():

    logging.info('Request Parsed')

    lbl_encoder_container = pickle.load(open(ENCODER_PATH, 'rb'))

    model = pickle.load(open(MODEL_PATH, 'rb'))

    out = model_predict('',
                        lbl_encoder_container,
                        model,
                        totalshits=  int(request.args['totalshits']),
                        totalspageviews= int(request.args['totalspageviews']),
                        visitNumber=  int(request.args['visitNumber']) ,
                        visitStartTime=int(request.args['visitStartTime']),
                        totalsbounces=int(request.args['totalsbounces']),
                        totalsnewVisits = int(request.args['totalsnewVisits']),
                        channelGrouping=str(request.args['channelGrouping']),
                        devicebrowser=str(request.args['devicebrowser']),
                        deviceisMobile=str(request.args['deviceisMobile']),
                        devicedeviceCategory=str(request.args['devicedeviceCategory']),
                        geoNetworkcontinent=str(request.args['geoNetworkcontinent']),
                        geoNetworksubContinent=str(request.args['geoNetworksubContinent']),
                        geoNetworkcountry=str(request.args['geoNetworkcountry']),
                        geoNetworkregion=str(request.args['geoNetworkregion']),
                        geoNetworkmetro=str(request.args['geoNetworkmetro']),
                        geoNetworkcity=str(request.args['geoNetworkcity']),
                        trafficSourcecampaign=str(request.args['trafficSourcecampaign']),
                        trafficSourcesource=str(request.args['trafficSourcesource']),
                        trafficSourcemedium=str(request.args['trafficSourcemedium']),
                        trafficSourceisTrueDirect=str(request.args['trafficSourceisTrueDirect']),
                        date=str(request.args['date']))




    return str(out)


@app.route('/downloadsample/<path:filename>', methods=['GET', 'POST'])
def download_results(filename):
    return send_from_directory('uploads', filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
