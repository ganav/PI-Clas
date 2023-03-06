import Network, math
import  Utils,random
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input,Lambda
from tensorflow.keras.layers import TimeDistributed
from tqdm import tqdm
import numpy as np
import argparse,csv, time,cv2
from tensorflow.keras.optimizers import Adam,SGD
import metrics
from layer_utils import ReflectionPadding2D, res_block

def main():

    np.random.seed(10)

    ext= '.png'
    n_class = 28

    data_csv = []
    x_test,label = Utils.load_training_data('G:\\projects\\paper 18\\source\\data\\visible dataset divided\\db2\\class_blur\\','', ext,n_class)
                
    batch_count2 = len(x_test)

    generator = load_model('G:\\projects\\paper 18\\source\\deblur\\model db1\\model200.h5', custom_objects={'ReflectionPadding2D': ReflectionPadding2D})
    cnn = load_model('./model blur db1/gen_model100.h5')#model - Copy
    #print(generator.summary())
                                   
    for num in range(batch_count2):
        #st = time.time()
        res = cnn.predict(generator.predict(x_test[num]))
        #en = time.time()
        #print(en-st)
        r=np.argmax(res)
        r2=np.argmax(label[num])

        Utils.app_('./acc',[r,r2])

main()
metrics.main()