import sys
import time
import numpy as np
from multiprocessing.connection import Client
from mic_array import MicArray
print('Loading TensorFlow...')
import tensorflow as tf


print('Loading YAMNet...')
import params as yamnet_params
import yamnet as yamnet_model
params = yamnet_params.Params()
yamnet = yamnet_model.yamnet_frames_model(params)
yamnet.load_weights('yamnet.h5')
yamnet_classes = yamnet_model.class_names('yamnet_class_map.csv')



RATE = 16000
CHANNELS = 4
DOA_FRAMES = 1024    # ms
LOUDNESS_THRESHOLD = DOA_FRAMES * 16000
THRESHOLD = 0.4

def main():
    
    # Interprocessing
    address = ('localhost', 6000)
    conn = Client(address, authkey=b'thanhcolonhue')

    try:
        with MicArray(RATE, CHANNELS, RATE * DOA_FRAMES / 1000)  as mic:
            start = time.time()
            lastTimeDirection = 0

            for chunk in mic.read_chunks():
                wav_data = chunk[0::4]
                waveform = wav_data / tf.int16.max # Max Value 32768.0
                waveform = waveform.astype('float32')
                scores, embeddings, spectrogram = yamnet(waveform)
                prediction = np.mean(scores[:-1], axis=0) # last one scores comes from insufficient samples
                # assert (prediction==scores[0]).numpy().all() # only one scores at RECORD_SECONDS = 1.024
                # assert len(scores[:-1]) == CHUNK * len(CHUNKs) / RATE // 0.48 - 1 # hop 0.48 seconds
                top5 = np.argsort(prediction)[::-1][:5]
                print(time.ctime().split()[3],
                    ''.join((f" {prediction[i]:.2f} 👉{yamnet_classes[i][:7].ljust(7, '　')}" if prediction[i] >= THRESHOLD else '...') for i in top5))
            
               
                
                
                loudness = np.sum(np.abs(chunk))
                # print('loudness', loudness)
                if loudness >= LOUDNESS_THRESHOLD:
        
                    direction = int (mic.get_direction(chunk))
                    print('direction', direction)
                    
                    end = time.time()
                    elapsed_time = end - start
                    # print('elapsed_time', elapsed_time)

                    angle_difference = np.abs(direction - lastTimeDirection)
                    
                    # print('angle_difference', angle_difference)

                    if elapsed_time >= 2.1 and angle_difference > 10 :

                        conn.send(str(direction))
                        start = time.time()
                        #print("Goc xoay la: ")
                        # print('\n{}'.format(int(direction)))
                        lastTimeDirection = direction
                        
                
                        
    except KeyboardInterrupt:
        pass

    conn.close()
    
    
if __name__ == '__main__':
    main()
