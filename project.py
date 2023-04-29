import cv2
import os
import pickle as pkl
from datetime import datetime
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, RandomContrast, RandomBrightness, RandomZoom, RandomRotation, RandomFlip
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import itertools

cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

class FaceRecognization:
    def __init__(self):
        self.per_name = []
    
    def capture(self, name):
        video_capture = cv2.VideoCapture(0)
        # dirname = 'test'
        size =(100, 100)
        par_dir = "image_data"
        save_path = os.path.join(par_dir, name)
        os.makedirs(save_path)
        t1=datetime.now()
        t2=datetime.now()
        i=0
        while True:
            t2=datetime.now()
            print((t2-t1).total_seconds())
            # Capture frame-by-frame
            ret, frames = video_capture.read()
            gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            print(i)
            # Draw a rectangle around the faces
            if((t2-t1).total_seconds()>0.5):
                i=i+1
                for (x, y, w, h) in faces:
                    res = cv2.resize(gray[y:y+h,x:x+w], size, interpolation = cv2.INTER_AREA)
                    cv2.imwrite(os.path.join(save_path,'face'+str(i)+'.jpg'), res)
                    cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 255, 0), 2)
                t1=t2

            # Display the resulting frame
            cv2.imshow('Video', frames)
            if(i==50):
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

    def data_capture(self):
        name=input("Enter the student name: ")
        self.capture(name)
        # self.per_name.append(name)
        # print(self.per_name)

    def data_conversion(self):
        dim=(100,100)
        df = pd.DataFrame(columns = [f'pix-{i}' for i in range(1, 1+(dim[0]*dim[1]))]+['class'])
        cls = 0
        filename = 'dataset.csv'
        j=1
        directories = os.listdir("image_data")
        self.per_name = []
        for i in range(len(directories)):
            self.per_name.append(directories[i])
            path=os.path.join("image_data", directories[i])

            per_imgs = os.listdir(path)
            for k in range(1, 1+len(per_imgs)):
                img =Image.open(path+'/'+per_imgs[k-1])
                df.loc[j] = list(img.getdata()) + [cls]
                j=j+1
            cls=cls+1
        df.to_csv(filename,index = False)
        print(self.per_name)

    def fit(self):
        self.data_conversion()
        df = pd.read_csv('dataset.csv', index_col=0)
        X = df.iloc[:, :100*100].values.reshape(-1, 100, 100, 1) 
        y = df.iloc[:, -1].values
        X.shape, y.shape

        X_train,X_test,y_train,y_test=train_test_split(X, y, random_state=42, test_size=0.15)

        print(f'Train Size - {X_train.shape}\nTest Size - {X_test.shape}')
        print(f'Train Size - {y_train.shape}\nTest Size - {y_test.shape}')

        im_shape=(100,100,1)
        data_augmentation = Sequential(
        [
            # RandomFlip("horizontal", input_shape=im_shape),
            RandomRotation(0.1),
            # RandomZoom(0.1),
            RandomContrast(factor=0.1),
            RandomBrightness(factor=0.1),
        ])

        model= Sequential([
            data_augmentation,
            Conv2D(filters=36, kernel_size=7, activation='relu', input_shape= im_shape),
            MaxPooling2D(pool_size=2),
            Conv2D(filters=54, kernel_size=5, activation='relu', input_shape= im_shape),
            MaxPooling2D(pool_size=2),
            Flatten(),
            Dense(2024, activation='relu'),
            Dropout(0.5),
            Dense(1024, activation='relu'),
            Dropout(0.5),
            Dense(512, activation='relu'),
            Dropout(0.5),
            #20 is the number of outputs
            Dense(len(self.per_name), activation='softmax')  
        ])



        model.compile(
            loss='sparse_categorical_crossentropy',#'categorical_crossentropy',
            optimizer=Adam(learning_rate=0.00008),
            metrics=['accuracy']
        )


        history=model.fit(
            np.array(X_train), np.array(y_train), batch_size=5,
            epochs=15, verbose=2,
            validation_data=(np.array(X_test),np.array(y_test)),
        )
        print(history.history.keys())
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        with open('model.pkl', 'wb') as file:
            pkl.dump(model, file)

        predicted=model.predict(X_test)
        # print(predicted) 
        ynew=np.argmax(predicted,axis=1)
        # ynew = model.predict(X_test)
        cnf_matrix=confusion_matrix(np.array(y_test),ynew)
        # plot_confusion_matrix(cnf_matrix,)
        list_of_numbers = [number for number in range(0, len(self.per_name))]
        plot_confusion_matrix(cnf_matrix[0:len(self.per_name),0:len(self.per_name)], classes=list_of_numbers,title='Confusion matrix, without normalization')

        print(classification_report(np.array(y_test),ynew))
        scor = model.evaluate( np.array(X_test),  np.array(y_test), verbose=0)

        print('test los {:.4f}'.format(scor[0]))
        print('test acc {:.4f}'.format(scor[1]))

    def predict(self):

        video_capture = cv2.VideoCapture(0)
        size =(100, 100)
        par_dir = "test_images"
        t1=datetime.now()
        t2=datetime.now()
        i=0
        while True:
            t2=datetime.now()
            # print((t2-t1).total_seconds())
            # Capture frame-by-frame
            ret, frames = video_capture.read()
            gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            # print(i)
            # Draw a rectangle around the faces
            if((t2-t1).total_seconds()>2.0):
                # print(" ######################## faces : ", faces)
                # if(faces == ()):
                #     continue
                i=i+1

                for (x, y, w, h) in faces:
                    res = cv2.resize(gray[y:y+h,x:x+w], size, interpolation = cv2.INTER_AREA)
                    # path = par_dir+"/"+name
                    cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.imwrite(os.path.join(par_dir,'test.jpg'), res)
                    # print("res shape  :", res.shape)
                t1=t2

                if(not os.path.exists("test_images/test.jpg")):
                    print("image is not captured")
                    i -= 1
                    continue
                else:
                    print("Image got Captured")

            if(i==1):
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        img =Image.open("test_images/test.jpg")
        x = np.array(img.getdata()).reshape((-1, 100, 100, 1))
        print("shape of x: ", x.shape)

        model = None
        with open('model.pkl', 'rb') as f:
            model = pkl.load(f)

        os.remove("test_images/test.jpg")
        
        predict_x=model.predict(x) 
        print("predict_x :", predict_x[0])
        
        classes_x=int(np.argmax(predict_x,axis=1))
        print("class_x : ", classes_x)
        count = 0
        for i in range(len(predict_x[0])):
            if predict_x[0][i] < 0.5:
                count += 1
        if(count == len(predict_x[0])):
            print("Failed recognizing your face")
            classes_x = -1
                 
        video_capture.release()
        cv2.destroyAllWindows()
        return classes_x

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


if __name__ == "__main__":
    FR = FaceRecognization()
    count = 0
    while True:      
        FR.data_conversion()
        cmd = int(input("Enter 1 to store data, 2 to train the data, 3 to mark attendance and 0 to quit."))
        if(cmd == 1):
            FR.data_capture()
            # print(FR.per_name)
        if(cmd == 2):
            FR.fit()
            if count == 0:
                f = open("attendance.txt", 'w')
                f.write("")
                f.close()
                file = open("attendance.txt", 'a')
                for i in range(len(FR.per_name)):
                    s = FR.per_name[i] + "-0-None" + "\n"
                    file.write(s)
                count = 1
                file.close()
        elif(cmd == 3):
            id = FR.predict()
            identity = 'N'
            if(id >= 0):
                print("Are You", FR.per_name[id], "?")
                identity = input("If Yes press Y, else press N :")
            while(identity == "N" and id >= 0):
                print("#################### PLEASE STAND STILL AND WATCH THE CAMERA #######################")
                id = FR.predict()
                print("Are You", FR.per_name[id], "?")
                identity = input("If Yes press Y, else press N :")
            if(identity == 'Y' and id >= 0):
                s = ""
                file = open("attendance.txt", 'r')
                for line in file:
                    line1 = line
                    line = line.strip()
                    l = line.split('-')
                    if(l[0] == FR.per_name[id]):
                        l[1] = '-1-'
                        l[2] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        s1 = l[0] + l[1] + l[2] + "\n"
                        s += s1
                    else:
                        s += line1
                file.close()
                write_file = open("attendance.txt", 'w')
                write_file.write(s)
                write_file.close()
        elif(cmd == 0):
            print("Exiting!")
            break
        else:
            print("Invalid Command")
