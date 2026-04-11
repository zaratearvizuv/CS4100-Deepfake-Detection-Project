import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import random
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, average_precision_score
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras
import pandas as pd



trainRealDir = os.path.join("datasets", "stylegan1-dataset", "real")
trainFakeDir = os.path.join("datasets", "stylegan1-dataset", "fake")

testRealDir = os.path.join("datasets", "stylegan2-dataset", "real")
testFakeDir = os.path.join("datasets", "stylegan2-dataset", "fake")

trainName = "StyleGAN1"
testName = "StyleGAN2"

maxTrain = 50000
maxVal = 10000
maxTest = 10000

batchSize = 32
targetSize = (224, 224)
epochs = 15
lr = 0.0001

# here I am creating a new experiment folder 
def experiments_folder(name):
    path = "experiments"
    os.makedirs(path, exist_ok=True)
    n = 1
    while os.path.exists(os.path.join(path, name + "-" + str(n))):
        n += 1
    folder = os.path.join(path, name + "-" + str(n))
    os.makedirs(folder)
    return folder

# here I am collecting image paths and putting them into a dataframe
def collectFiles(dir, label, maxPerClass, nested=False):
    li = []
    if nested:
        for dp, dirs, fnames in os.walk(dir):
            for f in fnames:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    li.append(os.path.join(dp, f))
    else:
        for f in os.listdir(dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                li.append(os.path.join(dir, f))
    random.shuffle(li)
    li = li[:maxPerClass]
    result = []
    for f in li:
        result.append((f, label))
    return result



def buildDf(realDir, fakeDir, maxPerClass, nested=False):
    import pandas as pd
    realData = collectFiles(realDir, "real", maxPerClass, nested)
    fakeData = collectFiles(fakeDir, "fake", maxPerClass, nested)
    allData = realData + fakeData
    random.shuffle(allData)
    return pd.DataFrame(allData, columns=["filename", "class"])

# here I am building the VGG16 architecture with BatchNorm and loading VGGFace weights
def buildVgg(inputShape, path=None):
    inp = Input(shape=inputShape)


    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(inp)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)


    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)


    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)


    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)

    mdl = Model(inp, x, name='vggface_vgg16')

    if path and os.path.exists(path):
        print("Loading pretrained weights from " + path)
        mdl.load_weights(path, by_name=True)

    return mdl

def buildClassifier(inputShape, path=None):
    vgg = buildVgg(inputShape, path)
    last = vgg.get_layer('pool5').output
    x = GlobalAveragePooling2D()(last)
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dense(1024, activation='relu', name='fc2')(x)
    x = Dense(512, activation='relu', name='fc3')(x)
    out = Dense(2, activation='softmax', name='output')(x)
    mdl = Model(vgg.input, out)
    return mdl


def plotLoss(ep, loss, valLoss):
    plt.figure(figsize=(10, 6))
    plt.plot(ep, loss, 'bo', label='Training Loss')
    plt.plot(ep, valLoss, 'orange', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plotAcc(ep, acc, valAcc):
    plt.figure(figsize=(10, 6))
    plt.plot(ep, acc, 'bo', label='Training Accuracy')
    plt.plot(ep, valAcc, 'orange', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    
    startTime = time.time()
    folder = experiments_folder("vgg-experiment")

    # here I am building the dataframes for the train and test sets
    print("Building TRAIN dataset (" + trainName + ")")
    trainDf = buildDf(trainRealDir, trainFakeDir, maxTrain)
    print("  Train samples: " + str(len(trainDf)))

    # here I am splitting off 20% of the training data to use for validation
    split = int(len(trainDf) * 0.8)
    valDf = trainDf.iloc[split:].reset_index(drop=True)
    trainDf = trainDf.iloc[:split].reset_index(drop=True)
    print("  After split, Train: " + str(len(trainDf)) + ", Val: " + str(len(valDf)))

    print("\nBuilding TEST dataset (" + testName + ")")
    testDf = buildDf(testRealDir, testFakeDir, maxTest)
    print("  Test samples: " + str(len(testDf)))

    # here I am setting up image generators to rescale and batch the images
    imgGen = ImageDataGenerator(rescale=1./255.)

    trainGen = imgGen.flow_from_dataframe(dataframe=trainDf, x_col="filename", y_col="class",
        target_size=targetSize, batch_size=batchSize, class_mode="categorical")
    
    valGen = imgGen.flow_from_dataframe(dataframe=valDf, x_col="filename", y_col="class",
        target_size=targetSize, batch_size=batchSize, class_mode="categorical")
    
    testGen = imgGen.flow_from_dataframe(dataframe=testDf, x_col="filename", y_col="class",
        target_size=targetSize, batch_size=1, shuffle=False, class_mode="categorical")

 
    weightsPath = "weights/vgg16_weights.h5"
    model = buildClassifier((224, 224, 3), path=weightsPath)

    model.compile(loss=keras.losses.categorical_crossentropy, metrics=['acc'],
        optimizer=tf.keras.optimizers.Adam(lr))
    model.summary()

    # here I am setting up a callback to reduce lr on plateau
    reduceLr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2,
        verbose=1, mode='max', min_lr=0.00001)

    # here I am training the model
    trainSteps = len(trainDf) // batchSize
    valSteps = len(valDf) // batchSize

    print("\nTraining VGG on " + trainName)
    hist = model.fit(trainGen,steps_per_epoch=trainSteps, validation_data=valGen,
        validation_steps=valSteps,
        epochs=epochs,
        verbose=1,
        callbacks=[reduceLr])

    # here I am plotting the training curves
    epRange = range(1, len(hist.history['loss']) + 1)
    plotLoss(epRange, hist.history['loss'], hist.history['val_loss'])
    plotAcc(epRange, hist.history['acc'], hist.history['val_acc'])

    # saving the accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(epRange, hist.history['acc'], 'bo', label='Training Accuracy')
    plt.plot(epRange, hist.history['val_acc'], 'orange', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(folder, "accuracy.png"))
    plt.close()

    print("\nEvaluating on TRAIN data (" + trainName + ")")
    trainEvalGen = imgGen.flow_from_dataframe(dataframe=trainDf, x_col="filename", y_col="class",
        target_size=targetSize, batch_size=1, shuffle=False, class_mode="categorical")
    pred1 = model.predict(trainEvalGen)
    true1 = trainEvalGen.classes
    class1 = np.argmax(pred1, axis=1)
    trainAcc = accuracy_score(true1, class1)

    print("\nEvaluating on TEST data (" + testName + ")")
    pred2 = model.predict(testGen)
    true2 = testGen.classes
    class2 = np.argmax(pred2, axis=1)
    testAcc = accuracy_score(true2, class2)

    roc = roc_auc_score(true2, class2)
    ap = average_precision_score(true2, class2)

 
    print("\nCross-GAN Results: " + trainName + " -> " + testName)
    print("Train Accuracy (" + trainName + "): " + str(round(trainAcc * 100, 2)) + "%")
    print("Test Accuracy (" + testName + "): " + str(round(testAcc * 100, 2)) + "%")
    print("Difference: " + str(round(abs(trainAcc - testAcc) * 100, 2)) + "%")
    print("ROC-AUC Score: " + str(round(roc, 5)))
    print("AP Score: " + str(round(ap, 5)))
    print()
    print(classification_report(true2, class2, target_names=['Real', 'Fake']))

    cm = confusion_matrix(true2, class2)
    tn, fp, fn, tp = cm.ravel()
    print("Confusion Matrix:")
    print("                  Predicted Real  Predicted Fake")
    print("  Actual Real     " + str(tn).ljust(15) + str(fp))
    print("  Actual Fake     " + str(fn).ljust(15) + str(tp))

    totalTime = time.time() - startTime

    with open(os.path.join(folder, "results.txt"), "w") as f:
        f.write("Experiment: VGG16 + BatchNorm\n")
        f.write("Train Dataset: " + trainName + "\n")
        f.write("Test Dataset: " + testName + "\n")
        f.write("Train images: " + str(len(trainDf)) + "\n")
        f.write("Val images: " + str(len(valDf)) + "\n")
        f.write("Test images: " + str(len(testDf)) + "\n\n")
        f.write("Train Accuracy: " + str(round(trainAcc * 100, 2)) + "%\n")
        f.write("Test Accuracy: " + str(round(testAcc * 100, 2)) + "%\n")
        f.write("Difference: " + str(round(abs(trainAcc - testAcc) * 100, 2)) + "%\n")
        f.write("ROC-AUC Score: " + str(round(roc, 5)) + "\n")
        f.write("AP Score: " + str(round(ap, 5)) + "\n\n")
        f.write("Epochs: " + str(epochs) + "\n")
        f.write("Batch Size: " + str(batchSize) + "\n")
        f.write("Learning Rate: " + str(lr) + "\n")
        f.write("Optimizer: Adam\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(true2, class2, target_names=['Real', 'Fake']))
        f.write("\nConfusion Matrix:\n")
        f.write("                  Predicted Real  Predicted Fake\n")
        f.write("  Actual Real     " + str(tn).ljust(15) + str(fp) + "\n")
        f.write("  Actual Fake     " + str(fn).ljust(15) + str(tp) + "\n")
        f.write("\nTrue Positives  (Fake correctly detected): " + str(tp) + "\n")
        f.write("True Negatives  (Real correctly detected): " + str(tn) + "\n")
        f.write("False Positives (Real misclassified as Fake): " + str(fp) + "\n")
        f.write("False Negatives (Fake misclassified as Real): " + str(fn) + "\n")
        f.write("\nTotal Time: " + str(round(totalTime/60, 2)) + " minutes\n")

    print("\nResults saved to " + folder)
    print("Total time: " + str(round(totalTime/60, 2)) + " minutes")