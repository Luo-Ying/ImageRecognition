import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def display_report(dataset, model, typeModel, typeData):
    print("Classification report ", typeData, ":")
    label_true = []
    x_true = []
    for a in dataset:
        x_true.extend(a[0].numpy().tolist())
        label_true.extend(a[1].numpy().tolist())

    y_pred = model.predict(x_true, batch_size=64, verbose=1)
    y_pred = np.argmax(y_pred, axis=1)
    # print(y_pred)
    # print(len(y_pred))
    # print(label_true)
    # print(len(label_true))

    cr = classification_report(label_true, y_pred, digits=4)
    print(cr)
    
    display_confusing_matrix(label_true, y_pred, typeModel, typeData)
    
def display_confusing_matrix(label_true, y_pred, typeModel, typeData):
    cm = confusion_matrix(label_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig("./confusion_matrix/confusion_matrix_" + str(typeModel) + "_" + str(typeData) + ".png")