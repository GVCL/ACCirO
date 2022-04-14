
import numpy as np
from PIL import Image
from keras.models import model_from_json


def classifyImage(path):
    # load json and create model
    json_file = open('Classifier_Models/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("Classifier_Models/model.h5")

    json_file2 = open('Classifier_Models/model_scatt1.json', 'r')
    loaded_model_json2 = json_file2.read()
    json_file2.close()
    loaded_model2 = model_from_json(loaded_model_json2)
    # load weights into new model
    loaded_model2.load_weights("Classifier_Models/model_scatt1.h5")

    image = Image.open(path)
    rgb_im = image.convert('RGB')
    image = rgb_im.resize((200, 200), Image.ANTIALIAS)
    image = np.asarray(image)
    image = image[...,:3].reshape(1, 200, 200, 3)
    image = image.astype('float32')
    image /= 255.0

    # pred = loaded_model.predict_classes(image)
    pred = np.argmax(loaded_model.predict(image),axis=1)
    if pred == 3:
        chart_type = "pie"
    else:
        # pred2 = loaded_model2.predict_classes(image)
        pred2 = np.argmax(loaded_model2.predict(image),axis=1)
        if pred2 == 0:
            chart_type="bar"
        elif pred2 == 1:
            chart_type = "scatter"
        elif pred2 == 2:
            chart_type = "line"
        elif pred2 == 3:
            chart_type = "pie"
        else:
            chart_type = "other"

    return chart_type

