import flask
import json
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

from flask import Response, request
from Algorithm.BCVAR import train_model, make_prediction


application = flask.Flask(__name__)

trained = False


@application.route("/trainModel", methods=["POST"])
def train():
    if request.json is None:
        # Expect application/json request
        response = Response("Empty request", status=415)
    else:
        try:
            request_content = json.loads(request.data)
            message = request_content

            print("Training - JSON content ", message)

            currency_o = message['currency_o']
            currency_d = message['currency_d']

            fecha_ini = message['date_ini']
            fecha_fin = message['date_fin']

            Y_predicted, Y_real, dates, labels = train_model(currency_o, currency_d, fecha_ini, fecha_fin)

            global trained
            trained = True

            #print('Y predicted', Y_predicted)
            #print('Y real', Y_real)

            dates = dates.values.ravel().astype('datetime64[D]').tolist()
            #print('Dates', dates)

            service_response = {'Predicted_values': Y_predicted.tolist(), 'Real_values': Y_real.tolist(), 'Dates': dates}

            response = Response(json.dumps(service_response, default=str).encode('UTF-8'), 200)

        except Exception as ex:
            print(ex)
            response = Response("Error processing", 500)

    return response


def addDAys(dates):
    for i in range(len(dates)):
        new_date = pd.to_datetime(dates[i]) + pd.DateOffset(days=i + 1)
        dates[i] = new_date

    return dates


@application.route("/predict", methods=["POST"])
def predict():
    global trained
    if request.json is None:
        # Expect application/json request
        response = Response("Empty request", status=415)
    else:
        try:
            if trained:
                request_content = json.loads(request.data)
                message = request_content
                days_to_predict = message["days_to_predict"]
                print("Predict - JSON content ", message)

                y_pred, _dates = make_prediction(prediction_days=days_to_predict)
                _dates = addDAys(_dates.values.ravel()).astype('datetime64[D]')
                print("Prediction: ", y_pred)
                print("Dates: ", _dates)
                service_response = {'Predicted_values': y_pred.tolist(), 'Dates': _dates.tolist()}
                response = Response(json.dumps(service_response, default=str).encode('UTF-8'), 200)
            else:
                response = Response("Call the training model method first", 405)
        except Exception as ex:
            print(ex)
            response = Response("Error processing", 500)

    return response

if __name__ == "__main__":
    application.run(host="0.0.0.0", threaded=True)