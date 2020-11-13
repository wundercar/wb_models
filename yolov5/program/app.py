import json
import torch
import os

import flask
from flask import Flask, Response
from model_handler import ModelHandler

SAGEMAKER_BATCH = None
try:
    SAGEMAKER_BATCH = os.environ['SAGEMAKER_BATCH']
except KeyError:
    SAGEMAKER_BATCH = False

MAX_CONCURRENT_TRANSFORMERS = 1
BATCH_STRATEGY = 'MULTI_RECORD'
MAX_PAYLOAD_IN_MB = 6

TEST_IMG_PATH = '../images/00001.jpg'
app = Flask(__name__)
service = ModelHandler(batch_size=128, preferred_device='cuda')


def init_or_append_list(dictionary: dict, key: str, value: str):
    try:
        dictionary[key].append(value)
    except KeyError:
        dictionary[key] = [value]


def get_batch_inference_data(data: str):
    inference_data = {}
    for line in data.split('\n'):
        try:
            bucket, image = line.split(',')
            init_or_append_list(inference_data, bucket, image)
        except ValueError:
            continue

    return inference_data


@app.route('/ping', methods=['GET'])
def ping():
    """
    Determine if the container is healthy by running a sample through the algorithm.
    """
    # we will return status ok if the model can load the model and run predictions.
    gpu_availability = torch.cuda.is_available()
    device_name = 'not_available'
    if gpu_availability:
        device_name = torch.cuda.get_device_name('cuda')

    gpu_info = {
        'gpu_availability': gpu_availability,
        'gpu_device': device_name,
    }

    try:
        try:
            import boto3
            s3 = boto3.client('s3')
            s3.download_file('wb-inference-data',
                             'vehicle-detection/real-time-inference/6994861b-202a-11eb-b185-e322df8faf29.jpg',
                             '/tmp/test.jpg')
        except Exception as e:
            err = {
                'error': 'boto3',
                'message': str(e),
            }
            err = err.update(gpu_info)
            return Response(response=json.dumps(err), status=500, mimetype='application/json')

        _ = service.predict_ping(TEST_IMG_PATH)

        success_result = {'status': 'OK'}
        success_result = success_result.update(gpu_info)
        return Response(response=json.dumps(success_result), status=200, mimetype='application/json')
    except Exception as e:
        error = {
            'status': 'error',
            'message': str(e),
        }
        error = error.update(gpu_info)
        return Response(response=json.dumps(error), status=500, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def invoke():
    """
    Do an inference on a single batch of data.
    """
    data = flask.request.data.decode('utf-8')
    if SAGEMAKER_BATCH:
        inference_data = get_batch_inference_data(data)
        results, mimetype = service.predict_dictionary(dictionary=inference_data)
    else:
        data = json.loads(data)
        bucket = data['bucket']
        images = data['images']
        results, mimetype = service.predict_list(bucket=bucket, images_paths=images)

    return Response(response=results, status=200, mimetype=mimetype)


@app.route('/execution-parameters', methods=['GET'])
def execution_parameters():
    execution_params = {
        'MaxConcurrentTransforms': MAX_CONCURRENT_TRANSFORMERS,
        'BatchStrategy': BATCH_STRATEGY,
        'MaxPayloadInMB': MAX_PAYLOAD_IN_MB
    }

    return Response(response=json.dumps(execution_params), status=200, mimetype='application/json')
