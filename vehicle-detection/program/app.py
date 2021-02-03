import json
import os

import torch
import flask
from flask import Flask, Response

from model_handler import ModelHandler

SAGEMAKER_BATCH = os.environ.get('SAGEMAKER_BATCH', False)
TEST_S3_BUCKET = os.environ.get('TEST_S3_BUCKET')
TEST_S3_IMG_PATH = os.environ.get('TEST_S3_IMG_PATH')
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 64))
DEBUG_MODE = os.environ.get('DEBUG_MODE') == 'True'
TEST_LOCAL_IMG_PATH = '/tmp/test.jpg'

# todo: make int values with int ...
MAX_CONCURRENT_TRANSFORMERS = int(os.environ.get('MAX_CONCURRENT_TRANSFORMERS', 1))
BATCH_STRATEGY = os.environ.get('BATCH_STRATEGY', 'MULTI_RECORD')
MAX_PAYLOAD_IN_MB = int(os.environ.get('MAX_PAYLOAD_IN_MB', 6))

app = Flask(__name__)
service = ModelHandler(batch_size=BATCH_SIZE, preferred_device='cuda')


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


def get_env_vars():
    return {
        'SAGEMAKER_BATCH': SAGEMAKER_BATCH,
        'TEST_S3_BUCKET': TEST_S3_BUCKET,
        'TEST_S3_IMG_PATH': TEST_S3_IMG_PATH,
        'DEBUG_MODE': DEBUG_MODE,
        'BATCH_SIZE': BATCH_SIZE,
        'MAX_CONCURRENT_TRANSFORMERS': MAX_CONCURRENT_TRANSFORMERS,
        'BATCH_STRATEGY': BATCH_STRATEGY,
        'MAX_PAYLOAD_IN_MB': MAX_PAYLOAD_IN_MB,
    }


def get_gpu_info():
    gpu_availability = torch.cuda.is_available()
    device_name = 'not_available'
    cuda_version = 'not_available'
    arch_list = []
    if gpu_availability:
        device_name = torch.cuda.get_device_name('cuda')
        arch_list = torch.cuda.get_arch_list()
        cuda_version = torch.version.cuda

    return {
        'gpu_availability': gpu_availability,
        'gpu_device': device_name,
        'arch_list': arch_list,
        'cuda_version': cuda_version,
    }


@app.route('/ping', methods=['GET'])
def ping():
    """
    Determine if the container is healthy by running a sample through the algorithm.
    we will return status ok if sage maker have access to S3, can load the model and run predictions.
    """
    gpu_info = get_gpu_info()
    print(json.dumps(gpu_info), flush=DEBUG_MODE)
    print(json.dumps(get_env_vars()), flush=DEBUG_MODE)

    try:
        import boto3
        s3 = boto3.client('s3')
        s3.download_file(TEST_S3_BUCKET, TEST_S3_IMG_PATH, TEST_LOCAL_IMG_PATH)
        print('finish downloading file', flush=DEBUG_MODE)
    except Exception as e:
        err = {'error': 'boto3', 'message': str(e)}
        err.update(gpu_info)
        print(json.dumps(err), flush=DEBUG_MODE)
        return Response(response=json.dumps(err), status=500, mimetype='application/json')

    try:
        print('predicting ping test image...', flush=DEBUG_MODE)
        _ = service.predict_ping(TEST_LOCAL_IMG_PATH)
        success_result = {'status': 'OK'}
        success_result.update(gpu_info)
        print('finish inference on test file', flush=DEBUG_MODE)
        return Response(response=json.dumps(success_result), status=200, mimetype='application/json')
    except Exception as e:
        error = {
            'status': 'error',
            'message': str(e),
        }
        error.update(gpu_info)
        print(json.dumps(error), flush=DEBUG_MODE)
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
