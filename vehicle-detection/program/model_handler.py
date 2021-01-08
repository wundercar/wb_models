import os
import json
import torch
from typing import List, Dict
from PIL import Image
from enum import Enum
from time import sleep

from utils.torch_utils import select_device
from models.experimental import attempt_load
from data_loader import DataLoader


class OutputFormat(Enum):
    JSON = 'JSON'
    TEXT = 'TEXT'


def as_list_float(values: list):
    return [float(x) for x in list(values)]


def abs_s3_images_paths(bucket: str, image_paths: List[str]):
    return [os.path.join('s3://', bucket, img_path) for img_path in image_paths]


class ModelHandler:
    def __init__(self,
                 images_size=416,
                 batch_size=64,
                 saved_model_path='../ml/model/vehicle_detection.pt',
                 preferred_device='cuda'):
        self.images_size = images_size
        self.batch_size = batch_size
        self.saved_model_path = saved_model_path
        self.preferred_device = preferred_device
        self._model = None
        self._device = None  # select_device(self.preferred_device, batch_size=self.batch_size)

    @property
    def device(self):
        if self._device is None:
            self._device = self.get_device_if_ready()

        return self._device

    @property
    def model(self):
        if self._model is None:
            self._model = attempt_load(self.saved_model_path, map_location=self.device)
            self._model = self._model.autoshape()

        return self._model

    @staticmethod
    def post_process(images_paths: List[str], predictions: list, output_format: OutputFormat):
        img_pred_dict = dict(zip(images_paths, predictions))
        if output_format == OutputFormat.JSON:
            return json.dumps(img_pred_dict), 'application/json'
        elif output_format == OutputFormat.TEXT:
            text = ''
            for image, predictions in img_pred_dict.items():
                text += '{},{}\n'.format(image, str(predictions))
            return text[:-2], 'text/plain'
        else:
            return None, None

    def get_device_if_ready(self, time_step: int = 1, timeout: int = 30):
        waiting_time = 0
        while waiting_time < timeout:
            if torch.cuda.is_available():
                break
            else:
                waiting_time += time_step
                sleep(time_step)

        return select_device(self.preferred_device, batch_size=self.batch_size)

    def predict(self, batch: list):
        """
        run the model on a data batch and return predictions.
        :param batch: List of PIL.Image objects
        :return: Tensor (predictions)
        """
        return self.model(batch, self.images_size)

    def prediction_iterator(self, dataloader: DataLoader, images_paths: List[str]):
        result = []
        for batch in dataloader.image_generator(images_paths, self.batch_size):
            output = self.predict(batch)
            result += [self.to_list_unless_none(t) for t in output]
            torch.cuda.empty_cache()

        return result

    def predict_dictionary(self, dictionary: Dict[str, List[str]]):
        results = []
        abs_image_paths = []
        for bucket, images in dictionary.items():
            abs_image_paths += abs_s3_images_paths(bucket, images)
            results += self.prediction_iterator(DataLoader(bucket), images)

        return self.post_process(abs_image_paths, results, OutputFormat.TEXT)

    def predict_list(self, bucket: str, images_paths: List[str]):
        """
        predicts a list of images stored in a single bucket
        :param bucket:
        :param images_paths:
        :return:
        """
        dataloader = DataLoader(bucket)
        result = self.prediction_iterator(dataloader, images_paths)
        abs_image_paths = abs_s3_images_paths(bucket, images_paths)

        return self.post_process(abs_image_paths, result, OutputFormat.JSON)

    def predict_ping(self, image_path: str):
        """
        this method is only used to respond to the health check send by AWS SageMaker via GET (/ping)
        :param image_path: test image, stored in the container.
        :return: predictions: list of detected vehicles in the image..
        """
        batch = [Image.open(image_path)]
        output = self.predict(batch)
        output = [self.to_list_unless_none(t) for t in output]

        return self.post_process([image_path], output, OutputFormat.JSON)

    def to_list_unless_none(self, tensor: torch.Tensor):
        if tensor is None:
            return []

        object_list = self.tensor_to_numpy(tensor)
        return [as_list_float(obj) for obj in object_list]

    def tensor_to_numpy(self, tensor: torch.Tensor):
        if self.preferred_device == 'cpu':
            return tensor.numpy()
        else:
            return tensor.cpu().numpy()

    @staticmethod
    def get_gpu_info():
        gpu_availability = torch.cuda.is_available()
        device_name = 'not_available'
        if gpu_availability:
            device_name = torch.cuda.get_device_name('cuda')

        return {
            'gpu_availability': gpu_availability,
            'gpu_device': device_name,
        }
