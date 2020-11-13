import boto3
from PIL import Image
from typing import List
import uuid
import os
import shutil


class DataLoader:
    def __init__(self, bucket: str):
        self.bucket = bucket
        self._s3 = None

    @property
    def s3(self):
        if self._s3 is None:
            self._s3 = boto3.client('s3')

        return self._s3

    @staticmethod
    def create_tmp_dir():
        random_dir_name = 'local_image_{}'.format(uuid.uuid1())
        dir_path = os.path.join('/tmp', random_dir_name)
        os.mkdir(dir_path)

        return dir_path

    @staticmethod
    def remove_tmp_dir(tmp_dir: str):
        shutil.rmtree(tmp_dir)

    def download_images(self, images_paths):
        local_images = []
        local_dir = self.create_tmp_dir()
        for s3_img_path in images_paths:
            image_filename = os.path.basename(s3_img_path)
            local_path = os.path.join(local_dir, image_filename)
            self.s3.download_file(self.bucket, s3_img_path, local_path)
            local_images.append(local_path)

        return local_images, local_dir

    def image_generator(self, images_paths: List[str], batch_size=4):
        nbr_images = len(images_paths)
        local_paths, local_dir = self.download_images(images_paths)
        yielded_images = 0

        while yielded_images < nbr_images:
            images = []
            for _ in range(min(batch_size, nbr_images - yielded_images)):
                img = Image.open(local_paths[yielded_images])
                images.append(img)
                yielded_images += 1

            yield images

        self.remove_tmp_dir(local_dir)
