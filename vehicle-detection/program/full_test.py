from utils.torch_utils import select_device
from models.experimental import attempt_load
import torch


cuda_a = torch.cuda.is_available()
print(cuda_a)
if cuda_a:
    print('device: {}'.format(torch.cuda.get_device_name()))
else:
    print('no device found')

device = select_device('0', batch_size=128)
print(device)

model = attempt_load('/opt/ml/model/vehicle_detection.pt', map_location=device)
print(model)
print('is model on gpu?', next(model.parameters()).is_cuda)
