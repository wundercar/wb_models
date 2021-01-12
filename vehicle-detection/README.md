## Building and running the project

### Environments

I am using two separate environments:

- "dev": development env, in this env:
    - We are using `fleetbird-experimental` AWS profile.
    - The trained model weights are located in `ml/model/` folder
    - We are using a docker volume, type `bind`, `program` -> `/opt/program`
- "prod": production env, in this env:
    - We are using production AWS profile
    - The trained model weights are hosted in S3, will be imported automatically by **SageMaker**
    - The folder `program` will be copied to `/opt/program`
    
### Building the images:

- "dev":

```
docker-compose -f docker-compose.gpu.yaml -f docker-compose.gpu.dev.yaml build
```

If you already built `wunder-brain/cuda-base` you can use the following command to build only the
vehicle-detection image:
```
docker-compose -f docker-compose.gpu.yaml -f docker-compose.gpu.dev.yaml build vehicle-detection
```

- "prod":

```
docker-compose -f docker-compose.gpu.yaml build
```

For both environments, each image will be built twice one time with the tag *latest* and the other with the version tag *vx.y.z*

### Running the containers (Image with *latest* tag):

- "dev": If you already built `wunder-brain/cuda-base` & `wunder-brain/vehicle-detection-gpu:dev-latest` you can 
  start the container using this command:
  
```
docker-compose -f docker-compose.gpu.yaml -f docker-compose.gpu.dev.yaml run --service-ports vehicle-detection
```

- "prod": Production image is not supposed to work, because the image is missing the model weights
that will be later injected by *SageMaker*. Use *dev* environment to try the project.
