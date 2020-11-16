#!/bin/bash


curl --location --request POST 'localhost:8080/invocations' --header 'Content-Type: application/json' --data-raw '{"bucket": "wb-inference-data","images": ["vehicle-detection/ping-test-images/image_01.jpg"]}'
