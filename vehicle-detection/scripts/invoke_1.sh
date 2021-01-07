#!/bin/bash

res1=$(date +%s.%N)

curl --location --request POST 'localhost:8080/invocations' --header 'Content-Type: application/json' --data-raw '{"bucket": "wb-inference-data","images": ["vehicle-detection/ping-test-images/image_01.jpg"]}'

res2=$(date +%s.%N)

dt=$(echo "$res2 - $res1" | bc)
dd=$(echo "$dt/86400" | bc)
dt2=$(echo "$dt-86400*$dd" | bc)
dh=$(echo "$dt2/3600" | bc)
dt3=$(echo "$dt2-3600*$dh" | bc)
dm=$(echo "$dt3/60" | bc)
ds=$(echo "$dt3-60*$dm" | bc)

LC_NUMERIC=C printf "\nTotal runtime: %d:%02d:%02d:%02.4f\n" $dd $dh $dm $ds
