#!/bin/bash


curl --location --request POST 'localhost:8080/invocations' --header 'Content-Type: application/json' --data-raw '{"bucket": "wb-inference-data","images": ["vehicle-detection/batch-transform-input/images/first-batch-transform/ksacarsharing/ksacarsharing_000001.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/ksacarsharing/ksacarsharing_000002.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/ksacarsharing/ksacarsharing_000003.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/ksacarsharing/ksacarsharing_000004.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/ksacarsharing/ksacarsharing_000005.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/ksacarsharing/ksacarsharing_000006.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/ksacarsharing/ksacarsharing_000007.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/ksacarsharing/ksacarsharing_000008.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/ksacarsharing/ksacarsharing_000009.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/ksacarsharing/ksacarsharing_000010.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/ksacarsharing/ksacarsharing_000011.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/ksacarsharing/ksacarsharing_000012.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/ksacarsharing/ksacarsharing_000013.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000001.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000002.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000003.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000004.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000005.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000006.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000007.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000008.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000009.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000010.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000011.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000012.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000013.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000014.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000015.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000016.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000017.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000018.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000019.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000020.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000021.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000022.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000023.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000024.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000025.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000026.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000027.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000028.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000029.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000030.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000031.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000032.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000033.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000034.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000035.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000036.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000037.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000038.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000039.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000040.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000041.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000042.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000043.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000044.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000045.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000046.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000047.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000048.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000049.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000050.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000051.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000052.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000053.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000054.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000055.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000056.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000057.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000058.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000059.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000060.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000061.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000062.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000063.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000064.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000065.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000066.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000067.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000068.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000069.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000070.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000071.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/emco/emco_000072.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000001.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000002.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000003.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000004.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000005.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000006.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000007.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000008.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000009.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000010.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000011.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000012.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000013.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000014.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000015.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000016.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000017.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000018.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000019.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000020.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000021.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000022.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000023.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000024.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000025.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000026.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000027.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000028.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000029.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000030.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000031.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000032.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000033.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000034.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000035.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000036.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000037.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000038.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000039.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000040.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000041.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000042.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000043.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000044.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000045.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000046.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000047.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000048.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000049.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000050.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000051.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000052.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000053.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000054.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000055.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000056.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000057.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000058.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000059.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000060.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000061.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000062.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000063.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000064.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000065.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000066.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000067.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000068.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000069.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000070.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000071.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000072.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000073.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000074.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000075.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000076.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000077.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000078.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000079.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000080.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000081.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000082.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000083.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000084.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000085.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000086.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000087.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000088.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000089.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000090.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000091.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000092.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000093.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000094.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000095.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000096.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000097.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000098.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000099.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000100.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000101.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000102.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000103.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000104.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000105.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000106.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000107.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000108.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000109.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000110.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000111.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000112.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000113.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000114.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000115.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000116.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000117.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000118.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000119.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000120.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000121.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000122.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000123.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000124.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000125.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000126.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000127.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000128.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000129.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000130.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000131.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000132.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000133.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000134.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000135.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000136.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000137.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000138.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000139.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000140.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000141.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000142.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000143.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000144.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000145.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000146.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000147.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000148.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000149.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000150.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000151.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000152.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000153.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000154.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000155.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000156.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000157.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000158.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000159.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000160.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000161.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000162.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000163.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000164.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000165.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000166.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000167.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000168.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000169.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000170.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000171.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000172.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000173.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000174.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000175.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000176.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000177.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000178.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000179.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000180.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000181.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000182.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000183.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000184.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000185.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000186.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000187.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000188.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000189.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000190.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000191.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000192.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000193.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000194.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000195.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000196.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000197.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000198.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000199.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000200.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000201.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000202.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000203.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000204.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000205.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000206.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000207.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000208.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000209.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000210.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000211.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000212.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000213.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000214.jpg","vehicle-detection/batch-transform-input/images/first-batch-transform/evo-sharing/evo-sharing_000215.jpg"]}'
