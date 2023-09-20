#Results
## Determine max Filesize for classification

### Trained model after Epoch 10

- Loss=0.0701
- Accuracy=0.9513

### Filesize 40

#### Quality:
-  JPEG_XL =
-  JPEG_XR =
-  WebP = 90
-  AVIF =
-  BPG = 40
-  HEIC = 70


#### Score:
- Accuracy = 0.8950
- Precision = 0.8999
- Recall = 0.8950
- F1-Score = 0.8963

### Filesize 50

#### Quality:

-  JPEG_XL = 80
-  JPEG_XR = 0.4
-  WebP = 90
-  AVIF = 60
-  BPG = 40
-  HEIC = 65

#### Score:

- Accuracy: 0.8150
- Precision: 0.8359
- Recall: 0.8150
- F1-score: 0.8183

### Filesize 60

#### Quality:
-  JPEG_XL = 90
-  JPEG_XR = 0.6
-  WebP = 90
-  AVIF = 85
-  BPG = 25
-  HEIC = 55

#### Score:
- Accuracy: 0.7267
- Precision: 0.7694
- Recall: 0.7267
- F1-score: 0.7291

### Filesize 75

#### Quality:
-  JPEG_XL = 90
-  JPEG_XR = 0.8
-  WebP = 95
-  AVIF = 80
- BPG = 30
-  HEIC = 55

#### Score:

- Accuracy: 0.5900
- Precision: 0.6592
- Recall: 0.5900
- F1-score: 0.5866

### Filesize 100

#### Quality:
-  JPEG_XL = 98
-  JPEG_XR = 0.9
-  WebP = 98
-  AVIF = 95
- BPG = 15
-  HEIC = 70

#### Score:

- Accuracy: 0.4283
- Precision: 0.4842
- Recall: 0.4283
- F1-score: 0.3858

## Train CNN for each codec and compare with combinated one

In the results below we see that the models for each filesize are bad classifiers and our model with trained mix filesizes is much better.

-------------------------------------------------------------------------------------
Train model with filesize = 40 kB
Train Epoch: 1, Loss: 1.8104
Validation accuracy: 0.1867 in epoch: 1
Train Epoch: 2, Loss: 1.7635
Validation accuracy: 0.2933 in epoch: 2
Train Epoch: 3, Loss: 1.5952
Validation accuracy: 0.3367 in epoch: 3
Train Epoch: 4, Loss: 1.4350
Validation accuracy: 0.4300 in epoch: 4
Train Epoch: 5, Loss: 1.2939
Validation accuracy: 0.5150 in epoch: 5
Train Epoch: 6, Loss: 1.1256
Validation accuracy: 0.4400 in epoch: 6
Train Epoch: 7, Loss: 1.0417
Validation accuracy: 0.5333 in epoch: 7
Train Epoch: 8, Loss: 0.9537
Validation accuracy: 0.5883 in epoch: 8
Train Epoch: 9, Loss: 0.8258
Validation accuracy: 0.6017 in epoch: 9
Train Epoch: 10, Loss: 0.8316
Validation accuracy: 0.6217 in epoch: 10

Accuracy: 0.6217
Precision: 0.6525
Recall: 0.6217
F1-score: 0.6131
Model saved
-------------------------------------------------------------------------------------
Train model with filesize = 50 kB
Train Epoch: 1, Loss: 1.8196
Validation accuracy: 0.1700 in epoch: 1
Train Epoch: 2, Loss: 1.7839
Validation accuracy: 0.2333 in epoch: 2
Train Epoch: 3, Loss: 1.7489
Validation accuracy: 0.3067 in epoch: 3
Train Epoch: 4, Loss: 1.6849
Validation accuracy: 0.2883 in epoch: 4
Train Epoch: 5, Loss: 1.5974
Validation accuracy: 0.3767 in epoch: 5
Train Epoch: 6, Loss: 1.4242
Validation accuracy: 0.3150 in epoch: 6
Train Epoch: 7, Loss: 1.2955
Validation accuracy: 0.4683 in epoch: 7
Train Epoch: 8, Loss: 1.2098
Validation accuracy: 0.4467 in epoch: 8
Train Epoch: 9, Loss: 1.0890
Validation accuracy: 0.5817 in epoch: 9
Train Epoch: 10, Loss: 1.0030
Validation accuracy: 0.5900 in epoch: 10

Accuracy: 0.5900
Precision: 0.6215
Recall: 0.5900
F1-score: 0.5884
Model saved
-------------------------------------------------------------------------------------
Train model with filesize = 60 kB
Train Epoch: 1, Loss: 1.8216
Validation accuracy: 0.1700 in epoch: 1
Train Epoch: 2, Loss: 1.7959
Validation accuracy: 0.1667 in epoch: 2
Train Epoch: 3, Loss: 1.7954
Validation accuracy: 0.1867 in epoch: 3
Train Epoch: 4, Loss: 1.7932
Validation accuracy: 0.1900 in epoch: 4
Train Epoch: 5, Loss: 1.7920
Validation accuracy: 0.1667 in epoch: 5
Train Epoch: 6, Loss: 1.7856
Validation accuracy: 0.2117 in epoch: 6
Train Epoch: 7, Loss: 1.7717
Validation accuracy: 0.2367 in epoch: 7
Train Epoch: 8, Loss: 1.7547
Validation accuracy: 0.2433 in epoch: 8
Train Epoch: 9, Loss: 1.7137
Validation accuracy: 0.2783 in epoch: 9
Train Epoch: 10, Loss: 1.6946
Validation accuracy: 0.1717 in epoch: 10

Accuracy: 0.1717
Precision: 0.1170
Recall: 0.1717
F1-score: 0.1320
Model saved
-------------------------------------------------------------------------------------
Train model with filesize = 75 kB
Train Epoch: 1, Loss: 1.8220
Validation accuracy: 0.1650 in epoch: 1
Train Epoch: 2, Loss: 1.7962
Validation accuracy: 0.1667 in epoch: 2
Train Epoch: 3, Loss: 1.7942
Validation accuracy: 0.1817 in epoch: 3
Train Epoch: 4, Loss: 1.7940
Validation accuracy: 0.1683 in epoch: 4
Train Epoch: 5, Loss: 1.7934
Validation accuracy: 0.1750 in epoch: 5
Train Epoch: 6, Loss: 1.7931
Validation accuracy: 0.1717 in epoch: 6
Train Epoch: 7, Loss: 1.7923
Validation accuracy: 0.1750 in epoch: 7
Train Epoch: 8, Loss: 1.7913
Validation accuracy: 0.2000 in epoch: 8
Train Epoch: 9, Loss: 1.7912
Validation accuracy: 0.1950 in epoch: 9
Train Epoch: 10, Loss: 1.7858
Validation accuracy: 0.1933 in epoch: 10

Accuracy: 0.1933
Precision: 0.2316
Recall: 0.1933
F1-score: 0.1435
Model saved
-------------------------------------------------------------------------------------
Train model with filesize = 100 kB
Train Epoch: 1, Loss: 1.8178
Validation accuracy: 0.1733 in epoch: 1
Train Epoch: 2, Loss: 1.7958
Validation accuracy: 0.1633 in epoch: 2
Train Epoch: 3, Loss: 1.7948
Validation accuracy: 0.1750 in epoch: 3
Train Epoch: 4, Loss: 1.7933
Validation accuracy: 0.1667 in epoch: 4
Train Epoch: 5, Loss: 1.7936
Validation accuracy: 0.1717 in epoch: 5
Train Epoch: 6, Loss: 1.7921
Validation accuracy: 0.1700 in epoch: 6
Train Epoch: 7, Loss: 1.7922
Validation accuracy: 0.1833 in epoch: 7
Train Epoch: 8, Loss: 1.7899
Validation accuracy: 0.1850 in epoch: 8
Train Epoch: 9, Loss: 1.7908
Validation accuracy: 0.1950 in epoch: 9
Train Epoch: 10, Loss: 1.7801
Validation accuracy: 0.2100 in epoch: 10

Accuracy: 0.2100
Precision: 0.2130
Recall: 0.2100
F1-score: 0.1393
Model saved




## Train CNN for each filesize and compare with mixed model. (Same number of images for filesize model and mixed model)

Evaluate pretrained model ( cnnParams_resnet18bigModel_fs_5.pt ) with Filesize = 5 kB
Accuracy: 0.9692
Precision: 0.9691
Recall: 0.9692
F1-score: 0.9689
Evaluate pretrained model ( cnnParams_resnet18bigModel_fs_5.pt ) with Filesize = 10 kB
Accuracy: 0.9294
Precision: 0.9311
Recall: 0.9294
F1-score: 0.9289
Evaluate pretrained model ( cnnParams_resnet18bigModel_fs_5.pt ) with Filesize = 17 kB
Accuracy: 0.7662
Precision: 0.7902
Recall: 0.7662
F1-score: 0.7626
Evaluate pretrained model ( cnnParams_resnet18bigModel_fs_5.pt ) with Filesize = 25 kB
Accuracy: 0.5732
Precision: 0.6369
Recall: 0.5732
F1-score: 0.5629
Evaluate pretrained model ( cnnParams_resnet18bigModel_fs_5.pt ) with Filesize = 32 kB
Accuracy: 0.4160
Precision: 0.5125
Recall: 0.4160
F1-score: 0.4011
Evaluate pretrained model ( cnnParams_resnet18bigModel_fs_5.pt ) with Filesize = 40 kB
Accuracy: 0.3156
Precision: 0.4403
Recall: 0.3156
F1-score: 0.2941
Evaluate pretrained model ( cnnParams_resnet18bigModel_fs_10.pt ) with Filesize = 5 kB
Accuracy: 0.9630
Precision: 0.9656
Recall: 0.9630
F1-score: 0.9635
Evaluate pretrained model ( cnnParams_resnet18bigModel_fs_10.pt ) with Filesize = 10 kB
Accuracy: 0.9716
Precision: 0.9720
Recall: 0.9716
F1-score: 0.9712
Evaluate pretrained model ( cnnParams_resnet18bigModel_fs_10.pt ) with Filesize = 17 kB
Accuracy: 0.9424
Precision: 0.9461
Recall: 0.9424
F1-score: 0.9416
Evaluate pretrained model ( cnnParams_resnet18bigModel_fs_10.pt ) with Filesize = 25 kB
Accuracy: 0.8380
Precision: 0.8610
Recall: 0.8380
F1-score: 0.8345
Evaluate pretrained model ( cnnParams_resnet18bigModel_fs_10.pt ) with Filesize = 32 kB
Accuracy: 0.7184
Precision: 0.7819
Recall: 0.7184
F1-score: 0.7146
Evaluate pretrained model ( cnnParams_resnet18bigModel_fs_10.pt ) with Filesize = 40 kB
Accuracy: 0.5834
Precision: 0.7007
Recall: 0.5834
F1-score: 0.5738
Evaluate pretrained model ( cnnParams_resnet18bigModel_fs_17.pt ) with Filesize = 5 kB
Accuracy: 0.9040
Precision: 0.9287
Recall: 0.9040
F1-score: 0.9047
Evaluate pretrained model ( cnnParams_resnet18bigModel_fs_17.pt ) with Filesize = 10 kB
Accuracy: 0.9652
Precision: 0.9653
Recall: 0.9652
F1-score: 0.9651
Evaluate pretrained model ( cnnParams_resnet18bigModel_fs_17.pt ) with Filesize = 17 kB
Accuracy: 0.9586
Precision: 0.9587
Recall: 0.9586
F1-score: 0.9584
Evaluate pretrained model ( cnnParams_resnet18bigModel_fs_17.pt ) with Filesize = 25 kB
Accuracy: 0.9270
Precision: 0.9297
Recall: 0.9270
F1-score: 0.9267
Evaluate pretrained model ( cnnParams_resnet18bigModel_fs_17.pt ) with Filesize = 32 kB
Accuracy: 0.8934
Precision: 0.9005
Recall: 0.8934
F1-score: 0.8932
Evaluate pretrained model ( cnnParams_resnet18bigModel_fs_17.pt ) with Filesize = 40 kB
Accuracy: 0.8296
Precision: 0.8497
Recall: 0.8296
F1-score: 0.8312
Evaluate pretrained model ( cnnParams_resnet18bigModel_fs_25.pt ) with Filesize = 5 kB
Accuracy: 0.8424
Precision: 0.8763
Recall: 0.8424
F1-score: 0.8373
Evaluate pretrained model ( cnnParams_resnet18bigModel_fs_25.pt ) with Filesize = 10 kB
Accuracy: 0.9530
Precision: 0.9583
Recall: 0.9530
F1-score: 0.9532
Evaluate pretrained model ( cnnParams_resnet18bigModel_fs_25.pt ) with Filesize = 17 kB
Accuracy: 0.9662
Precision: 0.9677
Recall: 0.9662
F1-score: 0.9662
Evaluate pretrained model ( cnnParams_resnet18bigModel_fs_25.pt ) with Filesize = 25 kB
Accuracy: 0.9630
Precision: 0.9635
Recall: 0.9630
F1-score: 0.9630
Evaluate pretrained model ( cnnParams_resnet18bigModel_fs_25.pt ) with Filesize = 32 kB
Accuracy: 0.9518
Precision: 0.9530
Recall: 0.9518
F1-score: 0.9518
Evaluate pretrained model ( cnnParams_resnet18bigModel_fs_25.pt ) with Filesize = 40 kB
Accuracy: 0.9260
Precision: 0.9284
Recall: 0.9260
F1-score: 0.9261
Evaluate pretrained model ( cnnParams_resnet18bigModel_fs_32.pt ) with Filesize = 5 kB
Accuracy: 0.8296
Precision: 0.8625
Recall: 0.8296
F1-score: 0.8204
Evaluate pretrained model ( cnnParams_resnet18bigModel_fs_32.pt ) with Filesize = 10 kB
Accuracy: 0.9334
Precision: 0.9400
Recall: 0.9334
F1-score: 0.9337
Evaluate pretrained model ( cnnParams_resnet18bigModel_fs_32.pt ) with Filesize = 17 kB
Accuracy: 0.9544
Precision: 0.9562
Recall: 0.9544
F1-score: 0.9544
Evaluate pretrained model ( cnnParams_resnet18bigModel_fs_32.pt ) with Filesize = 25 kB
Accuracy: 0.9550
Precision: 0.9555
Recall: 0.9550
F1-score: 0.9550
Evaluate pretrained model ( cnnParams_resnet18bigModel_fs_32.pt ) with Filesize = 32 kB
Accuracy: 0.9448
Precision: 0.9449
Recall: 0.9448
F1-score: 0.9446
Evaluate pretrained model ( cnnParams_resnet18bigModel_fs_32.pt ) with Filesize = 40 kB
Accuracy: 0.9282
Precision: 0.9292
Recall: 0.9282
F1-score: 0.9275
Evaluate pretrained model ( cnnParams_resnet18bigModel_fs_40.pt ) with Filesize = 5 kB
Accuracy: 0.7325
Precision: 0.7915
Recall: 0.7325
F1-score: 0.7235
Evaluate pretrained model ( cnnParams_resnet18bigModel_fs_40.pt ) with Filesize = 10 kB
Accuracy: 0.9006
Precision: 0.9167
Recall: 0.9006
F1-score: 0.9007
Evaluate pretrained model ( cnnParams_resnet18bigModel_fs_40.pt ) with Filesize = 17 kB
Accuracy: 0.9380
Precision: 0.9446
Recall: 0.9380
F1-score: 0.9379
Evaluate pretrained model ( cnnParams_resnet18bigModel_fs_40.pt ) with Filesize = 25 kB
Accuracy: 0.9542
Precision: 0.9562
Recall: 0.9542
F1-score: 0.9541
Evaluate pretrained model ( cnnParams_resnet18bigModel_fs_40.pt ) with Filesize = 32 kB
Accuracy: 0.9572
Precision: 0.9588
Recall: 0.9572
F1-score: 0.9572
Evaluate pretrained model ( cnnParams_resnet18bigModel_fs_40.pt ) with Filesize = 40 kB
Accuracy: 0.9494
Precision: 0.9505
Recall: 0.9494
F1-score: 0.9491



