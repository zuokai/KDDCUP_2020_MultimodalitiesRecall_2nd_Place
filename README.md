# KDD Cup 2020 Challenges for Modern E-Commerce Platform: Multimodalities Recall

### Team: MTDP_CVA

Members: Kai Zuo, Chao Ma, Dongshuai Li, Zuo Cao, Xing Xu

### Introduction

![Alt text](https://github.com/zuokai/KDDCUP_2020_MultimodalitiesRecall_2nd_Place/blob/master/pics/1596178061922-image.png)

the official have prepared the real-scenario multimodal data from the mobile Taobao. The dataset consists of Taobao search 
queries and product image features, which is organized into a query-based multimodal retrieval task.The whole training set 
consists of 3M pairs of query and ground-truth product image features. These sample pairs can be taken as the positive 
examples to train your retrieval model. For each product image, the official release the features, locations and 
classification labels of each detected object.
See more details: https://tianchi.aliyun.com/competition/entrance/231786/information

### Codes

```
kdd_evaluate_ensemble 
  |--external_resources
  |--user_data
     |--bert_config.json
     |--bert_model.ckpt.data-00000-of-00001
     |--bert_model.ckpt.index
     |--pytorch_model.bin
     |--query_labels.txt
     |--vocab.txt
  |--data
     |--multimodal_labels.txt
     |--testA
        |--testA.tsv
     |--testB
        |--testB.tsv
     |--train
     |--valid
  |--models
     |--BEST.pth
     |--ImageBertKDD.ckpt-85002.data-00000-of-00001
     |--ImageBertKDD.ckpt-85002.index
     |--ImageBertKDD.ckpt-85002.meta
     |--model_attention_kdd_am_word_match_finetune_valid.ckpt-251.data-00000-of-00001
     |--model_attention_kdd_am_word_match_finetune_valid.ckpt-251.index
     |--model_attention_kdd_am_word_match_finetune_valid.ckpt-251.meta
  |--prediction_result
  |--code
```

cuda 9.0.176
cudnn 7.1.4

下载模型文件
链接:https://pan.baidu.com/s/1tkmpLdF_uguB7VFPNn0cYQ  密码:gwdc

运行脚本
python2 code/main.py
