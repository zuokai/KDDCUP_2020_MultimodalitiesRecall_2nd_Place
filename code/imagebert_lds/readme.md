# KDDImageBert
## 环境
python2.7
tensorflow-gpu 1.12.0
其余参见requirements.txt
`pip install -r requirements.txt`
## 运行方式

valid_score:`source main/evaluate_score.sh`或者`cd src && python run_pretraining_evaluate.py`
pred_score:`source main/predict_score.sh`或者`cd src && python run_pretraining_predict_score.py`

## 结果存放位置
`src/validscore_imagebert.txt`
`src/testBscore_imagebert.txt`

## 数据文件存放位置
`dataset/valid.tsv`
`pred/testB.tsv`
