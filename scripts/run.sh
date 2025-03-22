python main.py --train True --train_gender both --test_gender both --epochs 120 --batch_size 16 --learning_rate 1e-4 --model DepMamba --dataset dvlog-dataset

python main.py --train True --train_gender both --test_gender both --epochs 120 --batch_size 16 --learning_rate 1e-4 --model DepMamba --dataset lmvd-dataset

python main.py --train True --train_gender both --test_gender both --epochs 200 --batch_size 16 --learning_rate 1e-4 --model MultiModalDepDet --dataset dvlog-dataset

python main.py --train True --train_gender both --test_gender both --epochs 200 --batch_size 16 --learning_rate 1e-4 --model MultiModalDepDet --dataset lmvd-dataset


python main.py --train True --train_gender both --test_gender both --num_folds 10 --epochs 120 --batch_size 16 --learning_rate 1e-5 --model MultiModalDepDet --dataset dvlog-dataset

python main.py --train True --train_gender both --test_gender both --num_folds 10 --epochs 120 --batch_size 16 --learning_rate 1e-5 --model MultiModalDepDet --dataset lmvd-dataset