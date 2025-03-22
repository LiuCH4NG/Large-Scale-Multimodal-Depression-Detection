python mainkfold.py --train True --num_folds 10 --epochs 125 --batch_size 16 --learning_rate 1e-5 --model MultiModalDepDet --fusion lt --dataset dvlog-dataset

python mainkfold.py --train True --num_folds 10 --epochs 125 --batch_size 16 --learning_rate 1e-5 --model MultiModalDepDet --fusion it --dataset dvlog-dataset

python mainkfold.py --train True --num_folds 10 --epochs 125 --batch_size 16 --learning_rate 1e-5 --model MultiModalDepDet --fusion ia --dataset dvlog-dataset


python mainkfold.py --train True --num_folds 10 --epochs 125 --batch_size 16 --learning_rate 1e-5 --model MultiModalDepDet --fusion lt --dataset lmvd-dataset

python mainkfold.py --train True --num_folds 10 --epochs 125 --batch_size 16 --learning_rate 1e-5 --model MultiModalDepDet --fusion it --dataset lmvd-dataset

python mainkfold.py --train True --num_folds 10 --epochs 125 --batch_size 16 --learning_rate 1e-5 --model MultiModalDepDet --fusion ia --dataset lmvd-dataset


python main.py --train True --epochs 225 --batch_size 16 --learning_rate 1e-5 --model MultiModalDepDet --fusion lt --dataset dvlog-dataset

python main.py --train True --epochs 225 --batch_size 16 --learning_rate 1e-5 --model MultiModalDepDet --fusion lt --dataset lmvd-dataset
