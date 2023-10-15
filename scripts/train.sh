poetry run python $PYTHONPATH/experiments/train.py \
-m model.pretrained=roberta-large \
train.lr=3e-5 \
train.batch_size=32 \
preprocess.max_length=32 \
train.epochs=5 \
train.start_factor=0.5 \
train.save_model=True


poetry run python $PYTHONPATH/experiments/train.py \
-m model.pretrained=bert-base-uncased  \
train.lr=4e-5 \
train.batch_size=16 \
preprocess.max_length=64 \
train.epochs=5 \
train.start_factor=0.5 \
train.save_model=True

poetry run python $PYTHONPATH/experiments/train.py \
-m model.pretrained=bert-large-uncased  \
train.lr=3e-5 \
train.batch_size=32 \
preprocess.max_length=32 \
train.epochs=5 \
train.start_factor=0.5 \
train.save_model=True

poetry run python $PYTHONPATH/experiments/train.py \
-m model.pretrained=roberta-base  \
train.lr=3e-5 \
train.batch_size=128 \
preprocess.max_length=32 \
train.epochs=5 \
train.start_factor=0.5 \
train.save_model=True
