create_dataset:
	poetry run python src/segmentation/utils/sample_dataset.py
	poetry run python src/segmentation/utils/train_test_split.py

train:
	poetry run python src/segmentation/scripts/train.py

evaluate:
	poetry run python src/segmentation/scripts/evaluation.py

predict:
	poetry run python src/segmentation/scripts/predict.py