# demining_challange
Code repository for AI Data Jam: Demining Challange

# Run Steps

1. Setup [poetry](https://python-poetry.org/docs/) env
2. Download data and unzip to `data`
3. Run data preprocessing `python split_data_and_convert_into_coco8.py --image_root data/train/images --annotation_root data/train/annotations --save_root data/coco8_without_empty --remove_empty_train`
4. Run training `python scripts/train.py`
