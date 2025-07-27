from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model on the COCO8 dataset for 100 epochs
train_results = model.train(
    # data="/gpfs/helios/home/volodymyr1/src/demining_challange/data/dummy_coco8/coco8.yaml",  # Path to dataset configuration file
    data="data/coco8_without_empty/fold_0/coco8.yaml",
    epochs=50,  # Number of training epochs
    imgsz=640,  # Image size for training
    device=0,  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
)