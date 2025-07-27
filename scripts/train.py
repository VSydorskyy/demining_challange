from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model on the COCO8 dataset for 100 epochs
train_results = model.train(
    # data="/gpfs/helios/home/volodymyr1/src/demining_challange/data/dummy_coco8/coco8.yaml",  # Path to dataset configuration file
    data="/gpfs/helios/home/volodymyr1/src/demining_challange/data/coco8_without_empty/fold_2/coco8.yaml",
    epochs=50,  # Number of training epochs
    imgsz=640,  # Image size for training
    device=0,  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
    augment=True,  # Enable data augmentation
    degrees=25.0,
    translate=0.25,
    scale=0.2,
    shear=10.0,
    perspective=0.001,
    fliplr=0.5,
    flipud=0.5,
    batch=32
)