#freeze=0 ou freeze=5 avant freeze=10
#nb epochs de 50 à 100
#tester mosaic = 0 -> mosaic utile quand petits objets


from ultralytics import YOLO
import os

YOLO_PATH = "/sps/mnhn/ameurisse/ElephantReId/data/processed/crops/yolo"
RESULTS_PATH = "/sps/mnhn/ameurisse/ElephantReId/data/results/yolo"
SEED = 42
model = YOLO("yolo26n.pt")

results = model.train(
    data=os.path.join(YOLO_PATH, "data.yaml"),
    project=RESULTS_PATH,
    name="v4",
    pretrained=True,
    seed=SEED,

    classes=[0, 1, 2],

    epochs=100,
    patience=20,

    batch=16,
    workers=1,
    cache=False,

    freeze=0,

    lr0=0.001,

    imgsz=640,

    mosaic=0.2,
    mixup=0.0,
    copy_paste=0.0,

    degrees=5,
    translate=0.1,
    scale=0.3,
    shear=0.0,
    perspective=0.0,

    fliplr=0.0, #rajouter flip si oreille et pas right/lrft
    flipud=0.0,
)


