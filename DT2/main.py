# %%
! pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.9/index.html

# %% register dataset
from detectron2.data import DatasetCatalog, MetadataCatalog

def get_document_data_dicts(img_dir, json_file):
    # Function to load your dataset in Detectron2 format
    # Read the json_file and convert the data into Detectron2's format
    # Return a list of dicts with 'file_name', 'height', 'width', and 'annotations' keys

DatasetCatalog.register("document_train", lambda: get_document_data_dicts("train_images", "train_annotations.json"))
DatasetCatalog.register("document_val", lambda: get_document_data_dicts("val_images", "val_annotations.json"))

MetadataCatalog.get("document_train").set(thing_classes=["caption", "heading", "paragraph", "divider", "footnote", "image", "table"])
MetadataCatalog.get("document_val").set(thing_classes=["caption", "heading", "paragraph", "divider", "footnote", "image", "table"])


# %% configure model
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file("path/to/model/config.yaml")
cfg.DATASETS.TRAIN = ("document_train",)
cfg.DATASETS.TEST = ("document_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "path/to/model/weights.pth"  # Path to the pre-trained weights
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 3000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7  # heading, paragraph, divider, footnote, image, table

# %% train
from detectron2.engine import DefaultTrainer

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# %% evaluate
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

evaluator = COCOEvaluator("document_val", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "document_val")
in
