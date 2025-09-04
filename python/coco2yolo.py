from ultralytics.data.converter import convert_coco

convert_coco(labels_dir="/home/kjonghun0828/firewatcher_ai_learning/sample_data/labels", use_segments=True, use_keypoints=False, cls91to80=False)
#coco json -> yolo format txt로 변환.