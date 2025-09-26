import json
import os
import shutil

def cocoToYolo(coco_json_path, output_label_dir):
    # COCO JSON 파일 로드
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    # 카테고리 매핑 (클래스 ID -> 이름)
    category_mapping = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
    # 이미지 ID 매핑
    image_id_mapping = {img["id"]: img["file_name"] for img in coco_data["images"]}

    # 어노테이션 변환
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        class_id = ann["category_id"]
        segmentation = ann["segmentation"]

        if img_id not in image_id_mapping or not segmentation:
            continue  # 해당 이미지가 없거나 세그멘테이션 데이터가 없으면 건너뜀

        # 파일명 생성
        image_name = image_id_mapping[img_id]
        label_file_path = os.path.join(output_label_dir, image_name.replace(".jpg", ".txt"))

        # 세그멘테이션 좌표를 YOLO 형식으로 변환
        yolo_lines = []
        for poly in segmentation:
            normalized_poly = [
                f"{x / coco_data['images'][img_id]['width']} {y / coco_data['images'][img_id]['height']}"
                for x, y in zip(poly[::2], poly[1::2])
            ]
            yolo_lines.append(f"{class_id} " + " ".join(normalized_poly))

        # YOLO 레이블 파일 저장
        with open(label_file_path, "w") as f:
            f.write("\n".join(yolo_lines))

def moveImg(origin_path, moved_path):
    # train 이미지 이동
    for file in os.listdir(origin_path):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):  # 이미지 파일만 이동
            shutil.move(os.path.join(origin_path, file), os.path.join(moved_path, file))

def convertToZeroIndex():
    label_dir = "./datasets/labels/"
    # 모든 .txt 파일 가져오기
    for subset in ["train", "val"]:
        path = os.path.join(label_dir, subset)
        for file_name in os.listdir(path):
            if file_name.endswith(".txt"):
                file_path = os.path.join(path, file_name)

                # 파일 읽고 클래스 ID 변경
                with open(file_path, "r") as f:
                    lines = f.readlines()

                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) > 0 and parts[0] == "1":
                        parts[0] = "0"  # 클래스 ID를 0으로 변경
                        new_lines.append(" ".join(parts))
                    else:
                        print("error 발생")
                # 변경된 내용 저장
                with open(file_path, "w") as f:
                    f.write("\n".join(new_lines))

# 데이터 폴더 경로
data_folder_path = "./data"

#경로 설정
coco_train_path = data_folder_path + "/train/_annotations.coco.json"
coco_valid_path = data_folder_path + "/valid/_annotations.coco.json"
output_label_dirs = ["./datasets/images/train", "./datasets/images/val", "./datasets/labels/train", "./datasets/labels/val"]  # YOLO 레이블이 저장될 폴더

# 폴더 생성
for path_dir in output_label_dirs:
    os.makedirs(path_dir, exist_ok=True)

# Convert COCO TO YOLO
for p in [(coco_train_path, 2), (coco_valid_path, 3)]:
    cocoToYolo(p[0], output_label_dirs[p[1]])

# move Img
moveImg(data_folder_path+"/train", output_label_dirs[0])
moveImg(data_folder_path+"/valid", output_label_dirs[1])

convertToZeroIndex() # 클래스 1 -> 0으로 변경
shutil.rmtree(data_folder_path) # 기존 폴더 삭제