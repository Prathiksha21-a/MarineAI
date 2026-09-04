from ultralytics import YOLO
model = YOLO("models/fish_best.pt")
metrics = model.val(data=r"C:\Users\Prathiksha\Downloads\Fish_SingleClass_Dataset\Fish_SingleClass_Dataset\data.yaml")
print("mAP50-95:", metrics.box.map)
print("mAP50:", metrics.box.map50)
print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)