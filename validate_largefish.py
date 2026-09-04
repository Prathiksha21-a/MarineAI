from ultralytics import YOLO
model = YOLO("models/largefish_best.pt")
metrics = model.val(data=r"C:\Users\Prathiksha\Downloads\largefish_v3\largefish_v3\data.yaml")
print("mAP50-95:", metrics.box.map)
print("mAP50:", metrics.box.map50)
print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)
