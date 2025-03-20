# from ultralytics import YOLO

# model = YOLO("best.pt")
# results = model.predict("test_sample.jpg", save=True)  # Replace "test.jpg" with an actual image

# print("✅ Processing complete! Check `runs/detect/predict/` for results.")













# from ultralytics import YOLO
# import cv2
# import os

# # Load the YOLOv8 model
# # model = YOLO("best.pt")  # Ensure "best.pt" is in the same directory
# MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")
# model = YOLO(MODEL_PATH)


# # Run inference on the test image
# input_image = "test_sample.jpg"  # Replace with the actual test image filename
# results = model.predict(input_image, save=True)  # Process and save results

# # Get the saved output image path
# output_dir = "runs/detect/predict"
# output_images = sorted(os.listdir(output_dir), key=lambda x: os.path.getmtime(os.path.join(output_dir, x)))

# if output_images:
#     output_image_path = os.path.join(output_dir, output_images[-1])  # Get the latest processed image
#     print(f"✅ Processed image saved at: {output_image_path}")

#     # Display the output image
#     output_image = cv2.imread(output_image_path)
#     cv2.imshow("Processed Image", output_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print("❌ No processed images found. Check if YOLO saved the output correctly.")











from ultralytics import YOLO
import cv2
import os
import glob

# Load the YOLOv8 model
model = YOLO("best.pt")  # Ensure "best.pt" is in the same directory

# Run inference on the test image
input_image = "sample2.jpg"  # Replace with your test image
results = model.predict(input_image, save=True)  # Process and save results

# Find the latest YOLOv8 output directory
runs_dir = "runs/detect/"
latest_folder = max(glob.glob(os.path.join(runs_dir, "predict*")), key=os.path.getmtime)

# Get the latest processed image
processed_images = sorted(glob.glob(os.path.join(latest_folder, "*.jpg")), key=os.path.getmtime)

if processed_images:
    output_image_path = processed_images[-1]  # Get the latest processed image
    print(f"✅ Processed image saved at: {output_image_path}")

    # Display the output image
    output_image = cv2.imread(output_image_path)
    cv2.imshow("Processed Image", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("❌ No processed images found. Check if YOLO saved the output correctly.")








