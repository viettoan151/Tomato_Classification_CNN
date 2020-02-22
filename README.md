# Tomato_Classification
**grabcut.py:**
Do background removal

**feature_extraction.py:**
image to be resized, converted to HSV, removed background, and measured color histogram.
Each histogram will be stored to .npy file and labeled in file label.csv

**main.py:**
handle traning and evaluating for tomato color classification network

**image_testing.py:**
Test trained model with a wild image
