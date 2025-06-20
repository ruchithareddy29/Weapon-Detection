# Weapon-Detection

# Introduction
 - Automated weapon detection systems in surveillance are essential for enhancing security and preventing crimes.

 - EfficientDet is a cutting-edge object detection framework known for its accuracy and efficiency in real-time applications like weapon detection.

 - Integrating audio alerts ensures immediate notification to security personnel or individuals during weapon detection.


# Methodology

**Dataset Preparation:**
A custom labeled image dataset was created, containing class guns, knives, etc. The images were labeled with bounding boxes and class labels, and the annotations were saved in COCO format.

**Image Preprocessing:**
Resized all input images to a fixed resolution of 512 × 512 pixels ensured consistent Input and hence more robustness for the model. The images were normalized using ImageNet to keep the input distribution aligned with that of the pre-trained backbone. During training, the dataset were augmented by resizing, horizontal flipping, and color normalization to make the model generalized.

**Model Architecture:**
The architecture used is EfficientDet-D0. It is considered to be an efficient balance of performance and computational cost. EfficientNet acts as the base network for Feature extraction and the BiFPN 
is responsible for feature aggregation across various scales. A custom HeadNet is used to predict object classes.

**Model Training:**
The model was trained with the AdamW optimizer. During training, the minimization of a multitask loss function was performed combining the focal loss and smooth L1 loss.

**Real-Time Inference and Alert System**
During inference, user-uploaded images go through the same preprocessing pipeline before being sent through the trained model for inference. Post-processing includes removing detections that are 
classified below a confidence level. If any object detected is classified as a weapon, an alert is automatically triggered.
                                       
# Conclusion
In conclusion, the system delivers a reliable and efficient solution for weapon detection using the EfficientDet model. It accurately classifies images into three categories: gun, knife, and no weapon. The integration of an alert mechanism ensures immediate sound notifications when a weapon is detected. This weapon detection and alert system enhances public safety by combining accurate weapon identification with  alerts. Its ability to process  datasets and classify weapon types ensures timely threat detection, making it highly suitable for high-risk environments.


                                                                                                               
