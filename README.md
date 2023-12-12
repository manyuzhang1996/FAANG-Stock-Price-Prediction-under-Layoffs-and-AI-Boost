
# Read-Time Deep Face Detection Model

A project leverages Python and TensorFlow to create a deep face detection model which can perform real-time detection.

![Deep Face Detection Project Architecture copy](https://github.com/manyuzhang1996/Deep-Face-Detection-Model-with-Python-and-TensorFlow/assets/111943220/88f2dc1b-073b-44f2-ab6b-013295b8af15)




## Tech Stack
Languages: 
* Python

Other Platforms, Tools and Libraries: 
* TensorFlow: https://github.com/tensorflow/tensorflow
* OpenCV: https://github.com/opencv/opencv
* Label Me: https://github.com/wkentaro/labelme
* Albumentations: https://github.com/albumentations-team/albumentations
* Keras: https://github.com/keras-team/keras


## Data Source
The images used for this project are real images collected with webcam using OpenCV.

There are 90 images collected. After image augmentation with albumentations, we have 5400 images in total. 

Images are of different backgrounds, with and/ or without human face in it. 
## Project Architecture


1. Collect images with webcam using OpenCV
2. Annotate images by adding bounding boxes around the face with LabelMe
3. Conduct image augmentation with Albumentations
4. Define loss function which combines face classification loss and bounding coordinates regression loss
5. Design the neural network for this project. Uses the VGG16 model (without final layer) as base, append the classification model and regression model to VGG16 base model respectively
6. Train and save the model with Keras.model
7. Perform real time face detection with OpenCV

![Deep Face Detection Project Architecture copy](https://github.com/manyuzhang1996/Deep-Face-Detection-Model-with-Python-and-TensorFlow/assets/111943220/88f2dc1b-073b-44f2-ab6b-013295b8af15)





## Run Locally

Clone the project

```bash
  gh repo clone manyuzhang1996/Deep-Face-Detection-Model-with-Python-and-TensorFlow
```

## Contact
Manyu Zhang (zhangmanyuzmy@gmail.com)
