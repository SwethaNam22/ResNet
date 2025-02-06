🖼️ ResNet on CIFAR-10 🖥️
Overview 🌟
This project uses ResNet, a deep learning model, to classify images from the CIFAR-10 dataset. CIFAR-10 consists of 60,000 32x32 color images across 10 classes. We utilize the ResNet architecture, known for its residual connections, which help train deeper networks effectively by mitigating the vanishing gradient problem.

In this project, we employ ResNet18, a smaller version of ResNet, and fine-tune it on the CIFAR-10 dataset to classify the 10 image classes accurately. This project demonstrates the power of transfer learning, using a pre-trained ResNet model to achieve strong performance on a relatively small dataset.

----------------------------
Dataset 📊
The CIFAR-10 dataset contains images in 10 categories:

Airplane,Automobile,Bird,Cat,Deer,Dog,Frog,Horse,Ship,Truck
The dataset contains 50,000 training images and 10,000 test images. Each image is 32x32 pixels, making it suitable for testing small-scale models like ResNet18.

----------------------------
Workflow 🔄
1. Data Preprocessing 🔧
We apply the following transformations to the CIFAR-10 images:
Resize: Resize the images to 224x224 pixels to match the expected input size of the ResNet model.
ToTensor: Convert images into PyTorch tensors.
Normalize: Normalize the images with a mean and standard deviation of 0.5 for each RGB channel.
----------------
2. Model Architecture 🏗️
We use the pre-trained ResNet18 model, adjusting the final fully connected layer to output 10 classes for CIFAR-10
---------------
3. Training the Model 🏋️
We train the model using Stochastic Gradient Descent (SGD) and CrossEntropyLoss as the loss function. The model is trained for a specified number of epochs.
-------------------
4. Testing and Accuracy Evaluation 🎯
Once the model is trained, we evaluate its performance on the test dataset and calculate the accuracy.
------------------------
Future Improvements 🚀
Implementing data augmentation to increase training data diversity and improve model robustness.
Fine-tuning the learning rate and number of epochs for better performance.
Trying ResNet50 or ResNet101 for deeper models to see if the performance improves.
Exploring other optimization techniques such as Adam for faster convergence.
Integrating real-time image classification capabilities.
