# Traffic Sign Classification using Custom CNN and MobileNetV2

## Overview
This repository contains two distinct approaches for classifying traffic signs using the GTSRB - German Traffic Sign Recognition Benchmark dataset:
1. A fully custom Convolutional Neural Network (CNN) built from scratch.
2. A model based on the pre-trained MobileNetV2 for feature extraction and classification.

The main goal of the project is to accurately classify images of traffic signs into 43 different categories. Both models have been designed, trained, and evaluated using Python and TensorFlow, leveraging tools like Kaggle, Google Colab, and Jupyter Notebook.

## Repository Structure
- `CAV_A2.ipynb`: Python script for the custom CNN model.
- `cav-project.ipynb`: Python script for the MobileNetV2-based model.
- `CAV_A2.tex`: LaTeX code for the project report detailing both approaches.
- `README.md`: This file, providing an overview of the repository.
- `LICENSE`: License information for the project.
- `.gitignore`: Git ignore file to exclude certain files and directories.

## Project Links
- GitHub Repository: [https://github.com/sakibsadmanshajib/ELEC8900-CAV/](https://github.com/sakibsadmanshajib/ELEC8900-CAV/)
- Kaggle Notebook: [https://www.kaggle.com/code/sakibsadmanshajib/cav-project](https://www.kaggle.com/code/sakibsadmanshajib/cav-project)

## Custom CNN Approach
The custom CNN was built from scratch with the goal of exploring a basic yet effective architecture for traffic sign classification. The model consists of multiple convolutional layers, followed by max-pooling layers, dropout layers to prevent overfitting, and finally dense layers for classification. The final model achieved an accuracy of 94.30% on the test set.

### Key Features:
- Custom-built convolutional architecture.
- Data augmentation to enhance generalization.
- Achieved 94.30% accuracy, precision of 0.95, recall of 0.94, and F1 score of 0.94.

## MobileNetV2 Approach
To improve classification performance, I used MobileNetV2 as a pre-trained model. This allowed for better feature extraction as MobileNetV2 has been trained on the ImageNet dataset. Additional layers such as GlobalAveragePooling, Dropout, and Dense were added to the model to adapt it to the 43-category classification task. The MobileNetV2-based model achieved an accuracy of 97.08%, outperforming the custom CNN.

### Key Features:
- Utilizes MobileNetV2 as a feature extractor.
- Fine-tuned using the GTSRB dataset.
- Achieved 97.08% accuracy, precision of 0.97, recall of 0.97, and F1 score of 0.97.

## Data Augmentation and Training Details
- **Data Augmentation**: To enhance generalization, random brightness, contrast, saturation, and JPEG quality adjustments were applied during training. Data augmentation was performed in-line, leveraging GPU resources for efficient training.
- **Training Environment**: Initially, training was performed on a local CPU, but due to computational constraints, I switched to Kaggle and Google Colab, which provided GPU support and enabled faster training.
- **Concurrency**: Loading the dataset involved significant pre-processing, such as resizing and cropping images. To expedite this, concurrency using ThreadPoolExecutor was used for parallel data loading, making the process much more efficient.

## How to Run the Code
1. Clone the repository:
    ```sh
    git clone https://github.com/sakibsadmanshajib/ELEC8900-CAV.git
    ```
2. Navigate to the project directory:
    ```sh
    cd ELEC8900-CAV
    ```
3. Install Jupyter Notebook and required libraries:
    ```sh
    pip install jupyter
    pip install tensorflow
    pip install matplotlib
    pip install numpy
    pip install pandas
    ```
4. Run the Jupyter Notebook:
    ```sh
    jupyter notebook
    ```
5. Open the `CAV_A2.ipynb` or `cav-project.ipynb` script in the Jupyter Notebook and run the cells to train and evaluate the models.

## Report
The complete project report, including details about both approaches, data augmentation, and training decisions, can be found in the LaTeX file (`CAV_A2.tex`). The report details my observations, comparisons, and justifications for each design decision. This file can be compiled using any LaTeX editor or by using command-line tools like `pdflatex`.

## Results
The MobileNetV2 model achieved better results in terms of accuracy and performance compared to the custom CNN model:
- **Custom CNN Model**: Accuracy of 94.30%
- **MobileNetV2 Model**: Accuracy of 97.08%

Both models have been evaluated using metrics such as Confusion Matrix, Precision, Recall, and F1 score, with MobileNetV2 showing superior performance in all categories.

## Contact
For any questions or issues, feel free to contact me via GitHub.

## License
This project is open source and available under the [MIT License](LICENSE).

\textbf{Note}: The dataset used in this project is available from the GTSRB - German Traffic Sign Recognition Benchmark dataset on Kaggle.

