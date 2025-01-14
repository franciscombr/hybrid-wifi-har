# Fard Moshiri et al., 2021

Title: A CSI-Based Human Activity Recognition Using Deep Learning
Year: 2021
Status: Skimmed
Type: Method
Tags: Deep learning, Internet of Things, activity recognition, channel state information, smart house
Last edited: November 29, 2024 11:12 AM
Abstract: The Internet of Things (IoT) has become quite popular due to advancements in Information and Communications technologies and has revolutionized the entire research area in Human Activity Recognition (HAR). For the HAR task, vision-based and sensor-based methods can present better data but at the cost of users’ inconvenience and social constraints such as privacy issues. Due to the ubiquity of WiFi devices, the use of WiFi in intelligent daily activity monitoring for elderly persons has gained popularity in modern healthcare applications. Channel State Information (CSI) as one of the characteristics of WiFi signals, can be utilized to recognize different human activities. We have employed a Raspberry Pi 4 to collect CSI data for seven different human daily activities, and converted CSI data to images and then used these images as inputs of a 2D Convolutional Neural Network (CNN) classifier. Our experiments have shown that the proposed CSI-based HAR outperforms other competitor methods including 1D-CNN, Long Short-Term Memory (LSTM), and Bi-directional LSTM, and achieves an accuracy of around 95% for seven activities.
Item Type: Journal Article
Authors: Fard Moshiri, Parisa
Shahbazian, Reza
Nabati, Mohammad
Ghorashi, Seyed Ali
URL: https://www.mdpi.com/1424-8220/21/21/7225
Project: ML

### Overview

- Objective
    - To develop a novel CSI-based human activity recognition framework using deep learning techniques that leverage CSI-to-image conversion and convolutional neural networks (CNNs) for improved accuracy, with the application in smart homes and elderly care systems.
- Key Contributions
    - Introduced a public dataset collected using Raspberry Pi 4 and Nexmon CSI tool for seven activities (walk, run, fall, sit down, stand up, lie down, bend).
    - Proposed converting CSI data into RGB images using pseudocolor plots for 2D-CNN processing, significantly improving HAR accuracy.
    - Compared 2D-CNN with other techniques, including LSTM, BiLSTM with attention, 1D-CNN, and conventional models, achieving a high accuracy of 95%.
    - Demonstrated that 2D-CNN offers lower computational complexity and faster training compared to BiLSTM.
- Keywords
    - CSI
    - HAR
    - Deep Learning
    - Smart Home
    - Internet of Things (IoT)
    - Convolutional Neural Networks (CNN)

### Methods

- **Approach/Architecture:**
    - **2D-CNN:** Converts CSI data to RGB pseudocolor images, allowing parallel analysis of signal patterns.
    - **BiLSTM with Attention:** Processes raw CSI time-series data, focusing on past and future states with weighted features.
    - **Comparison:** Benchmarks 2D-CNN and BiLSTM against 1D-CNN and LSTM.
- **Dataset:**
    - Name: Custom dataset for CSI-based HAR, publicly available on GitHub.
    - Details: Data collected from three participants for seven activities (walk, run, fall, sit down, stand up, lie down, bend), with 420 samples.
    - Preprocessing: Raw CSI amplitude data converted to RGB images; no additional filtering applied to preserve signal integrity.
- **Techniques Used:**
    - Feature Extraction:
        - RGB image generation from CSI matrices for 2D-CNN.
        - Temporal dependency modeling using BiLSTM with attention.
    - Classifiers:
        - 2D-CNN
        - BiLSTM with attention
        - 1D-CNN
        - LSTM

### Results

- **Metrics Reported:**
    - Accuracy, confusion matrices, training and testing time
- **Performance Highlights:**
    - **2D-CNN:** Achieved 95% accuracy with the fastest training time among all methods.
    - **BiLSTM with Attention:** Performed comparably (96% for specific activities) but required higher computational resources.
    - **Other Techniques:** LSTM and 1D-CNN lagged behind in accuracy and were prone to confusion in similar activities like “sit down” and “lie down.”
- **Comparison to Baselines:**
    - Outperformed conventional models like Random Forest and HMM.
    - Improved recognition accuracy and reduced computational complexity compared to ConvLSTM, DenseLSTM, and Fully Connected networks.

### Analysis

- **Strengths:**
    - Novel CSI-to-image conversion for leveraging the strengths of 2D-CNNs.
    - Demonstrates high accuracy for activities with significant body movement (e.g., fall, run) and improved distinction between similar activities using attention mechanisms.
    - Provides a public dataset with accessible hardware, facilitating reproducibility.
- **Weaknesses/Gaps:**
    - Limited sample size (420 samples); potential overfitting risks in neural networks.
    - Dataset restricted to a controlled indoor environment, limiting generalizability to diverse scenarios.
    - Computational efficiency of BiLSTM needs optimization for real-time applications.
- **Opportunities for Improvement:**
    - Collect larger datasets with varied participants and settings for better model generalization.
    - Investigate hybrid models combining CNNs and attention-based temporal models to balance accuracy and computational efficiency.
    - Extend the study to multi-user scenarios and dynamic environments.

### Relevance to the Project:

- **Insights or Ideas:**
    - Converting CSI data to images opens possibilities for applying advanced vision models like Vision Transformers in your project.
    - Attention mechanisms in BiLSTM highlight the potential for incorporating temporal dependencies effectively.
- **Potential for Reuse:**
    - Dataset and CSI-to-image methodology can be adopted for benchmarking your models.
    - Model architectures (2D-CNN and BiLSTM with attention) can serve as baselines for comparison with your proposed methods.
- **Unanswered Questions:**
    - Can the CSI-to-image technique generalize to multi-user or outdoor environments?
    - How does the model handle noise or environmental changes in raw CSI data?