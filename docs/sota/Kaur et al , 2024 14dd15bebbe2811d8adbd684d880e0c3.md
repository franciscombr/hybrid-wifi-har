# Kaur et al., 2024

Title: Human activity recognition: A comprehensive review
Year: 2024
Status: Need to Read
Type: Review
Tags: actions, data acquisition, gestures, sensors, wearable devices
Last edited: November 29, 2024 12:54 PM
Abstract: Human Activity Recognition (HAR) is a highly promising research area meant to automatically identify and interpret human behaviour using data received from sensors in various contexts. The potential uses of HAR are many, among them health care, sports coaching or monitoring the elderly or disabled. Nonetheless, there are numerous hurdles to be circumvented for HAR's precision and usefulness to be improved. One of the challenges is that there is no uniformity in data collection and annotation making it difficult to compare findings among different studies. Furthermore, more comprehensive datasets are necessary so as to include a wider range of human activities in different contexts while complex activities, which consist of multiple sub-activities, are still a challenge for recognition systems. Researchers have proposed new frontiers such as multi-modal sensor data fusion and deep learning approaches for enhancing HAR accuracy while addressing these issues. Also, we are seeing more non-traditional applications such as robotics and virtual reality/augmented world going forward with their use cases of HAR. This article offers an extensive review on the recent advances in HAR and highlights the major challenges facing this field as well as future opportunities for further researches.
Item Type: Journal Article
Authors: Kaur, Harmandeep
Rani, Veenu
Kumar, Munish
URL: https://onlinelibrary.wiley.com/doi/abs/10.1111/exsy.13680
Project: ML

### Overview

- Objective
    - To provide a comprehensive review of Human Activity Recognition (HAR) systems, including their frameworks, state-of-the-art techniques, applications and challenges, while outlining future directions for advancing HAR research.
- Key Contributions
    - Extensive survey of HAR techniques, ranging from traditional machine learning methods to deep learning-based approaches.
    - Overview of data acquisition, preprocessing, feature extraction, and classification methods in HAR systems.
    - Discussion on challenges, including data variability, generalization, scalability, and the need for multimodal data fusion.
    - Comprehensive presentation of publicly available datasets and evaluation metrics.
    - Identification of gaps in current research and proposal of future research directions.

- Keywords
    - Human Activity Recognition (HAR)
    - Machine Learning
    - Deep Learning
    - Feature Extraction
    - Multimodal Data Fusion
    
    ---
    

### Methods

- **Approach/Architecture:**
    - **Framework Overview:**
        - Five-phase HAR process: data acquisition, preprocessing, feature extraction, dimensionality reduction, and classification.
        - Emphasis on integrating multimodal data for improved accuracy and robustness.
    - **Data Processing Techniques:**
        - Preprocessing steps, including denoising (linear, wavelet, Kalman filters) and background subtraction.
        - Feature extraction using both handcrafted and deep learning techniques.
    - **Classification Techniques:**
        - Traditional machine learning (SVM, Random Forest).
        - Deep learning methods (CNN, RNN, LSTM, and hybrid models).
- **Dataset:**
    - Comprehensive review of 20+ publicly available datasets (e.g., UCI-HAR, PAMAP2, WISDM).
    - Detailed dataset attributes:
        - **UCI-HAR:** 6 activities, multivariate time-series data from 30 subjects.
        - **PAMAP2:** 18 activities, data collected from 9 subjects with inertial sensors.
        - **WISDM:** Smartphone-based dataset with accelerometer data for 6 activities.
- **Techniques Used:**
    - **Feature Extraction:**
        - Handcrafted (e.g., HOG, SIFT, LBP).
        - Automated (e.g., CNN for spatial, RNN for temporal patterns).
    - **Dimensionality Reduction:**
        - PCA, LDA, Kernel PCA for feature compression.
    - **Classification:**
        - Machine learning (e.g., k-NN, SVM).
        - Hybrid models (e.g., CNN-LSTM).

### Results

- **Metrics Reported:**
    - Accuracy, precision, recall, F1-score, and computational efficiency.
    - Benchmark comparisons for deep learning models across datasets.
- **Performance Highlights:**
    - CNN and RNN models outperform traditional approaches in capturing complex activity patterns.
    - Hybrid approaches (e.g., CNN-LSTM) show superior accuracy for sequential data.
    - Identified limitations in generalization for cross-dataset evaluations.
    

### Analysis

- **Strengths:**
    - Extensive coverage of HAR frameworks and methodologies.
    - Comprehensive cataloging of datasets and benchmarks.
    - In-depth discussion on preprocessing and feature extraction techniques.
    - Clear identification of challenges and future opportunities.
- **Weaknesses/Gaps:**
    - Limited exploration of HAR systems for multi-user scenarios.
    - Insufficient focus on real-time applications and edge deployment.
    - Challenges in handling noisy and incomplete data remain underexplored.
- **Opportunities for Improvement:**
    - Explore robust methods for multi-user activity recognition.
    - Develop lightweight models optimized for edge computing.
    - Enhance dataset diversity to improve generalization capabilities.

### Relevance to the Project:

- **Insights or Ideas:**
    - The emphasis on hybrid deep learning models aligns with your exploration of Vision Transformers for HAR.
    - Preprocessing techniques (e.g., denoising, feature fusion) can enhance the robustness of your pipeline.
    - Dataset descriptions offer a benchmark for evaluating your model.
- **Potential for Reuse:**
    - Dataset list and preprocessing methods can directly apply to your work.
    - Hybrid approaches and multimodal fusion techniques are relevant for extending your model architecture.
- **Unanswered Questions:**
    - How do hybrid models like CNN-LSTM compare to Vision Transformers in cross-dataset evaluations?
    - Can multimodal fusion of CSI and other data types improve HAR accuracy?