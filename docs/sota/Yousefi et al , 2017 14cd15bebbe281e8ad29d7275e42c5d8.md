# Yousefi et al., 2017

Title: A Survey on Behavior Recognition Using WiFi Channel State Information
Year: 2017
Status: Skimmed
Type: Review
Tags: Antennas, Behavioral sciences, Doppler shift, OFDM, Receivers, Wireless communication, Wireless fidelity
Interesting?: ⭐️⭐️
Last edited: November 29, 2024 11:04 AM
Abstract: In this article, we present a survey of recent advances in passive human behavior recognition in indoor areas using the channel state information (CSI) of commercial WiFi systems. The movement of the human body parts cause changes in the wireless signal reflections, which result in variations in the CSI. By analyzing the data streams of CSIs for different activities and comparing them against stored models, human behavior can be recognized. This is done by extracting features from CSI data streams and using machine learning techniques to build models and classifiers. The techniques from the literature that are presented herein have great performance; however, instead of the machine learning techniques employed in these works, we propose to use deep learning techniques such as long-short term memory (LSTM) recurrent neural networking (RNN) and show the improved performance. We also discuss different challenges such as environment change, frame rate selection, and the multi-user scenario; and finally suggest possible directions for future work.
Item Type: Journal Article
Authors: Yousefi, Siamak
Narui, Hirokazu
Dayal, Sankalp
Ermon, Stefano
Valaee, Shahrokh
URL: https://ieeexplore.ieee.org/document/8067693
Project: ML

### Overview

- Objective
    - To provide a comprehensive survey of recent advancements in human behaviour recognition using Wi-Fi Channel State Information (CSI) in indoor environments, discussing techniques, challenges, and opportunities for future research.
- Key Contributions
    - Summarizes the use of CSI for passive activity recognition without requiring wearable devices, contrasting it with traditional active systems like wearable sensors
    - Highlights the advantages of CSI over other metrics like RSSI, emphasizing its ability to capture fine-grained changes due to human motion
    - Explores various feature extraction methods (e.g., PCA, STFT, DWT) and classification techniques (e.g., SVM, HMM, LSTM) used in CSI-based HAR.
    - Proposes leveraging deep learning techniques such as LSTM for improved performance and discusses future directions like multi-user activity recognition and robustness to environmental changes.

- Keywords
    - CSI
    - Passive Human Activity Recognition
    - Deep Learning
    - Feature Extraction
    - Deep Learning
    - Indoor Behavior Monitoring

### Methods

- **Approach/Architecture:**
    - Provides a structured pipeline for CSI-based HAR:
        - CSI de-noising using filtering methods (PCA, Butterworth)
        - Feature extraction via FFT, STFT, DWT for frequency and temporal analysis
        - Classification using machine learning methods like Random Forest, HMM, and deep learning (LSTM)
    - Proposes using LSTM networks to directly process raw CSI for feature extraction and classification, bypassing manual preprocessing steps
- **Dataset:**
    - Name: not specified
    - Details:  Six activities (e.g., Lie Down, Fall, Walk, Run, Sit Down, Stand Up), were performed by six participants with 20 trials each.
    - Preprocessing: Includes PCA for de-noising, filtering of high-frequency noise, and transformations like STFT and DWT for feature extraction.
- **Techniques Used:**
    - Feature Extraction: PCA, FFT, STFT, DWT
    - Classifiers: [Mention classification methods, e.g., SVM, RF]

### Results

- **Metrics reported**
    - Accuracy, confusion matrices for activity classification.
- **Performance Highlights:**
    - Random Forest: Moderate performance, struggling with closely related activities (Lie down vs sit down)
    - HMM: improved classification accuracy leveraging temporal dependencies
    - LSTM: best performance across activities due to direct processing of CSI data and ability to model temporal state information
- **Comparison to Baselines:**
    - LSTM outperformed traditional methods (e.g., Random Forest, HMM), especially in distinguishing similar activities like "Fall" and "Lie Down.”

### Analysis

- **Strengths:**
    - Comprehensive overview of methods and challenges in CSI-based HAR.
    - Proposes LSTM as a robust alternative to traditional classification techniques.
    - Discusses practical considerations like environmental changes, transmission rates, and multi-user scenarios.
- **Weaknesses/Gaps:**
    - Limited discussion on leveraging CSI phase information alongside amplitude.
    - Focuses primarily on single-user scenarios; multi-user activity recognition is highlighted as a challenge but not explored in depth.
    - Lack of exploration into real-time or resource-constrained applications.
- **Opportunities for Improvement:**
    - Investigate the integration of amplitude and phase information for feature extraction.
    - Address scalability issues for dynamic and multi-user environments.
    - Explore lightweight deep learning models for edge-device deployment.

### Relevance to the Project:

- **Insights or Ideas:**
    - The use of LSTM for direct feature extraction aligns well with the interest in exploring advanced deep learning methods (Vision Transformers)
    - Emphasizes the importance of de-noising techniques and frequency-domain analysis which can inform the preprocessing pipeline
- **Potential for Reuse:**
    - Methodologies like PCA for de-noising and STFT/DWT for feature extraction can be directly applied or extended.
    - Dataset parameters (6 activities, 6 participants, 20 trials) provide a good benchmark for designing your experiments.
- **Unanswered Questions:**
    - How to adapt and enhance LSTM or transformer architectures to exploit CSI phase information effectively?
    - What techniques can ensure robust performance across varied environments and multi-user scenarios?