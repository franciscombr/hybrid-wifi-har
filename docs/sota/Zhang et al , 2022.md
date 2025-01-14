# Zhang et al., 2022

Title: Widar3.0: Zero-Effort Cross-Domain Gesture Recognition With Wi-Fi
Year: 2022
Status: Need to Read
Type: Method
Tags: COTS WiFi, Gesture recognition, Training, Wireless communication, Wireless fidelity, Wireless sensor networks, feature extraction, sensors, wireless sensing
Interesting?: ⭐️⭐️
Last edited: December 5, 2024 2:01 PM
Abstract: With the development of signal processing technology, the ubiquitous Wi-Fi devices open an unprecedented opportunity to solve the challenging human gesture recognition problem by learning motion representations from wireless signals. Wi-Fi-based gesture recognition systems, although yield good performance on specific data domains, are still practically difficult to be used without explicit adaptation efforts to new domains. Various pioneering approaches have been proposed to resolve this contradiction but extra training efforts are still necessary for either data collection or model re-training when new data domains appear. To advance cross-domain recognition and achieve fully zero-effort recognition, we propose Widar3.0, a Wi-Fi-based zero-effort cross-domain gesture recognition system. The key insight of Widar3.0 is to derive and extract domain-independent features of human gestures at the lower signal level, which represent unique kinetic characteristics of gestures and are irrespective of domains. On this basis, we develop a one-fits-all general model that requires only one-time training but can adapt to different data domains. Experiments on various domain factors (i.e. environments, locations, and orientations of persons) demonstrate the accuracy of 92.7% for in-domain recognition and 82.6%-92.4% for cross-domain recognition without model re-training, outperforming the state-of-the-art solutions.
Item Type: Journal Article
Authors: Zhang, Yi
Zheng, Yue
Qian, Kun
Zhang, Guidong
Liu, Yunhao
Wu, Chenshu
Yang, Zheng
URL: https://ieeexplore.ieee.org/document/9516988
Project: ML

### Overview

- Objective
    - To address the challenges of cross-domain gesture recognition using Wi-Fi Channel State Information (CSI) by proposing **Widar3.0**, a system capable of domain-independent recognition through a unique feature extraction method called the **Body-Coordinate Velocity Profile (BVP)**.
- Key Contributions
    - **BVP Generation:** Developed a novel domain-independent feature (BVP) that models the velocity distribution of body parts during gestures in body coordinates.
    - **Zero-Effort Cross-Domain Recognition:** Introduced a system requiring no retraining or additional data collection for new domains.
    - **Hybrid CNN-GRU Model:** Designed a deep learning model that fully exploits spatial and temporal features of BVP for gesture recognition.
    - Demonstrated state-of-the-art performance across different domain factors, such as locations, orientations, and environments.

- Keywords
    - Channel State Information (CSI)
    - Cross-Domain Gesture Recognition
    - Body-Coordinate Velocity Profile (BVP)
    - Deep Learning
    - Zero-Effort Recognition

### Methods

- **Approach/Architecture:**
    - **BVP Generation Module:**
        - Extracts domain-independent features from raw CSI by applying compressed sensing to Doppler frequency shifts (DFS).
        - Dynamic link selection mitigates interference from occlusions and irrelevant motion.
    - **Gesture Recognition Module:**
        - Uses BVP as input to a hybrid **CNN-GRU** model:
            - **CNN:** Extracts spatial features from each BVP snapshot.
            - **GRU:** Models temporal dependencies in the BVP series.
        - Outlier detection ensures classification only within the predefined gesture set.
- **Dataset:**
    - **Gesture Data:**
        - **Dataset 1:** 16 participants, 5 locations, 5 orientations, 6 gestures, 5 instances each (~12,750 samples).
        - **Dataset 2:** 2 participants, 10 complex gestures (e.g., writing digits 0-9), 5,000 samples.
    - Collected in 3 diverse environments: classroom, hall, and office.
    - [https://ieee-dataport.org/open-access/widar-30-wifi-based-activity-recognition-dataset](https://ieee-dataport.org/open-access/widar-30-wifi-based-activity-recognition-dataset)
- **Techniques Used:**
    - **Feature Extraction:** Compressed sensing for BVP generation from DFS profiles.
    - **Classification:** Hybrid CNN-GRU deep learning model for gesture recognition.
    - **Normalization:** Standardizes BVP features across individuals and instances.

### Results

- **Metrics Reported:**
    - Accuracy (in-domain and cross-domain).
    - Generalization to new domains (location, orientation, and environment).
    - Computational efficiency.
- **Performance Highlights:**
    - **In-Domain Accuracy:** 92.7% for Dataset 1 gestures.
    - **Cross-Domain Accuracy:**
        - Location: 89.7%
        - Orientation: 82.6%
        - Environment: 92.4%
    - **Robustness:** High performance across users with diverse physical characteristics and under different clothing conditions.
- **Comparison to Baselines:**
    - Outperformed existing methods (e.g., CARM, EI, and CrossSense) in cross-domain recognition.
    - BVP demonstrated a 15% improvement over DFS features in accuracy, showcasing its domain independence.

### Analysis

- **Strengths:**
    - Introduced a novel domain-independent feature (BVP) that significantly improves cross-domain generalization.
    - Demonstrated practical implementation with off-the-shelf Wi-Fi hardware.
    - Required only one-time model training, reducing data collection and retraining efforts.
- **Weaknesses/Gaps:**
    - Performance depends on line-of-sight (LOS) conditions and horizontal placement of Wi-Fi devices.
    - Limited vertical resolution due to the experimental setup, impacting recognition of gestures with significant vertical movement.
    - Higher computational complexity for BVP extraction compared to traditional features.
- **Opportunities for Improvement:**
    - Extend BVP to capture 3D velocity distributions for better vertical resolution.
    - Explore lightweight models for edge-device deployment.
    - Investigate techniques to handle non-line-of-sight (NLOS) scenarios.

### Relevance to the Project:

- **Insights or Ideas:**
    - The domain-independent nature of BVP aligns with the goal of robust HAR systems.
    - Hybrid CNN-GRU architecture can be adapted for other features like CSI spectrograms or phase information.
    - The dataset diversity provides a strong benchmark for evaluating new models.
- **Potential for Reuse:**
    - BVP generation method and dynamic link selection can enhance feature extraction in your project.
    - Hybrid CNN-GRU model serves as a baseline for comparison with other architectures (e.g., Vision Transformers).
- **Unanswered Questions:**
    - How scalable is the BVP method for real-time applications on resource-constrained devices?
    - Can BVP be enhanced with multimodal data (e.g., integrating visual or inertial sensor data)?