# Määttä et al., 2024

Title: Spatio-Temporal 3D Point Clouds from WiFi-CSI Data via Transformer Networks
Year: 2024
Status: Skimmed
Tags: Computer Science - Machine Learning, Electrical Engineering and Systems Science - Signal Processing
Last edited: January 14, 2025 11:34 AM
Abstract: Joint communication and sensing (JC\&S) is emerging as a key component in 5G and 6G networks, enabling dynamic adaptation to environmental changes and enhancing contextual awareness for optimized communication. By leveraging real-time environmental data, JC\&S improves resource allocation, reduces latency, and enhances power efficiency, while also supporting simulations and predictive modeling. This makes it a key technology for reactive systems and digital twins. These systems can respond to environmental events in real-time, offering transformative potential in sectors like smart cities, healthcare, and Industry 5.0, where adaptive and multimodal interaction is critical to enhance real-time decision-making. In this work, we present a transformer-based architecture that processes temporal Channel State Information (CSI) data, specifically amplitude and phase, to generate 3D point clouds of indoor environments. The model utilizes a multi-head attention to capture complex spatio-temporal relationships in CSI data and is adaptable to different CSI configurations. We evaluate the architecture on the MM-Fi dataset, using two different protocols to capture human presence in indoor environments. The system demonstrates strong potential for accurate 3D reconstructions and effectively distinguishes between close and distant objects, advancing JC\&S applications for spatial sensing in future wireless networks.
Item Type: Preprint
Authors: Määttä, Tuomas
Sharifipour, Sasan
López, Miguel Bordallo
Casado, Constantino Álvarez
URL: http://arxiv.org/abs/2410.16303
Project: ML

### **Overview**

### **Objective**

The paper introduces a novel transformer-based framework, **CSI2PC**, for generating 3D point clouds from Wi-Fi Channel State Information (CSI) data. The system leverages temporal and spatial features of CSI amplitude and phase to reconstruct indoor environments, including human presence and furniture placement, enabling advanced applications in joint communication and sensing (JC&S).

### **Key Contributions**

1. **Transformer-Based Architecture:** Proposed a deep learning architecture that processes temporal CSI data to generate 3D point clouds, incorporating spatio-temporal relationships.
2. **Novel Dataset Use:** Utilized the **MM-Fi Dataset**, which includes diverse environments, multiple sensing modalities, and human activity data, to validate the framework.
3. **Cross-Domain Generalization:** Assessed the model’s performance using **subject-based** and **room-based split protocols**, showcasing its adaptability to unseen subjects and environments.
4. **Loss Function Design:** Combined **Chamfer Loss** (for geometric accuracy) and **Feature Transform Regularization** to ensure accurate and stable 3D point cloud generation.

### **Keywords**

- Wi-Fi CSI
- 3D Point Clouds
- Transformer Networks
- Spatial Sensing
- Joint Communication and Sensing (JC&S)
- Indoor Environment Modeling

---

### **Methods**

### **Architecture Overview**

1. **Input Data:**
    - CSI data of shape [3,114,2,10], representing 3 antennas, 114 subcarriers, 2 channels (amplitude and phase), and 10 time slices.
        
        [3,114,2,10][3, 114, 2, 10]
        
    - Data preprocessed into a tensor of shape [F,2,T], where F=A×S (antenna-subcarrier pairs).
        
        [F,2,T][F, 2, T]
        
        F=A×SF = A \times S
        
2. **Model Components:**
    - **Temporal Encoding:** One-dimensional convolutional layers to extract temporal patterns.
    - **Positional Encoding:** Embeddings for antennas and subcarriers to preserve spatial context.
    - **Transformer Encoder:** Multi-head self-attention mechanism to capture spatio-temporal dependencies.
    - **Transformer Decoder:** Maps features to 3D spatial coordinates using cross-attention and learned point queries.
3. **Loss Functions:**
    - **Chamfer Loss:** Ensures geometric similarity between predicted and ground truth point clouds.
    - **Feature Transform Regularization:** Maintains orthogonality of the transformation matrix, stabilizing reconstructions.

---

### **Dataset**

- **MM-Fi Dataset:** Multi-modal dataset including Wi-Fi CSI, LiDAR, mmWave radar, and RGB-D data.
    - Data collected from 40 participants performing 27 activities (daily activities and rehabilitation exercises) in two indoor room configurations.
    - Wi-Fi CSI captured at 1000Hz and downsampled to 100Hz across 114 subcarriers with customized OpenWrt firmware.

### **Protocols**

1. **Subject-Based Split Protocol:** Trains on data from 32 subjects, validates on 2, and tests on 8 unseen subjects.
2. **Room-Based Split Protocol:** Trains and validates on data from one room configuration and tests on another to evaluate spatial generalization.

---

### **Results**

### **Performance Metrics**

- **Qualitative:** Visual comparison between ground truth (LiDAR) and predicted (Wi-Fi CSI-based) point clouds.
- **Quantitative:**
    - **ICP Fitness:** Measures the proportion of aligned points between predicted and ground truth clouds.
    - **ICP RMSE:** Calculates the average Euclidean distance between corresponding points.

### **Highlights**

- **Subject-Based Split Protocol:**
    - **ICP Fitness:** 0.6358
    - **ICP RMSE:** 0.0103m
- **Room-Based Split Protocol:**
    - **ICP Fitness:** 0.6127
    - **ICP RMSE:** 0.0104m
- Model achieved fast inference times (2.31ms on GPU, 137.27ms on CPU).

### **Visual Results**

- Ground truth and predicted point clouds demonstrated the ability to capture human presence, room geometry, and furniture, although predicted clouds had lower density and resolution in critical areas.

---

### **Analysis**

### **Strengths**

1. **Novel Architecture:** The transformer-based approach effectively captures complex spatio-temporal patterns in CSI data.
2. **Generalization:** Demonstrated adaptability across unseen subjects and environments.
3. **Multi-Modal Dataset:** Validation using MM-Fi’s diverse data ensures robustness for real-world scenarios.
4. **Joint Communication and Sensing (JC&S):** Paves the way for environmental awareness in 5G/6G networks.

### **Weaknesses/Gaps**

1. **Point Density:** Predicted point clouds had lower resolution than ground truth, limiting fine-grained spatial representation.
2. **Vertical Angle and Depth:** Limitations in depth estimation and vertical coverage due to Wi-Fi transmitter-receiver proximity.
3. **No Multi-Modal Fusion:** Did not incorporate complementary modalities (e.g., LiDAR) to enhance accuracy.

### **Opportunities for Improvement**

1. Improve point density and resolution with enhanced transformer layers or additional attention mechanisms.
2. Extend to **multi-modal sensor fusion** to combine Wi-Fi CSI with other sensing modalities (e.g., RGB-D, mmWave radar).
3. Address depth and vertical angle limitations by optimizing CSI data collection configurations.

---

### **Relevance to the Project**

### **Insights or Ideas**

1. **Transformer-Based Feature Extraction:** The multi-head attention mechanism could be adapted for HAR tasks to capture spatio-temporal dependencies.
2. **Positional Encoding:** Embedding antenna and subcarrier positions can enhance spatial representation in HAR models.
3. **Loss Function:** The combined Chamfer Loss and Feature Transform Regularization may be useful for HAR tasks requiring geometric feature alignment.

### **Potential for Reuse**

- The MM-Fi dataset and the CSI2PC model architecture offer valuable baselines for exploring HAR using 3D spatial data.
- Temporal encoding and transformer-based spatio-temporal modeling are directly applicable to human activity recognition.

### **Unanswered Questions**

- How scalable is the approach for larger or more complex environments?
- Can the model handle multi-user scenarios with overlapping CSI signals?