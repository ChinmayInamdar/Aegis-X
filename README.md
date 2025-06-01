# Aegis-X+: Enhanced Self-Adaptive AI Framework for Real-Time Anomaly Detection

![Aegis-X+ Framework](images/aegisx_framework.png)

## Overview

Aegis-X+ is a self-adaptive cybersecurity framework integrating deep autoencoders, variational autoencoders (VAEs), and multi-agent reinforcement learning (MARL) to detect and respond to network intrusions in real time. It bridges the gap between passive detection and active defense, addressing challenges like class imbalance in network traffic using explainable AI techniques.

## Highlights

* **Detection:** Deep autoencoders and VAEs for unsupervised anomaly detection
* **Defense:** Multi-agent reinforcement learning for real-time adaptive network segmentation
* **Explainability:** SHAP (SHapley Additive exPlanations) to interpret model decisions
* **Dataset:** CICIDS2017 (2.8M records, 79 features)

## System Architecture

The Aegis-X+ framework comprises three main modules:

1. **Anomaly Detection:** Learns normal traffic patterns and flags deviations
2. **Segmentation via MARL:** Agents adaptively isolate affected network segments
3. **Explainability Layer:** SHAP values highlight feature contributions

![System Pipeline](images/system_pipeline.png)

## Data Preprocessing

* Removal of missing values (\~0.8%)
* Feature scaling and normalization
* One-hot encoding for categorical fields (e.g., protocol types)
* PCA used to reduce dimensions; 23 components retain 95% variance

## Model Details

### Autoencoder

* Symmetric encoder-decoder: 79 -> 64 -> 32 -> 16 -> 32 -> 64 -> 79
* Activation: ReLU, Optimizer: Adam
* BatchNorm + Dropout
* Loss: MSE and SHAP-weighted MSE
* Anomaly threshold: 95th percentile of validation MSEs

### Variational Autoencoder (VAE)

* Latent space with mean and log variance
* Reconstruction + KL Divergence loss
* Score = 0.3 \* MSE + 0.7 \* KL

### MARL for Network Segmentation

* Cooperative agents model network as graph (nodes=segments, edges=connections)
* Policy: Graph Convolutional Network (GCN)
* Reward function combines threat containment, service disruption, and action cost
* Centralized training with decentralized execution

## Explainability

* SHAP values from KernelSHAP
* Top features:

  * Average Packet Size (0.097)
  * Destination Port (0.079)
  * Total Length of Forward Packets (0.063)

## Evaluation Results

### Metrics

| Configuration           | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| ----------------------- | -------- | --------- | ------ | -------- | ------- | ------ |
| Autoencoder-only        | 80.23%   | 65.27%    | -      | 0.715    | 0.576   | 0.217  |
| Autoencoder + MARL      | 80.28%   | 65.12%    | -      | 0.715    | -       | -      |
| Full System (with SHAP) | 80.25%   | 65.12%    | -      | 0.715    | -       | -      |

* False Positives reduced by 30.8% in the full system
* Precision improved without sacrificing recall

## Visualizations

The following images offer deeper insights into the model’s performance and internal mechanics. Add them to the `images/` folder and embed them in your GitHub README as shown below.

Include the following result images in the `images/` folder:

* `confusion_matrix.png`: Confusion matrix of the autoencoder
* `roc_curve.png`: ROC curves for all models
* `precision_recall.png`: Precision-Recall curves
* `radar_chart.png`: Radar comparison of all configurations
* `segmentation_graph.png`: Learned network segmentation by MARL
* `shap_importance.png`: SHAP top 20 features

**Confusion Matrix:** Visualizes true/false positives and negatives, highlighting model accuracy.

![Confusion Matrix](images/confusion_matrix.png)

**ROC Curve:** Demonstrates the trade-off between true positive and false positive rates across thresholds.

![ROC Curve](images/roc_curve.png)

**Precision-Recall Curve:** Captures the balance between precision and recall under class imbalance.

![Precision Recall Curve](images/precision_recall.png)

**Radar Chart:** Compares key metrics (accuracy, F1-score, etc.) across system configurations.

![Radar Chart](images/radar_chart.png)

**Network Segmentation Graph:** Shows MARL-learned optimal defense topology with protected zones.

![Network Segmentation](images/segmentation_graph.png)

**SHAP Feature Importance:** Highlights the most impactful features for anomaly detection decisions.

![SHAP Importance](images/shap_importance.png)

Together, these visualizations reinforce the utility of combining detection, adaptive defense, and interpretability for robust cybersecurity systems.

## Limitations

* Low recall due to class imbalance
* Reconstruction-based scoring limits discriminative power
* High computational cost (training on GPU cluster)
* Static models prone to concept drift; periodic retraining needed

## Future Work

* Use LSTM-based autoencoders for temporal attack patterns
* Support dynamic topologies (cloud/SaaS systems)
* Transfer learning to new network environments
* Adversarial training for robustness

## Getting Started

```bash
git clone https://github.com/yourusername/aegisx-plus.git
cd aegisx-plus
pip install -r requirements.txt
```

### Run Steps

```bash
python preprocessing.py
python train_autoencoder.py
python train_vae.py  # optional
python train_marl.py
python run_aegisx.py
python explainability.py
```

## Authors

* Chinmay Inamdar – [chinmay.inamdar22@vit.edu](mailto:chinmay.inamdar22@vit.edu)
* Arya Doshi, Vijay Mane, Tanmay Gote – VIT Pune

## License

MIT License

## Citation

```bibtex
@article{inamdar2025aegisx,
  title={Aegis-X+: An Enhanced Self-Adaptive AI Framework for Real-Time Anomaly Detection},
  author={Inamdar, Chinmay and Kakkar, Preetish and Doshi, Arya and Mane, Vijay and Gote, Tanmay},
  year={2025},
  institution={Vishwakarma Institute of Technology},
  publisher={IEEE}
}
```
