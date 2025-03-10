# DeepMIMOv2-CNN-Channel-Estimation

This project presents a deep learning-based approach to channel estimation in mmWave and mMIMO systems using realistic simulation data from the DeepMIMO v2 dataset. By leveraging advanced CNN architectures, the system estimates channel responses from noisy observations, providing a robust solution for challenging propagation environments.

**Overview:**

**Realistic Data Integration:**
The project utilizes the DeepMIMO v2 dataset to generate realistic channel data. Complex channel coefficients are converted to amplitude, scaled appropriately, and corrupted with additive white Gaussian noise (AWGN) to mimic real-world conditions. A representative 1D channel vector is then used for training.

**Model Architecture:**
A 1D CNN-based channel estimator is implemented with five convolutional layers followed by a fully connected layer. The network maps noisy channel observations (input) to estimated channel responses (output) over a set of subcarriers, demonstrating the viability of deep learning in the channel estimation task.

**Training & Evaluation:**
The model is trained on 80% of the UE channels and tested on the remaining 20%. Performance is tracked via training loss curves, and the model’s predictions are visually compared against the ground truth for several test samples, ensuring that the network generalizes well to unseen data.

Clone the repository and run the project using:
> python cnn_mimo.py

**The script will:**
1) Load and process DeepMIMO v2 data.
2) Train the CNN-based channel estimator.
3) Plot the training loss curve.
4) Visualize the true versus estimated channel responses for test samples.

**Future Work:**
1) Physics-Informed Loss Enhancement
2) Full 2D Channel Estimation
3) Hyperparameter Optimization
