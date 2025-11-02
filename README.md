# Fuzzy Logic Classification with FCM + ANFIS

This repository implements a **fuzzy logic-based classification system** using **Fuzzy C-Means (FCM) clustering** and **Adaptive Neuro-Fuzzy Inference System (ANFIS)** training. The pipeline supports **multi-label outputs**, evaluates performance with standard metrics, and visualizes membership functions.

---

## Project Structure

| File / Section       | Description |
|----------------------|-------------|
| `calculate_metrics`  | Computes success rate, MSE, R², and EVS |
| `evaluate_algorithm` | Evaluates a trained FIS on validation/test sets |
| `evaluate_overall`   | Aggregates overall performance across labels |
| `fuzzycm`            | Generates initial FIS using FCM clustering |
| `train_neuron`       | Tunes the FIS using ANFIS optimization |
| `initGen`            | Orchestrates training and evaluation for all labels |
| **Main Script**      | Loads dataset, splits into train/val/test, runs training, saves results, and plots membership functions |

---

## How It Works

1. **Data Preparation**
   - Dataset is read from CSV.
   - Split into training, validation, and testing sets.

2. **FIS Generation**
   - For each label, a fuzzy inference system is created using **FCM clustering**.

3. **Training**
   - The FIS is tuned using **ANFIS** for a specified number of epochs.

4. **Evaluation**
   - Predictions are generated for validation and test sets.
   - Metrics are computed:  
     - Success rate  
     - Mean Squared Error (MSE)  
     - R² (Coefficient of Determination)  
     - Explained Variance Score (EVS)

5. **Visualization**
   - Membership functions for up to 25 inputs are plotted and saved.

---

## Example Usage

```matlab
% Constants
LABELS   = 6;
EPOCHS   = 5;
CLUSTERS = 3;
DATA_PATH = 'D:\Documents\MATLAB\Fuzzy_Projects\vran\vranf.csv';

% Load dataset
data = readmatrix(DATA_PATH);

% Run pipeline
[gralOut, learnOut, resul_data] = initGen(EPOCHS, CLUSTERS, resul_data);

% Display results
disp(resul_data.results);
```

---

## Output

After execution, you’ll get:

- **Results Table** (`resul_data.results`) with per-label and overall metrics.
- **CSV files**:
  - `Y_test_pred.csv` → Predicted test outputs
  - `Y_val_pred.csv` → Predicted validation outputs
  - `Y_test.csv` → Ground truth test labels
  - `Y_val.csv` → Ground truth validation labels
- **Membership function plots** saved as `membership_functions.eps`.

---

## Requirements

- MATLAB R2021a or later
- Fuzzy Logic Toolbox
- Statistics and Machine Learning Toolbox

---

## Data Format

- Input dataset: CSV file
- Columns:
  - First `features` columns → input features
  - Last `labels` columns → target outputs

Example: If you have 20 features and 6 labels, your dataset should be `[X1 ... X20 | Y1 ... Y6]`.

---

## Customization

- `LABELS`: Number of output labels
- `EPOCHS`: Training epochs for ANFIS
- `CLUSTERS`: Number of clusters for FCM
- `DATA_PATH`: Path to your dataset

---

## Metrics Explained

- **Success Rate**: Fraction of predictions within ±0.5 of true labels
- **MSE**: Mean Squared Error
- **R²**: Coefficient of Determination (goodness of fit)
- **EVS**: Explained Variance Score

---

## Contributing

Contributions are welcome!  
- Fork the repo  
- Create a feature branch  
- Submit a pull request  

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## FAQ

**Q: Can I use this for regression tasks?**  
A: Yes, the system outputs continuous predictions. Success rate is thresholded, but you can adapt metrics for pure regression.

**Q: What if my dataset has fewer than 25 inputs?**  
A: The plotting loop automatically adjusts to the number of available inputs.

**Q: Do I need the exporters (`qfiscgen`, `translator`)?**  
A: Not in this minimal version. They were removed for clarity.

---
