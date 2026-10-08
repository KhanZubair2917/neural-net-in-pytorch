# Neural Network in PyTorch

Rebuilding a from-scratch neural network in PyTorch to predict Titanic survival, and comparing it against the NumPy and XGBoost versions.

## Dataset

Same cleaned Titanic dataset as my earlier projects, 891 passengers, 11 features after cleaning and encoding.

## What I Did

- Scaled features with StandardScaler and converted data to PyTorch tensors
- Defined a network with nn.Module: 11 inputs, 4 hidden neurons, 1 output, sigmoid activation
- Trained with binary cross entropy loss and SGD for 1000 epochs
- Compared results against my NumPy from-scratch network and XGBoost

## Results

| Model | Accuracy |
|---|---|
| XGBoost | 80% |
| NumPy from scratch | 79% |
| PyTorch | 81% |

## Key Findings

All three approaches landed within two points of each other, so PyTorch didn't unlock better accuracy here. What it changed is how much code it takes: the manual backpropagation and gradient descent from the NumPy version collapsed into loss.backward() and optimizer.step(). Building the NumPy version first made it clear what those two lines automate.

## Tools Used

Python, PyTorch, pandas, scikit-learn (scaling and train/test split only)