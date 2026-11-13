# Hierarchical Multi-label Book Genres Classification

Hierarchical multi-label book classification using each book's description. This project is an **advanced continuation and evolution** of the [Book Genres Classification project](https://github.com/An1-sxd/Multi-label-Book-Genres-Classification-CNN-RNN-LSTM-GRU). While the earlier project reduced the dataset to **top-level** genres, this project treats the **hierarchical** label structure directly.

## Project contents

```text
Hierarchical Multi-label Classification/
├── bgc_train.csv
├── bgc_dev.csv
├── bgc_test.csv
├── train.ipynb
└── README.md
```

## Dataset

The data was obtained from the [Blurb Genre Collection](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/blurb-genre-collection.html) provided by the University of Hamburg.

The dataset contains book metadata with the following fields:

- `Number`: book identifier
- `Title`: book title
- `Body`: book description
- `Topics` / `Labels`: hierarchical labels separated by `>`

The notebook uses the following predefined splits:

| Split | Examples |
|---|---:|
| Training | 58,715 |
| Validation (`bgc_dev.csv`) | 14,785 |
| Test | 18,394 |
| **Total** | **91,894** |

The test file calls the label column `Labels`, so it is renamed to `Topics` before the datasets are combined.

## Objective

The original labels contain hierarchical paths, for example:

```text
Fiction>Mystery & Suspense>Crime Mysteries
```

Unlike the [Book Genres Classification project](../Book%20Genres%20Classification/README.md), which retained only the first level, this project treats the hierarchy as a multilabel target. Labels are split using both `>` and `,`: `>` separates the different levels of a hierarchy, while `,` separates multiple labels. All resulting parent and child categories are included in the target. This results in 147 label classes, ranging from broad parent categories to fine-grained child categories.

The task is multilabel classification because a book can have multiple labels, and one hierarchical path contributes multiple related labels to the target vector.

## Workflow

The notebook implements the following pipeline:

1. Mount Google Drive in Google Colab.
2. Load the training, validation, and test CSV files with pandas.
3. Standardize the test label-column name.
4. Tokenize each book description using whitespace splitting.
5. Split labels at `>` between hierarchy levels and at `,` between multiple labels, retaining the resulting parent and child categories.
6. Convert the hierarchical multilabel targets into binary vectors using `MultiLabelBinarizer`.
7. Inspect label frequencies and description lengths.
8. Build a vocabulary from the training descriptions only.
9. Keep the 50,000 most frequent training tokens and add:
   - `<PAD>` → `0`
   - `<UNK>` → `1`
10. Encode tokens as integer indices, mapping unseen words to `<UNK>`.
11. Pad batches dynamically and truncate CNN sequences to a maximum of 500 tokens.
12. Create a custom `BGCDataset` and PyTorch `DataLoader` objects.
13. Train CNN, vanilla RNN, LSTM, and GRU models.
14. Save the model state with the lowest validation loss.
15. Evaluate the best checkpoint on the held-out test set using multilabel metrics.

## Models

### CNN

The CNN uses:

- A learned embedding layer with dimension `100`
- Three parallel one-dimensional convolutions with kernel sizes `2`, `3`, and `4`
- ReLU activations
- Global max pooling for each convolution
- Concatenation of the pooled representations
- Dropout with probability `0.3`
- A fully connected multilabel output layer

### Vanilla RNN

The recurrent baseline uses:

- A learned embedding layer
- A one-layer `nn.RNN`
- Hidden size `128`
- Dropout with probability `0.3`
- A fully connected output layer

### LSTM and GRU

The LSTM and GRU use the same general recurrent configuration with hidden size `128`. For all recurrent models, `pack_padded_sequence` is used with the real sequence lengths so padded tokens are not processed as meaningful input.

## Training configuration

- Embedding dimension: `100`
- Maximum CNN sequence length: `500`
- Batch size: `128`
- Optimizer: Adam
- Learning rate: `0.001`
- Weight decay: `1e-4`
- Dropout: `0.3`
- Loss: `BCEWithLogitsLoss`
- Device: CUDA (`torch.device("cuda")`)
- CNN, LSTM, and GRU epochs: `30`
- Vanilla RNN epochs: `20`

## Handling class imbalance

The 147 hierarchical labels are highly imbalanced, particularly among fine-grained child categories. Per-label positive weights are calculated from the training split using:

```python
pos_weight = sqrt(negative_examples / positive_examples)
```

These weights are passed to `BCEWithLogitsLoss` for every model. The checkpoint with the lowest validation loss is restored before test evaluation.

## Results

The following results are recorded in the notebook after evaluating the best validation-loss checkpoint on the test set. Accuracy and F1 scores are multilabel metrics from `torchmetrics`.

| Model | Test loss | Accuracy | Macro F1 | Micro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| CNN | 0.1589 | **0.9804** | **0.3423** | 0.5767 | **0.5894** |
| Vanilla RNN | 0.2872 | 0.9784 | 0.0519 | 0.2845 | 0.2078 |
| LSTM | 0.1644 | 0.9703 | 0.2658 | 0.4906 | 0.5604 |
| GRU | 0.1693 | 0.9716 | 0.2613 | **0.4998** | 0.5492 |

The CNN achieved the strongest overall results in this experiment, with the highest accuracy, macro F1, and weighted F1. The GRU achieved the highest micro F1. The vanilla RNN had the weakest F1 scores, which reflects the difficulty of learning many sparse hierarchical labels with a basic recurrent architecture.

The notebook also computes multilabel confusion matrices for the evaluated labels.

## Limitations and Possible Improvements

The classification task involves a **large** number of hierarchical labels, including many fine-grained and **imbalanced** classes. Although the model can predict specific **child** categories, it may sometimes **fail** to predict their **corresponding parent** categories, resulting in hierarchical inconsistencies.

A possible improvement is to apply **hierarchy enforcement** during post-processing. This would ensure that whenever a child category is predicted, all its parent categories are also included in the final predictions.

## Running the notebook

### Google Colab

1. Upload `train.ipynb` and the three CSV files to Google Drive.
2. Update the path in the data-loading cell if necessary.
3. Run the notebook from top to bottom.
4. Select a CUDA runtime because the notebook explicitly selects a CUDA device.

### Local Jupyter environment

Install the required packages:

```bash
pip install numpy pandas matplotlib scikit-learn torch torchmetrics jupyter
```

Replace the Google Drive paths with local paths, then launch the notebook:

```bash
jupyter notebook train.ipynb
```

## Important implementation details

- The vocabulary is built from training descriptions only, preventing vocabulary leakage from validation and test data.
- Unknown validation and test words are mapped to `<UNK>`.
- `<PAD>` is used by the embedding layer as its padding index.
- CNN batches are padded to the longest sequence in each batch after truncation to 500 tokens.
- Recurrent batches retain sequence lengths and use packed sequences.
- `BCEWithLogitsLoss` is used because this is a multilabel classification problem.
- Parent labels are included in the target vector during training, but predictions are not currently post-processed to enforce parent-child consistency.
- Validation loss, not training loss, determines which model checkpoint is evaluated.
