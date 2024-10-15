import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from our_model import GrnTransformer
from sklearn.metrics import roc_auc_score, average_precision_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluation(y_pred, y_true, mask):
    # Detach tensors and move to CPU
    y_pred = y_pred.detach().cpu()
    y_true = y_true.detach().cpu()
    mask = mask.detach().cpu()

    # Flatten the tensors
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    mask = mask.view(-1)

    # Apply mask to select labeled pairs
    y_pred = y_pred[mask > 0]
    y_true = y_true[mask > 0]

    # Convert to numpy arrays
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()

    try:
        # Compute evaluation metrics
        auroc = roc_auc_score(y_true=y_true, y_score=y_pred)
        auprc = average_precision_score(y_true=y_true, y_score=y_pred)
    except Exception as e:
        auroc = 0
        auprc = 0

    return auroc, auprc


def train(model, dataloader, loss_func, optimizer, epoch):
    model.train()
    total_loss = 0

    for idx, (expr_embedding, label_matrix, mask) in enumerate(dataloader):
        # Move tensors to the appropriate device
        expr_embedding = expr_embedding.to(torch.float32).to(
            device
        )  # [batch_size, num_genes, embedding_dim]
        label_matrix = label_matrix.to(torch.float32).to(
            device
        )  # [batch_size, num_genes, num_genes]
        mask = mask.to(torch.float32).to(device)  # [batch_size, num_genes, num_genes]

        optimizer.zero_grad()
        predicted_output = model(
            expr_embedding
        )  # Should output [batch_size, num_genes, num_genes]
        loss = loss_func(predicted_output, label_matrix)
        masked_loss = loss * mask
        final_loss = masked_loss.sum() / mask.sum()
        total_loss += final_loss.item()
        final_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

    # there's only one batch so we can just use the last one
    return total_loss, predicted_output, label_matrix, mask


def validate(model, dataloader, loss_func):
    model.eval()
    total_loss = 0
    pre_list = []
    lb_list = []
    mask_list = []

    with torch.no_grad():
        for idx, (expr_embedding, label_matrix, mask) in enumerate(dataloader):
            expr_embedding = expr_embedding.to(torch.float32).to(
                device
            )  # [batch_size, num_genes, embedding_dim]
            label_matrix = label_matrix.to(torch.float32).to(
                device
            )  # [batch_size, num_genes, num_genes]
            mask = mask.to(torch.float32).to(
                device
            )  # [batch_size, num_genes, num_genes]

            # Forward pass
            predicted_output = model(
                expr_embedding
            )  # Should output [batch_size, num_genes, num_genes]

            # Compute loss (optional)
            loss = loss_func(predicted_output, label_matrix)
            masked_loss = loss * mask
            final_loss = masked_loss.sum() / mask.sum()
            total_loss += final_loss.item()

            # Collect predictions and labels
            pre_list.append(predicted_output.detach().cpu())
            lb_list.append(label_matrix.detach().cpu())
            mask_list.append(mask.detach().cpu())

        # Concatenate all batches
        y_pred = torch.cat(pre_list, dim=0)  # [total_samples, num_genes, num_genes]
        y_true = torch.cat(lb_list, dim=0)
        mask = torch.cat(mask_list, dim=0)

        # Flatten tensors
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        mask = mask.view(-1)

        # Apply mask to select labeled pairs
        y_pred = y_pred[mask > 0]
        y_true = y_true[mask > 0]

        # Convert to numpy arrays
        y_pred = y_pred.numpy()
        y_true = y_true.numpy()

        try:
            # Compute evaluation metrics
            AUROC = roc_auc_score(y_true=y_true, y_score=y_pred)
            AUPRC = average_precision_score(y_true=y_true, y_score=y_pred)
        except Exception as e:
            AUROC = 0
            AUPRC = 0

    return AUROC, AUPRC


class GeneInteractionDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, expression_data):
        # Load labeled gene pairs
        data = pd.read_csv(data_path, index_col=0, header=0)

        self.labeled_pairs = data[["TF", "Target"]].values
        self.labels = torch.tensor(
            data["Label"].values.astype(np.float32)
        )  # Convert labels to float32
        self.expression_data = torch.tensor(
            expression_data, dtype=torch.float32
        )  # [num_genes, embedding_dim]
        self.num_genes = self.expression_data.shape[0]

    def __len__(self):
        return 1  # Since the expression data is the same, we can return a single sample

    def __getitem__(self, idx):
        # Gene embeddings
        expr_embedding = self.expression_data  # [num_genes, embedding_dim]

        # Initialize label matrix (all -1) and mask (all 0)
        label_matrix = -torch.ones(
            (self.num_genes, self.num_genes), dtype=torch.float32
        )
        mask = torch.zeros((self.num_genes, self.num_genes), dtype=torch.float32)

        # Fill in the labels and mask for labeled pairs
        for (i, j), label in zip(self.labeled_pairs, self.labels):
            label_matrix[i, j] = label  # Assign the label (1 or 0)
            mask[i, j] = 1.0  # Mark as labeled

        return expr_embedding, label_matrix, mask


def our_main(data_dir, args):
    # data_dir = 'hESC500'
    expression_data_path = data_dir + "/BL--ExpressionData.csv"
    biovect_e_path = data_dir + "/biovect768.npy"
    train_data_path = data_dir + "/Train_set.csv"
    val_data_path = data_dir + "/Validation_set.csv"
    test_data_path = data_dir + "/Test_set.csv"
    expression_data = np.array(pd.read_csv(expression_data_path, index_col=0, header=0))

    # Data Preprocessing
    standard = StandardScaler()
    scaled_df = standard.fit_transform(expression_data.T)
    expression_data = scaled_df.T
    expression_data_shape = expression_data.shape

    # Model parameters
    batch_size = args.batch_size
    embed_size = args.embed_size
    num_layers = args.num_layers
    num_head = args.num_head
    lr = args.lr
    epochs = args.epochs
    step_size = args.step_size
    gamma = args.gamma
    global schedulerflag
    schedulerflag = args.scheduler_flag
    print(
        f"""hyperparameters:
          
          
          batch_size: {batch_size}
          embed_size: {embed_size}
          num_layers: {num_layers}
          num_head: {num_head}
          lr: {lr}
          epochs: {epochs}
          step_size: {step_size}
          gamma: {gamma}
          scheduler_flag: {schedulerflag}
          """
    )

    ########## ABOVE IS THE SAME AS THE ORIGINAL `main` ##########
    interaction_train_ds = GeneInteractionDataset(train_data_path, expression_data)
    interaction_val_ds = GeneInteractionDataset(val_data_path, expression_data)
    interaction_test_ds = GeneInteractionDataset(test_data_path, expression_data)

    interaction_train_loader = torch.utils.data.DataLoader(
        dataset=interaction_train_ds, batch_size=1, shuffle=True, num_workers=0
    )
    interaction_val_loader = torch.utils.data.DataLoader(
        dataset=interaction_val_ds, batch_size=1, shuffle=False, num_workers=0
    )
    interaction_test_loader = torch.utils.data.DataLoader(
        dataset=interaction_test_ds, batch_size=1, shuffle=False, num_workers=0
    )

    model = GrnTransformer(
        num_genes=interaction_train_ds.expression_data.shape[0],
        embed_dim=768,
        num_heads=num_head,
        hidden_dim=768,
        num_layers=num_layers,
        group_size=512,
        dropout=0.1,
    ).to(device)

    print(f"Model: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    num_positive_samples = interaction_train_ds.labels.sum()
    num_negative_samples = len(interaction_train_ds.labels) - num_positive_samples
    positive_weight = torch.tensor(
        [num_negative_samples / num_positive_samples], device=device
    )
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight, reduction="none")

    for epoch in range(1, epochs + 1):
        total_loss, predicted_output, label_matrix, mask = train(
            model, interaction_train_loader, loss_fn, optimizer, epoch
        )

        # only validate at the end of every 10 epochs
        if epoch % 50 == 0:
            auc_train, aupr_train = evaluation(predicted_output, label_matrix, mask)
            print(
                f"| end of epoch {epoch} | train AUROC {auc_train} | train AUPRC {aupr_train} | total_loss {total_loss}"
            )

            auc_val, aupr_val = validate(model, interaction_val_loader, loss_fn)
            print("-" * 100)
            print(
                "| end of epoch {:3d} | valid AUROC {:8.3f} | valid AUPRC {:8.3f}".format(
                    epoch, auc_val, aupr_val
                )
            )
            print("-" * 100)
            auc_test, aupr_test = validate(model, interaction_test_loader, loss_fn)
            print(
                "| end of epoch {:3d} | test  AUROC {:8.3f} | test  AUPRC {:8.3f}".format(
                    epoch, auc_test, aupr_test
                )
            )
            print("-" * 100)
