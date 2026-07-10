"""
Stacked Denoising Autoencoder (SAE) for DEC initialization.
Architecture from the paper: d - 500 - 500 - 2000 - 10 (encoder),
mirrored decoder. Greedy layer-wise pretraining, then end-to-end finetuning.
"""

import torch
import torch.nn as nn


class StackedAutoencoder(nn.Module):
    def __init__(self, input_dim=784, embedding_dim=10):
        super().__init__()
        dims = [input_dim, 500, 500, 2000, embedding_dim]

        # ----- Encoder -----
        # ReLU on all layers EXCEPT the final embedding layer,
        # so the 10-d embedding can hold any values (incl. negatives).
        encoder_layers = []
        for i in range(len(dims) - 1):
            encoder_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        # ----- Decoder (mirror image) -----
        # ReLU on all layers EXCEPT the final reconstruction layer.
        rev = dims[::-1]
        decoder_layers = []
        for i in range(len(rev) - 1):
            decoder_layers.append(nn.Linear(rev[i], rev[i + 1]))
            if i < len(rev) - 2:
                decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)          # 784 -> 10
        x_hat = self.decoder(z)      # 10 -> 784
        return x_hat, z

    def embed(self, x):
        """Encoder only — used by DEC after pretraining."""
        return self.encoder(x)


def pretrain_layerwise(model, data_loader, device,
                       epochs_per_layer=25, dropout_rate=0.2, lr=1e-3):
    """
    Greedy layer-wise pretraining (paper Section 3.2).
    Trains each (encoder layer, decoder layer) pair as a small
    denoising autoencoder, freezing everything learned before it.
    """
    # Grab just the Linear layers from encoder/decoder
    enc_linears = [m for m in model.encoder if isinstance(m, nn.Linear)]
    dec_linears = [m for m in model.decoder if isinstance(m, nn.Linear)]

    n_pairs = len(enc_linears)
    mse = nn.MSELoss()
    drop = nn.Dropout(dropout_rate)

    for k in range(n_pairs):
        enc_k = enc_linears[k]
        dec_k = dec_linears[n_pairs - 1 - k]   # matching mirror layer
        optimizer = torch.optim.Adam(
            list(enc_k.parameters()) + list(dec_k.parameters()), lr=lr
        )

        print(f"--- Pretraining layer pair {k + 1}/{n_pairs} ---")
        for epoch in range(epochs_per_layer):
            total = 0.0
            for x, _ in data_loader:
                x = x.to(device)

                # 1) Push x through the already-trained encoder layers
                #    to get the *input* for the current pair (no grad).
                with torch.no_grad():
                    h = x
                    for j in range(k):
                        h = torch.relu(enc_linears[j](h))

                # 2) Denoise: corrupt input, reconstruct clean version
                h_noisy = drop(h)
                hidden = enc_k(h_noisy)
                if k < n_pairs - 1:          # embedding layer stays linear
                    hidden = torch.relu(hidden)
                out = dec_k(hidden)
                if k > 0:                     # reconstruction of a ReLU output
                    out = torch.relu(out)

                loss = mse(out, h)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()
            print(f"  epoch {epoch + 1}/{epochs_per_layer}  loss={total / len(data_loader):.4f}")


def finetune(model, data_loader, device, epochs=60, lr=1e-3):
    """
    End-to-end finetuning of the full autoencoder (no dropout),
    minimizing reconstruction loss. Paper Section 3.2.
    """
    mse = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("--- Finetuning full autoencoder ---")
    for epoch in range(epochs):
        total = 0.0
        for x, _ in data_loader:
            x = x.to(device)
            x_hat, _ = model(x)
            loss = mse(x_hat, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"  epoch {epoch + 1}/{epochs}  loss={total / len(data_loader):.4f}")