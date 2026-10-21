import torch
import torch.nn as nn

from model.featureencoder import FeatureEncoder, FeatureEncoderLayer
from model.featureprojection import FeatureProjection
from model.positionalembedding import PositionalEmbedding
from model.transformerencoder import TransformerEncoder


class Wav2Vec2KWS(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.feature_encoder = FeatureEncoder(
            [
                FeatureEncoderLayer(512, 10, 5),
                FeatureEncoderLayer(512, 3, 2),
                FeatureEncoderLayer(512, 3, 2),
                FeatureEncoderLayer(512, 3, 2),
                FeatureEncoderLayer(512, 3, 2),
                FeatureEncoderLayer(512, 2, 2),
                FeatureEncoderLayer(512, 2, 2),
            ]
        )

        self.feature_projection = FeatureProjection(input_dim=512, hidden_dim=384)

        self.positional_embedding = PositionalEmbedding(
            dim=384, kernel_size=128, groups=16
        )

        self.transformer = TransformerEncoder(
            dim=384, num_layers=6, num_heads=6, ff_dim=384 * 4, dropout=0.1
        )

        self.classifier = nn.Linear(384, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_encoder(x)

        x = self.feature_projection(x)

        x = x + self.positional_embedding(x)

        x = self.transformer(x)

        x = x.mean(dim=1)

        logits = self.classifier(x)

        return logits
