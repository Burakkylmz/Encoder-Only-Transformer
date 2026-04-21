from __future__ import annotations

from encoder_only_transformer.config.config import ModelConfig, ProjectConfig
from encoder_only_transformer.layers.pooling import BasePooling
from encoder_only_transformer.models.model import EncoderForSequenceClassification


class ModelFactory:
    """
    Factory for constructing encoder-only models from configuration objects.
    """

    @staticmethod
    def create_sequence_classification_model(
        model_config: ModelConfig,
        num_classes: int,
        pooling: BasePooling | None = None,
    ) -> EncoderForSequenceClassification:
        if not isinstance(model_config, ModelConfig):
            raise TypeError("model_config must be an instance of ModelConfig.")

        return EncoderForSequenceClassification(
            config=model_config,
            num_classes=num_classes,
            pooling=pooling,
        )

    @staticmethod
    def create_sequence_classification_model_from_project_config(
        project_config: ProjectConfig,
        num_classes: int,
        pooling: BasePooling | None = None,
    ) -> EncoderForSequenceClassification:
        if not isinstance(project_config, ProjectConfig):
            raise TypeError("project_config must be an instance of ProjectConfig.")

        return ModelFactory.create_sequence_classification_model(
            model_config=project_config.model,
            num_classes=num_classes,
            pooling=pooling,
        )