import numpy as np
import torch
from transformers import Trainer
from torch import nn
from explanations.explanations_eval import eval_explanations
from explanations.explanations_preprocess import (
    create_embeddings,
    preprocess_explanation,
    get_token_weights,
)
from critic.nn.model import nn_model
from critic.catboost.model import catboost_model


class ExplanationsTrainer(Trainer):
    def __init__(self, critic_loss_weight, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_loss = nn.CrossEntropyLoss()
        self.critic_loss = nn.CrossEntropyLoss()
        self.tokenizer = tokenizer
        self.critic_loss_weight = critic_loss_weight

    def critic_predict(self, embeddings):
        predictions = np.argmax(
            np.average(
                np.asarray(
                    [
                        nn_model.predict(embeddings, verbose=0),
                        catboost_model.predict_proba(embeddings),
                    ]
                ),
                axis=0,
            ),
            axis=1,
        )
        return predictions

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        model.eval()
        explanations = eval_explanations(model, self.tokenizer, inputs)
        model.train()

        explanation_tokens_weights = []
        for explanation in explanations:
            explanation_tokens_weights.append(
                {
                    (key, weight)
                    for key, weight in preprocess_explanation(
                        get_token_weights(explanation)
                    )
                }
            )
        embeddings = np.array(create_embeddings(explanation_tokens_weights))
        critic_predictions = torch.Tensor(self.critic_predict(embeddings)).to(
            self.args.device
        )

        main_loss = self.main_loss(outputs.logits, labels.long())
        loss = (
            self.critic_loss_weight * self.critic_loss(critic_predictions, labels)
            + (1 - self.critic_loss_weight) * main_loss
        )
        return (loss, outputs) if return_outputs else loss
