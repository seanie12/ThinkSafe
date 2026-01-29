import torch
from dataclasses import dataclass
from typing import Any, Dict, List
from transformers import DataCollatorForLanguageModeling


@dataclass
class DataCollatorForKLTraining(DataCollatorForLanguageModeling):
    """
    Data collator that handles the forward_kl_mask field in addition to standard fields.

    This collator extends the standard DataCollatorForLanguageModeling to properly
    batch the forward_kl_mask tensor used in KL divergence training.
    """

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Extract forward_kl_mask if present before calling parent
        forward_kl_masks = None
        if features and "forward_kl_mask" in features[0]:
            forward_kl_masks = [f.pop("forward_kl_mask") for f in features]

        # Call parent to handle standard fields (input_ids, labels, attention_mask)
        batch = super().__call__(features)

        # Add forward_kl_mask back to batch as a tensor
        if forward_kl_masks is not None:
            batch["forward_kl_mask"] = torch.tensor(forward_kl_masks, dtype=torch.long)

        return batch
