from trl import SFTTrainer
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from peft import PeftModel

class ForwardKLMaskedSFTTrainer(SFTTrainer):
    def __init__(self, *args, use_kl=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_kl = use_kl

        # Initialize metrics for KL and CE losses
        if not hasattr(self, "_metrics"):
            self._metrics = {"train": {}, "eval": {}}
        self._metrics["train"]["kl_loss"] = []
        self._metrics["eval"]["kl_loss"] = []

        if use_kl:
            if isinstance(self.model, PeftModel):
                self.is_lora = True
            else:
                self.is_lora = False
                unwrapped_model = self.model.module if hasattr(self.model, "module") else self.model
                config = unwrapped_model.config
                self.ref_model = AutoModelForCausalLM.from_config(config)

                state_dict = unwrapped_model.state_dict()
                self.ref_model.load_state_dict(state_dict)
                self.ref_model = self.ref_model.to(self.model.device)
                self.ref_model.eval()
        

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None
    ):
        mode = "train" if model.training else "eval"
        forward_kl_mask = inputs.pop("forward_kl_mask", None)

        # If no mask provided, use standard CE loss for all examples
        if forward_kl_mask is None or not self.use_kl:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        # Split examples based on mask
        kl_mask = (forward_kl_mask == 1)
        ce_mask = (forward_kl_mask != 1)
        num_kl = torch.sum(kl_mask.long())
        num_ce = torch.sum(ce_mask.long())

        total_loss = torch.tensor(0.0, device=model.device)
        outputs = None
        kl_loss_value = 0.0

        # Compute CE loss for non-KL examples (forward_kl_mask != 1)
        if num_ce > 0:
            ce_inputs = {k: v[ce_mask] for k, v in inputs.items()}

            # Compute CE loss directly without calling super() to avoid collective ops
            ce_outputs = model(**ce_inputs, use_cache=False)
            ce_logits = ce_outputs.logits
            ce_labels = ce_inputs["labels"]

            # Shift logits and labels for causal LM
            shift_logits = ce_logits[..., :-1, :].contiguous()
            shift_labels = ce_labels[..., 1:].contiguous()

            # Compute cross entropy loss
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits_flat = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels_flat = shift_labels.view(-1)
            ce_loss = loss_fct(shift_logits_flat, shift_labels_flat)

            total_loss = total_loss + ce_loss * num_ce

            outputs = ce_outputs

        # Compute KL loss for KL examples (forward_kl_mask == 1)
        if num_kl > 0:
            kl_inputs = {k: v[kl_mask] for k, v in inputs.items()}

            # Get student outputs
            student_outputs = model(**kl_inputs, use_cache=False)

            # Get teacher outputs
            if self.is_lora:
                unwrapped_model = model.module if hasattr(model, "module") else model
                try:
                    with torch.no_grad():
                        unwrapped_model.disable_adapter_layers()
                        # Temporarily disable gradient checkpointing for reference pass
                        is_grad_ckpt_enabled = unwrapped_model.is_gradient_checkpointing
                        if is_grad_ckpt_enabled:
                            unwrapped_model.gradient_checkpointing_disable()
                        ref_outputs = unwrapped_model(**kl_inputs, use_cache=False)
                        if is_grad_ckpt_enabled:
                            unwrapped_model.gradient_checkpointing_enable()
                finally:
                    unwrapped_model.enable_adapter_layers()
            else:
                with torch.no_grad():
                    ref_outputs = self.ref_model(**kl_inputs, use_cache=False)

            # Extract and shift logits
            teacher_logits = ref_outputs.logits[:, :-1, :].contiguous()
            student_logits = student_outputs.logits[:, :-1, :].contiguous()

            # Free memory from outputs objects
            del ref_outputs

            shift_labels = kl_inputs["labels"][:, 1:].contiguous()

            # Only compute KL for non-padding tokens to save memory
            token_mask = (shift_labels != -100)

            if token_mask.any():
                # Flatten and select only valid tokens
                teacher_logits_flat = teacher_logits.view(-1, teacher_logits.size(-1))
                student_logits_flat = student_logits.view(-1, student_logits.size(-1))
                token_mask_flat = token_mask.view(-1)

                # Free memory from full logits
                del teacher_logits, student_logits

                # Select only non-padding tokens
                teacher_logits_valid = teacher_logits_flat[token_mask_flat]
                student_logits_valid = student_logits_flat[token_mask_flat]

                # Free memory from flat logits
                del teacher_logits_flat, student_logits_flat, token_mask_flat

                # Compute KL in chunks to save memory
                num_valid_tokens = teacher_logits_valid.size(0)
                chunk_size = 1024  # Process 1024 tokens at a time
                kl_sum = torch.tensor(0.0, device=model.device)

                for i in range(0, num_valid_tokens, chunk_size):
                    end_idx = min(i + chunk_size, num_valid_tokens)

                    # Get chunk
                    teacher_chunk = teacher_logits_valid[i:end_idx]
                    student_chunk = student_logits_valid[i:end_idx]

                    # Compute log probs for chunk
                    teacher_logprobs_chunk = F.log_softmax(teacher_chunk, dim=-1)
                    student_logprobs_chunk = F.log_softmax(student_chunk, dim=-1)

                    # Free chunk logits
                    del teacher_chunk, student_chunk

                    # Compute KL for chunk (sum over vocab, then sum over tokens in chunk)
                    kl_chunk = F.kl_div(
                        student_logprobs_chunk,
                        teacher_logprobs_chunk,
                        reduction="sum",
                        log_target=True
                    )

                    # Free chunk log probs
                    del teacher_logprobs_chunk, student_logprobs_chunk

                    # Accumulate (keeps gradient)
                    kl_sum = kl_sum + kl_chunk
                    del kl_chunk

                # Free full logits
                del teacher_logits_valid, student_logits_valid

                # Average over valid tokens
                kl_loss = kl_sum / num_valid_tokens
                kl_loss_value = kl_loss.item()
                total_loss = total_loss + kl_loss * num_kl

                del kl_loss

            if outputs is None:
                outputs = student_outputs

        # Normalize by total number of examples
        total_samples = num_ce + num_kl
        if total_samples > 0:
            total_loss = total_loss / total_samples

        # Gather metrics across all processes
        kl_loss_tensor = torch.tensor([kl_loss_value], device=model.device)
        kl_loss_gathered = self.accelerator.gather_for_metrics(kl_loss_tensor).mean().item()
        
        self._metrics[mode]["kl_loss"].append(kl_loss_gathered)

        return (total_loss, outputs) if return_outputs else total_loss
