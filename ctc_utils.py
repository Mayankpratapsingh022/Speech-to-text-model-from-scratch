import torch
import torch.nn as nn

def run_loss_function(log_probs: torch.Tensor, target: torch.Tensor, blank_token: int) -> torch.Tensor:
    """
    Computes CTC Loss.
    
    Args:
        log_probs: (batch, T, vocab) - Log-probabilities from the model.
        target: (batch, U) - Ground truth token IDs (padded).
        blank_token: Token ID used for the CTC blank.
    
    Returns:
        loss: Scalar CTC loss.
    """
    # zero_infinity=True prevents crashes if target length > input length for some samples (though we pad audio to handle this somewhat)
    loss_function = nn.CTCLoss(blank=blank_token, zero_infinity=True)

    # input lengths: all time steps used for each example (assumes no padding on input side or model handles it via mask, 
    # but for CTC usually we pass the full sequence length output by the model)
    input_lengths = tuple(log_probs.shape[1] for _ in range(log_probs.shape[0]))

    # target lengths: compute actual length by ignoring padding (assuming padding is same as blank or 0 or handled)
    # The snippet used (target != blank_token).sum(). 
    # In dataset.py, we padded with 0. 
    # WE MUST ENSURE BLANK_TOKEN matches the padding used or we manually check padding.
    # Usually <pad> is 0 or a specific token.
    # If the user uses <pad> as blank, then target != blank finds non-padding items.
    # If <pad> != <blank>, we should count non-padding explicitly.
    # For now, following snippet logic assuming blank_token is essentially the ignore/pad token too for length calc.
    target_lengths = (target != blank_token).sum(dim=1)
    target_lengths = tuple(t.item() for t in target_lengths)

    # CTC expects (T, batch, vocab)
    input_seq_first = log_probs.permute(1, 0, 2)
    loss = loss_function(input_seq_first, target, input_lengths, target_lengths)
    return loss

def decode_ctc_output(log_probs_batch, tokenizer, blank_token: int):
    """
    Greedy CTC decoding.

    Args:
        log_probs_batch: (B, T, V)  log-probs after log_softmax
        tokenizer: tokenizer with .decode(ids) method
        blank_token: ID used for CTC blank

    Returns:
        list[str]: Decoded transcripts
    """
    # Argmax over vocab to get most probable token at each timestep
    pred_ids = log_probs_batch.argmax(dim=-1)  # (B, T)

    decoded_texts = []

    for seq in pred_ids:  # seq: (T,)
        prev = blank_token
        ids = []

        for t in seq.tolist():
            # CTC rule: remove blanks, collapse repeats
            if t != blank_token and t != prev:
                ids.append(t)
            prev = t

        # Decode token IDs to text
        if len(ids) == 0:
            decoded_texts.append("")  # empty if nothing decoded
        else:
            decoded_texts.append(tokenizer.decode(ids))

    return decoded_texts
