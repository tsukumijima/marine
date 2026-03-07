import torch


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss_func = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=0)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is None:
            raise ValueError("mask is required for CrossEntropyLoss")

        loss = torch.zeros(1, device=logits.device)
        batch_size = logits.size(0)

        for logit, label, m in zip(logits, labels, mask):
            loss += self.loss_func(logit[m], label[m])

        return torch.sum(loss / batch_size)
