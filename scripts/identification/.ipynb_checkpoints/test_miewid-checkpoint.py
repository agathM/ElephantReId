from_pretrained transformers import AutoModel

model = AutoModel.from_pretrained(
    "conservationxlabs/miewid-msv3",
    trust_remote_code=True
)

#ajouter du padding -> 440*440 ou resize si trop grand