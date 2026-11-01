from_pretrained transformers import AutoImageProcessor, AutoModel

model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"

processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
#garder le brightness_normalized dans input transform enlever le reste deja dans processor(resize+normalize)