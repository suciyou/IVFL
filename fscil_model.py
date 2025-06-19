import os
import torch
import pickle
from clip import clip
import torch.nn as nn
import torch.nn.functional as F
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer


class TextEncoder(nn.Module):

    def __init__(self, clip_model, hidden_dim=512):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

        input_dim = clip_model.text_projection.shape[0]

        self.fc1 = nn.Linear(input_dim, hidden_dim).to(clip_model.dtype).to(clip_model.ln_final.weight.device)  # 关键修改
        self.fc2 = nn.Linear(hidden_dim, input_dim).to(clip_model.dtype).to(clip_model.ln_final.weight.device)

        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')

        self.relu = nn.ReLU()
        pass

    def forward(self, prompts, tokenized_prompts, imagine=None):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        if imagine == True:
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)

        return x

    pass


import torch
import torch.nn as nn
import clip


class PromptLearner(nn.Module):

    def __init__(self, clip_model, n_ctx, classnames, new_classnames=None, num_classes_new=None):
        super().__init__()
        n_cls = len(classnames)
        num_classes_new = num_classes_new
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        prompt_prefix = " ".join(["X"] * n_ctx)
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        print(f'Initial context: "{prompt_prefix}", Number of context words (tokens): {n_ctx}')

        if n_cls == num_classes_new:
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            new_ctx_vectors = (torch.empty(0, n_ctx, ctx_dim, dtype=dtype))
            nn.init.normal_(new_ctx_vectors, std=0.02)
            self.new_ctx = None
        else:
            ctx_vectors = torch.empty(n_cls - num_classes_new, n_ctx, ctx_dim, dtype=dtype)
            new_ctx_vectors = torch.empty(num_classes_new, n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            nn.init.normal_(new_ctx_vectors, std=0.02)
            self.new_ctx = nn.Parameter(new_ctx_vectors)
        self.ctx = nn.Parameter(ctx_vectors)

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.new_classnames = new_classnames
        pass

    def forward(self):
        device = self.prefix.device
        if self.new_ctx == None:
            ctx = self.ctx.to(device)
        else:
            ctx = torch.cat([self.ctx, self.new_ctx], dim=0).to(device)

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        return torch.cat([self.prefix, ctx, self.suffix], dim=1)


# 1
class CustomCLIPBaseline(nn.Module):

    def __init__(self, clip_model, num_classes_before, num_classes_new, classnames, n_ctx=8):
        super().__init__()
        self.clip_model = clip_model
        self.num_classes_before = num_classes_before
        self.num_classes_new = num_classes_new
        self.n_ctx = n_ctx
        self.classnames = classnames

        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.dtype = clip_model.dtype

        self.prompt_learner = PromptLearner(clip_model, classnames=self.classnames, n_ctx=self.n_ctx,
                                            num_classes_new=self.num_classes_new)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        pass

    def forward(self, image=None, text=None, pseudo_sim=None, tokenized_prompts=None):
        result_dict = {}
        prompt_features = self.text_encoder.forward(self.prompt_learner.forward(), self.tokenized_prompts)
        prompt_features = prompt_features / prompt_features.norm(dim=-1, keepdim=True)

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_logits = self.logit_scale.exp() * image_features @ prompt_features.t()
        result_dict["image_logits"] = image_logits
        result_dict["image_features"] = image_features

        if pseudo_sim is not None:
            pseudo_sim = pseudo_sim.half()
            logits_pseudo = self.logit_scale.exp() * pseudo_sim @ prompt_features.t()
            result_dict["pseudo_logits"] = logits_pseudo
            pass

        if text is not None and tokenized_prompts is not None:
            text_features = self.text_encoder.forward(text, tokenized_prompts, imagine=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_logits = self.logit_scale.exp() * text_features @ prompt_features.t()
            result_dict["text_logits"] = text_logits

            result_dict["text_feature"] = text_features

            pass

        return result_dict

    pass
