import os
import sys
import time
import torch
import datetime
import numpy as np
from tqdm import tqdm
from clip import clip
import torch.nn as nn
from alisuretool.Tools import Tools
from torch.nn import functional as F
from torch.utils.data import Dataset as TorchDataset
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import Resize, Compose, Normalize, ToTensor
from torchvision.transforms import CenterCrop, RandomResizedCrop, RandomHorizontalFlip
from fscil_data import MiniImageNet, CIFAR100, CUB200
from fscil_data import MiniImageNetForGDIST, CIFAR100ForGDIST, CUB200ForGDIST
from fscil_data import CLASSNAME_miniImageNet, CLASSNAME_CUB200, CLASSNAME_CFIAR100
from fscil_model import CustomCLIPBaseline
from fscil_util import AverageMeter, MetricMeter, Evaluator, EvaluatorPart, UtilTool
import os
print("Current workinx g directory:", os.getcwd())


class MyConfig(object):

    def __init__(self, data_name, exp_name, task_id):
        self.seed = 77
        self.task_id = task_id
        _output_dir = f"./output/{exp_name}/{data_name}/session"
        self.output_dir = f"{_output_dir}{self.task_id}"
        self.model_dir = "" if self.task_id == 0 else f"{_output_dir}{self.task_id - 1}"
        self.g_dist_file_new = os.path.join(self.output_dir, "g_dist.pkl")
        self.g_dist_file_old = "" if self.task_id == 0 else os.path.join(self.model_dir, "g_dist.pkl")
        self.log_file = Tools.new_dir(os.path.join(self.output_dir, "log.txt"))
        Tools.write_to_txt(self.log_file, "begin\n", reset=True)

        self.class_token_position = "end"

        self.device_count = torch.cuda.device_count()

        self.n_ctx = 32
        self.dataset_b = 20
        self.data_replay = True
        self.coop_prec = "fp16"
        self.model_backbone_name = "ViT-B/16"
        self.batch_size = 64
        self.batch_size_test = 100
        self.data_num_works = 4
        self.optim_lr = 0.002
        self.weight_decay = 5e-4
        self.is_cosine_scheduler = True
        self.optim_max_epoch = 50 if self.task_id == 0 else 100
        self.test_freq = 2 if self.task_id == 0 else 10
        self.train_print_freq = 30
        self.loss_weights = [0.99, 0.01]
        self.CustomCLIP = CustomCLIPBaseline

        if data_name == "miniImageNet":
            self.miniImageNet()
        elif data_name == "cub200":
            self.cub200()
        else:
            self.cifar100()

        if self.seed > 0:
            Tools.print("Setting fixed seed: {}".format(self.seed), txt_path=self.log_file)
            UtilTool.set_random_seed(self.seed)
        pass

    def cifar100(self):
        self.dataset_name = "CIFAR100"
        self.dataset_num_classes = 100
        self.dataset_num_classes_base = 60
        self.dataset_class_per_task = 5
        self.dataset_num_shots = 5
        self.task_num = int((self.dataset_num_classes - self.dataset_num_classes_base) / self.dataset_class_per_task)

        normalize = Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        self.transform_train = Compose(
            [RandomResizedCrop((224, 224), scale=(0.08, 1.0), interpolation=InterpolationMode.BICUBIC),
             RandomHorizontalFlip(), ToTensor(), normalize])
        self.transform_test = Compose([Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                                       ToTensor(), normalize])
        pass

    def cub200(self):
        self.dataset_root = "/root/autodl-tmp/data"
        self.dataset_name = "CUB200"
        self.dataset_num_classes = 200
        self.dataset_num_classes_base = 100
        self.dataset_class_per_task = 10
        self.dataset_num_shots = 5
        self.task_num = int((self.dataset_num_classes - self.dataset_num_classes_base) / self.dataset_class_per_task)

        normalize = Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        # self.transform_train = Compose([RandomResizedCrop((224, 224), scale=(0.08, 1.0), interpolation=InterpolationMode.BICUBIC),
        #                                 RandomHorizontalFlip(), ToTensor(), normalize])
        # self.transform_test = Compose([Resize((224, 224), interpolation=InterpolationMode.BICUBIC), ToTensor(), normalize])
        self.transform_train = Compose([Resize(256), RandomResizedCrop((224, 224), scale=(0.08, 1.0),
                                                                       interpolation=InterpolationMode.BICUBIC),
                                        RandomHorizontalFlip(), ToTensor(), normalize])
        self.transform_test = Compose([Resize(256, interpolation=InterpolationMode.BICUBIC),
                                       CenterCrop((224, 224)), ToTensor(), normalize])

        self.optim_max_epoch = 100 if self.task_id == 0 else 50
        self.test_freq = 10 if self.task_id == 0 else 10
        self.train_print_freq = 20
        pass

    def miniImageNet(self):
        self.dataset_root = "/root/autodl-tmp/data"
        self.dataset_name = "miniImageNet"
        self.dataset_num_classes = 100
        self.dataset_num_classes_base = 60
        self.dataset_class_per_task = 5
        self.dataset_num_shots = 5
        self.task_num = int((self.dataset_num_classes - self.dataset_num_classes_base) / self.dataset_class_per_task)

        normalize = Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        self.transform_train = Compose([RandomResizedCrop((224, 224), scale=(0.08, 1.0),
                                                          interpolation=InterpolationMode.BICUBIC),
                                        RandomHorizontalFlip(), ToTensor(), normalize])
        self.transform_test = Compose([Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                                       ToTensor(), normalize])
        pass

    pass


class MyDataLoader(object):

    def __init__(self, config):
        self.config = config
        if self.config.dataset_name == 'miniImageNet':
            train_set = MiniImageNet(data_root=self.config.dataset_root, tfm=self.config.transform_train,
                                     tfm_test=self.config.transform_test, task_id=self.config.task_id, mode='train',
                                     class_per_task=self.config.dataset_class_per_task,
                                     b=self.config.dataset_b, g_dist_file_old=self.config.g_dist_file_old)
            train_set_gdist = MiniImageNetForGDIST(dataset=train_set)  # 用于估计分布
            test_set = MiniImageNet(data_root=self.config.dataset_root, tfm=self.config.transform_test,
                                    tfm_test=self.config.transform_test, task_id=self.config.task_id, mode='test',
                                    class_per_task=self.config.dataset_class_per_task)
            classnames = CLASSNAME_miniImageNet
        elif self.config.dataset_name == 'CUB200':
            train_set = CUB200(data_root=self.config.dataset_root, shot=self.config.dataset_num_shots,
                               tfm=self.config.transform_train, tfm_test=self.config.transform_test,
                               task_id=self.config.task_id, mode='train',
                               class_per_task=self.config.dataset_class_per_task,
                               b=self.config.dataset_b, g_dist_file_old=self.config.g_dist_file_old)
            train_set_gdist = CUB200ForGDIST(dataset=train_set)
            test_set = CUB200(data_root=self.config.dataset_root,
                              tfm=self.config.transform_test, tfm_test=self.config.transform_test,
                              task_id=self.config.task_id, mode='test',
                              class_per_task=self.config.dataset_class_per_task)
            classnames = CLASSNAME_CUB200
        else:
            train_set = CIFAR100(shot=self.config.dataset_num_shots, tfm=self.config.transform_train,
                                 tfm_test=self.config.transform_test, task_id=self.config.task_id, mode='train',
                                 class_per_task=self.config.dataset_class_per_task,
                                 b=self.config.dataset_b, g_dist_file_old=self.config.g_dist_file_old)
            train_set_gdist = CIFAR100ForGDIST(dataset=train_set)
            test_set = CIFAR100(tfm=self.config.transform_test, tfm_test=self.config.transform_test,
                                task_id=self.config.task_id, mode='test',
                                class_per_task=self.config.dataset_class_per_task)
            classnames = CLASSNAME_CFIAR100
            pass

        self.classnames = classnames

        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.config.batch_size,
                                                        num_workers=self.config.data_num_works,
                                                        drop_last=False, shuffle=True)
        self.train_loader_gdist = torch.utils.data.DataLoader(train_set_gdist, batch_size=self.config.batch_size_test,
                                                              num_workers=self.config.data_num_works, drop_last=False)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.config.batch_size_test,
                                                       num_workers=self.config.data_num_works, drop_last=False)
        pass

    pass


class MyModel(object):

    def __init__(self, config, classnames):
        self.config = config
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.classnames = classnames
        self.num_classes_new = self.config.dataset_num_classes_base \
            if self.config.task_id == 0 else self.config.dataset_class_per_task
        self.num_classes_before = 0 if self.config.task_id == 0 else \
            (self.config.dataset_num_classes_base + self.config.dataset_class_per_task * (self.config.task_id - 1))
        self.encounter_class_id = self.num_classes_new + self.num_classes_before

        self.clip_model = self.load_clip_to_cpu(backbone_name=self.config.model_backbone_name)
        if self.config.coop_prec == "fp32":
            self.clip_model.float()

        Tools.print("Building custom CLIP", txt_path=self.config.log_file)
        self.model = self.config.CustomCLIP(self.clip_model, self.num_classes_before, self.num_classes_new,
                                            classnames=self.classnames[:self.encounter_class_id],
                                            n_ctx=self.config.n_ctx)

        Tools.print("Configuring trainable parameters...", txt_path=self.config.log_file)

        for name, param in self.model.named_parameters():
            param.requires_grad_(False)


        self.model.to(self.device)

        trainable_params = []

        class_params = [param for name, param in self.model.named_parameters()
                        if "prompt_learner.ctx" in name and "new_ctx" not in name]
        new_class_params = [param for name, param in self.model.named_parameters()
                            if "prompt_learner.new_ctx" in name]

        imagine_params = [param for name, param in self.model.named_parameters()
                          if "fc1" in name or "fc2" in name]

        if self.config.task_id == 0:
            trainable_params.extend([
                {'params': class_params, 'lr': 5e-3, 'name': 'base_prompt'},
                {'params': new_class_params, 'lr': 0.05, 'name': 'new_prompt'},
                {'params': imagine_params, 'lr': 0.01, 'name': 'imagine_module'}
            ])
            Tools.print("[Base] Training: prompt(新旧类别) + 想象模块", txt_path=self.config.log_file)
        else:
            trainable_params.extend([
                {'params': class_params, 'lr': 5e-3, 'name': 'base_prompt'},
                {'params': new_class_params, 'lr': 0.05, 'name': 'new_prompt'},
                {'params': imagine_params, 'lr': 0.5, 'name': 'imagine_module_finetune'}
            ])
            Tools.print("[Incremental] Finetuning: prompt(新旧类别) + 想象模块", txt_path=self.config.log_file)

        for group in trainable_params:
            for param in group['params']:
                param.requires_grad_(True)

        self.optim = torch.optim.SGD(
            trainable_params,
            momentum=0.9,
            weight_decay=self.config.weight_decay
        )

        Tools.print("Trainable parameters:", txt_path=self.config.log_file)
        for group in self.optim.param_groups:
            Tools.print(f"{group['name']}: lr={group['lr']}, num_params={len(group['params'])}",
                        txt_path=self.config.log_file)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, self.config.optim_max_epoch)

        if self.config.device_count > 1:
            device_ids = [i for i in range(self.config.device_count)]
            self.model = nn.DataParallel(self.model, device_ids)
            Tools.print(f"Multiple GPUs detected (n_gpus={self.config.device_count}), "
                        f"use all of them: {device_ids}", txt_path=self.config.log_file)
        else:
            Tools.print(f"Single GPU detected (n_gpus={self.config.device_count})", txt_path=self.config.log_file)
            pass
        pass

    @staticmethod
    def load_clip_to_cpu(backbone_name):
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)
        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None
        except RuntimeError:
            model = None
            state_dict = torch.load(model_path, map_location="cpu")
        model = clip.build_model(state_dict or model.state_dict())
        return model

    pass


class MyTrainer(object):

    def __init__(self, config):
        self.config = config
        self.my_dataloader = MyDataLoader(config=config)
        self.my_model = MyModel(config=config, classnames=self.my_dataloader.classnames)
        pass

    def train(self):
        self.load_model(self.config.model_dir)
        self.test(epoch=0)
        best_result = 0.0
        for epoch in range(self.config.optim_max_epoch):
            # 运行 Epoch
            self.run_epoch(epoch=epoch)
            # 学习率变化
            if self.config.is_cosine_scheduler:
                self.my_model.scheduler.step()
                pass
            # 测试并保存最好的模型
            if (epoch + 1) % self.config.test_freq == 0:
                curr_result = self.test(epoch=epoch + 1)
                if curr_result > best_result:
                    best_result = curr_result
                    self.save_model(epoch + 1, self.config.output_dir,
                                    val_result=curr_result, model_name="model-best.pth.tar")
                pass
            pass
        Tools.print("Finish training")
        # 先加载最好的模型，再保存分布
        self.load_model(self.config.output_dir)
        self.save_g_dist()
        pass

    def run_epoch(self, epoch=0):
        self.my_model.model.train()

        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        end = time.time()
        num_batches = len(self.my_dataloader.train_loader)
        for batch_idx, batch in enumerate(self.my_dataloader.train_loader):
            data_time.update(time.time() - end)

            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (batch_idx + 1) % self.config.train_print_freq == 0
            only_few_batches = num_batches < self.config.train_print_freq
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += num_batches - batch_idx - 1
                nb_remain += (self.config.optim_max_epoch - epoch - 1) * num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{epoch + 1}/{self.config.optim_max_epoch}]"]
                info += [f"batch [{batch_idx + 1}/{num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.my_model.optim.param_groups[0]['lr']:.4e}"]
                info += [f"eta {eta}"]
                Tools.print(" ".join(info), txt_path=self.config.log_file)
                pass
            end = time.time()
            pass



        pass

    def forward_backward(self, batch):
        if self.config.task_id == 0:
            image, label, text, text_target = batch
            pseudo_sim, pseudo_label = None, None
        else:
            image, label, pseudo_sim, pseudo_label, text, text_target = batch
            pass

        image = image.to(self.my_model.device)
        label = label.to(self.my_model.device)
        txt_embedding, tokenized_prompts = self.text_to_embedding(text)
        txt_embedding = txt_embedding.to(self.my_model.device)
        tokenized_prompts = tokenized_prompts.to(self.my_model.device)
        text_target = text_target.to(self.my_model.device)
        if pseudo_sim is not None and pseudo_label is not None:
            pseudo_sim = pseudo_sim.view(-1, pseudo_sim.shape[-1])
            pseudo_label = pseudo_label.view(-1)
            pseudo_label = pseudo_label.to(self.my_model.device)
            pseudo_sim = pseudo_sim.to(self.my_model.device)
            pass

        _outputs = self.my_model.model(image, txt_embedding, pseudo_sim, tokenized_prompts=tokenized_prompts)
        all_label = torch.cat((text_target, label))
        all_output = torch.cat((_outputs["text_logits"], _outputs["image_logits"]))

        loss_imagine = F.mse_loss(_outputs["image_features"], _outputs["text_feature"])

        if self.config.data_replay and "pseudo_logits" in _outputs:
            all_label = torch.cat((pseudo_label, all_label))
            all_output = torch.cat((_outputs["pseudo_logits"], all_output))

            pass

        loss = F.cross_entropy(all_output, all_label)

        total_loss = (
                self.config.loss_weights[0] * loss +
                self.config.loss_weights[1] * loss_imagine
        )

        loss_summary = {
            "acc": UtilTool.compute_accuracy(all_output, all_label)[0].item(),
            "image_text_loss": loss_imagine.item(),
            "ce_loss": loss.item(),
            "loss": total_loss.item()
        }


        self.my_model.optim.zero_grad()
        total_loss.backward()
        self.my_model.optim.step()
        return loss_summary
    def text_to_embedding(self, text):
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in text]).to(self.my_model.device)
        with torch.no_grad():
            embedding = self.my_model.clip_model.token_embedding(tokenized_prompts).type(self.my_model.clip_model.dtype)
        return embedding, tokenized_prompts

    def test(self, epoch=None):
        self.my_model.model.eval()

        evaluator = Evaluator()
        evaluator.reset()
        evaluator_part = EvaluatorPart(split_1=0, split_2=self.config.dataset_num_classes_base)
        evaluator_part.reset()
        evaluator_part2 = EvaluatorPart(split_1=self.config.dataset_num_classes_base,
                                        split_2=self.my_dataloader.train_loader.dataset.end_class_id + 1)
        evaluator_part2.reset()

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.my_dataloader.test_loader)):
                image, label = batch
                image = image.to(self.my_model.device)
                label = label.to(self.my_model.device)

                _outputs = self.my_model.model(image)
                output = _outputs["image_logits"]
                evaluator.process(output, label)
                evaluator_part.process(output, label)
                evaluator_part2.process(output, label)
                pass
            pass

        results = evaluator.evaluate()
        results_part = evaluator_part.evaluate()
        results_part2 = evaluator_part2.evaluate()
        Tools.print(f"epoch: {epoch} | total: {results['total']} | correct: {results['correct']} | "
                    f"accuracy: {results['accuracy']:.2f} | error: {results['error_rate']:.2f} | "
                    f"macro_f1: {results['macro_f1']:.2f}", txt_path=self.config.log_file)
        Tools.print(f"num: {evaluator_part.split_2 - evaluator_part.split_1} | total: {results_part['total']} | "
                    f"correct: {results_part['correct']} | "
                    f"accuracy: {results_part['accuracy']:.2f} | error: {results_part['error_rate']:.2f} | "
                    f"macro_f1: {results_part['macro_f1']:.2f}", txt_path=self.config.log_file)
        Tools.print(f"num: {evaluator_part2.split_2 - evaluator_part2.split_1} | total: {results_part2['total']} | "
                    f"correct: {results_part2['correct']} | "
                    f"accuracy: {results_part2['accuracy']:.2f} | error: {results_part2['error_rate']:.2f} | "
                    f"macro_f1: {results_part2['macro_f1']:.2f}", txt_path=self.config.log_file)
        return list(results.values())[0]

    def save_g_dist(self):
        feature_dict = {}
        self.my_model.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.my_dataloader.train_loader_gdist)):
                image, label = batch
                image = image.to(self.my_model.device)
                _outputs = self.my_model.model(image)
                output = _outputs["image_features"].detach().cpu()
                for one_idx, one_label in enumerate(label):
                    one_label = int(one_label)
                    if one_label not in feature_dict:
                        feature_dict[one_label] = []
                        pass
                    feature_dict[one_label].append(output[one_idx][None])
                    pass
                pass
            pass

        g_dist = {}
        for key in feature_dict:
            feat_list = torch.cat(feature_dict[key]).numpy()
            g_dist[key] = []
            for j in range(feat_list.shape[1]):
                f_j = feat_list[:, j]
                mean = np.mean(f_j)
                std = np.std(f_j)
                g_dist[key].append({'mean': mean, 'std': std})
                pass
            pass

        # 合并旧的分布
        g_dist_old = Tools.read_from_pkl(self.config.g_dist_file_old) if self.config.g_dist_file_old else {}
        for key in g_dist:
            if key not in g_dist_old:
                g_dist_old[key] = g_dist[key]
            pass

        Tools.print("Save to {}".format(self.config.g_dist_file_new))
        Tools.write_to_pkl(self.config.g_dist_file_new, g_dist_old)
        pass

    def save_model(self, epoch, directory, is_best=False, val_result=None, model_name=""):
        model_dict = self.my_model.model.state_dict()
        optim_dict = self.my_model.optim.state_dict()

        UtilTool.save_checkpoint({"state_dict": model_dict, "epoch": epoch + 1,
                                  "optimizer": optim_dict, "val_result": val_result},
                                 directory, is_best=is_best, model_name=model_name)
        pass

    def load_model(self, model_dir, epoch=None):
        if not model_dir or not os.path.exists(model_dir):
            Tools.print("Note that load_model() is skipped as no pretrained model is given",
                        txt_path=self.config.log_file)
            return

        model_file = "model-best.pth.tar"
        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)
            pass

        model_path = os.path.join(model_dir, model_file)
        if not os.path.exists(model_path):
            raise FileNotFoundError('Model not found at "{}"'.format(model_path))

        # _module = ""
        _module = "module."

        checkpoint = UtilTool.load_checkpoint(model_path)
        state_dict = checkpoint["state_dict"]
        state_dict = {f"{_module}{one}": state_dict[one] for one in state_dict}
        state_dict = {one: state_dict[one] for one in state_dict
                      if ("token_prefix" not in one) and ("token_concept" not in one) and ("token_suffix" not in one)
                      and ("prefix" not in one) and ("suffix" not in one)}

        Tools.print('Loading weights from "{}" (epoch = {})'.format(
            model_path, checkpoint["epoch"]), txt_path=self.config.log_file)
        model_state_dict = self.my_model.model.state_dict()

        # 部分加载权重
        for key in state_dict:
            if key in model_state_dict:
                if state_dict[key].shape == model_state_dict[key].shape:
                    model_state_dict[key] = state_dict[key]
                else:
                    if key == "module.prompt_learner.ctx":
                        old_num_classes = state_dict[key].shape[0]
                        new_num_classes = model_state_dict[key].shape[0]
                        if old_num_classes <= new_num_classes:
                            model_state_dict[key][:old_num_classes] = state_dict[key]
                        else:
                            model_state_dict[key] = state_dict[key][:new_num_classes]
                    else:
                        print(
                            f"Ignoring size mismatch for {key}: checkpoint shape {state_dict[key].shape}, model shape {model_state_dict[key].shape}")

        self.my_model.model.load_state_dict(model_state_dict, strict=False)
        pass

    pass


if __name__ == "__main__":
    argv = sys.argv[1:]
    print('mm')
    data_name, exp_name, task_id = (argv[0], argv[1], int(argv[2])) if len(argv) else ("mi", "train", 0)
    data_name = "miniImageNet" if data_name == "mi" else ("cub200" if data_name == "cu" else "cifar100")

    config = MyConfig(data_name, exp_name, task_id)
    Tools.print(f"data_name = {data_name}, exp_name = {exp_name}, task_id = {task_id}",
                txt_path=config.log_file)
    trainer = MyTrainer(config=config)
    trainer.train()
    pass
