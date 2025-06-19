import os
import sys
import json
import random
import pickle
import torchvision
import numpy as np
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset as TorchDataset

CLASSNAME_CFIAR100 = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle',
                      'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar',
                      'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile',
                      'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
                      'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster',
                      'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange',
                      'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate',
                      'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
                      'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
                      'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone',
                      'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe',
                      'whale', 'willow_tree', 'wolf', 'woman', 'worm']

CLASSNAME_CUB200 = ['Black_footed_Albatross', 'Laysan_Albatross', 'Sooty_Albatross', 'Groove_billed_Ani',
                    'Crested_Auklet', 'Least_Auklet', 'Parakeet_Auklet', 'Rhinoceros_Auklet', 'Brewer_Blackbird',
                    'Red_winged_Blackbird', 'Rusty_Blackbird', 'Yellow_headed_Blackbird', 'Bobolink', 'Indigo_Bunting',
                    'Lazuli_Bunting', 'Painted_Bunting', 'Cardinal', 'Spotted_Catbird', 'Gray_Catbird',
                    'Yellow_breasted_Chat', 'Eastern_Towhee', 'Chuck_will_Widow', 'Brandt_Cormorant',
                    'Red_faced_Cormorant', 'Pelagic_Cormorant', 'Bronzed_Cowbird', 'Shiny_Cowbird', 'Brown_Creeper',
                    'American_Crow', 'Fish_Crow', 'Black_billed_Cuckoo', 'Mangrove_Cuckoo', 'Yellow_billed_Cuckoo',
                    'Gray_crowned_Rosy_Finch', 'Purple_Finch', 'Northern_Flicker', 'Acadian_Flycatcher',
                    'Great_Crested_Flycatcher', 'Least_Flycatcher', 'Olive_sided_Flycatcher',
                    'Scissor_tailed_Flycatcher', 'Vermilion_Flycatcher', 'Yellow_bellied_Flycatcher', 'Frigatebird',
                    'Northern_Fulmar', 'Gadwall', 'American_Goldfinch', 'European_Goldfinch', 'Boat_tailed_Grackle',
                    'Eared_Grebe', 'Horned_Grebe', 'Pied_billed_Grebe', 'Western_Grebe', 'Blue_Grosbeak',
                    'Evening_Grosbeak', 'Pine_Grosbeak', 'Rose_breasted_Grosbeak', 'Pigeon_Guillemot',
                    'California_Gull', 'Glaucous_winged_Gull', 'Heermann_Gull', 'Herring_Gull', 'Ivory_Gull',
                    'Ring_billed_Gull', 'Slaty_backed_Gull', 'Western_Gull', 'Anna_Hummingbird',
                    'Ruby_throated_Hummingbird', 'Rufous_Hummingbird', 'Green_Violetear', 'Long_tailed_Jaeger',
                    'Pomarine_Jaeger', 'Blue_Jay', 'Florida_Jay', 'Green_Jay', 'Dark_eyed_Junco', 'Tropical_Kingbird',
                    'Gray_Kingbird', 'Belted_Kingfisher', 'Green_Kingfisher', 'Pied_Kingfisher', 'Ringed_Kingfisher',
                    'White_breasted_Kingfisher', 'Red_legged_Kittiwake', 'Horned_Lark', 'Pacific_Loon', 'Mallard',
                    'Western_Meadowlark', 'Hooded_Merganser', 'Red_breasted_Merganser', 'Mockingbird', 'Nighthawk',
                    'Clark_Nutcracker', 'White_breasted_Nuthatch', 'Baltimore_Oriole', 'Hooded_Oriole',
                    'Orchard_Oriole', 'Scott_Oriole', 'Ovenbird', 'Brown_Pelican', 'White_Pelican',
                    'Western_Wood_Pewee', 'Sayornis', 'American_Pipit', 'Whip_poor_Will', 'Horned_Puffin',
                    'Common_Raven', 'White_necked_Raven', 'American_Redstart', 'Geococcyx', 'Loggerhead_Shrike',
                    'Great_Grey_Shrike', 'Baird_Sparrow', 'Black_throated_Sparrow', 'Brewer_Sparrow',
                    'Chipping_Sparrow', 'Clay_colored_Sparrow', 'House_Sparrow', 'Field_Sparrow', 'Fox_Sparrow',
                    'Grasshopper_Sparrow', 'Harris_Sparrow', 'Henslow_Sparrow', 'Le_Conte_Sparrow', 'Lincoln_Sparrow',
                    'Nelson_Sharp_tailed_Sparrow', 'Savannah_Sparrow', 'Seaside_Sparrow', 'Song_Sparrow',
                    'Tree_Sparrow', 'Vesper_Sparrow', 'White_crowned_Sparrow', 'White_throated_Sparrow',
                    'Cape_Glossy_Starling', 'Bank_Swallow', 'Barn_Swallow', 'Cliff_Swallow', 'Tree_Swallow',
                    'Scarlet_Tanager', 'Summer_Tanager', 'Artic_Tern', 'Black_Tern', 'Caspian_Tern', 'Common_Tern',
                    'Elegant_Tern', 'Forsters_Tern', 'Least_Tern', 'Green_tailed_Towhee', 'Brown_Thrasher',
                    'Sage_Thrasher', 'Black_capped_Vireo', 'Blue_headed_Vireo', 'Philadelphia_Vireo', 'Red_eyed_Vireo',
                    'Warbling_Vireo', 'White_eyed_Vireo', 'Yellow_throated_Vireo', 'Bay_breasted_Warbler',
                    'Black_and_white_Warbler', 'Black_throated_Blue_Warbler', 'Blue_winged_Warbler', 'Canada_Warbler',
                    'Cape_May_Warbler', 'Cerulean_Warbler', 'Chestnut_sided_Warbler', 'Golden_winged_Warbler',
                    'Hooded_Warbler', 'Kentucky_Warbler', 'Magnolia_Warbler', 'Mourning_Warbler', 'Myrtle_Warbler',
                    'Nashville_Warbler', 'Orange_crowned_Warbler', 'Palm_Warbler', 'Pine_Warbler', 'Prairie_Warbler',
                    'Prothonotary_Warbler', 'Swainson_Warbler', 'Tennessee_Warbler', 'Wilson_Warbler',
                    'Worm_eating_Warbler', 'Yellow_Warbler', 'Northern_Waterthrush', 'Louisiana_Waterthrush',
                    'Bohemian_Waxwing', 'Cedar_Waxwing', 'American_Three_toed_Woodpecker', 'Pileated_Woodpecker',
                    'Red_bellied_Woodpecker', 'Red_cockaded_Woodpecker', 'Red_headed_Woodpecker', 'Downy_Woodpecker',
                    'Bewick_Wren', 'Cactus_Wren', 'Carolina_Wren', 'House_Wren', 'Marsh_Wren', 'Rock_Wren',
                    'Winter_Wren', 'Common_Yellowthroat']

CLASSNAME_miniImageNet = ['house_finch', 'robin', 'triceratops', 'green_mamba', 'harvestman', 'toucan', 'goose',
                          'jellyfish', 'nematode', 'king_crab', 'dugong', 'Walker_hound', 'Ibizan_hound', 'Saluki',
                          'golden_retriever', 'Gordon_setter', 'komondor', 'boxer', 'Tibetan_mastiff', 'French_bulldog',
                          'malamute', 'dalmatian', 'Newfoundland', 'miniature_poodle', 'white_wolf',
                          'African_hunting_dog', 'Arctic_fox', 'lion', 'meerkat', 'ladybug', 'rhinoceros_beetle', 'ant',
                          'black-footed_ferret', 'three-toed_sloth', 'rock_beauty', 'aircraft_carrier', 'ashcan',
                          'barrel', 'beer_bottle', 'bookshop', 'cannon', 'carousel', 'carton', 'catamaran', 'chime',
                          'clog', 'cocktail_shaker', 'combination_lock', 'crate', 'cuirass', 'dishrag', 'dome',
                          'electric_guitar', 'file', 'fire_screen', 'frying_pan', 'garbage_truck', 'hair_slide',
                          'holster', 'horizontal_bar', 'hourglass', 'iPod', 'lipstick', 'miniskirt', 'missile',
                          'mixing_bowl', 'oboe', 'organ', 'parallel_bars', 'pencil_box', 'photocopier', 'poncho',
                          'prayer_rug', 'reel', 'school_bus', 'scoreboard', 'slot', 'snorkel', 'solar_dish',
                          'spider_web', 'stage', 'tank', 'theater_curtain', 'tile_roof', 'tobacco_shop', 'unicycle',
                          'upright', 'vase', 'wok', 'worm_fence', 'yawl', 'street_sign', 'consomme', 'trifle', 'hotdog',
                          'orange', 'cliff', 'coral_reef', 'bolete', 'ear']


class MiniImageNet(TorchDataset):

    def __init__(self, data_root, tfm, tfm_test, task_id, mode, class_per_task=5, b=2, g_dist_file_old=''):
        root = os.path.join(data_root, 'miniImageNet')
        self.IMAGE_PATH = os.path.join(root, 'images')
        self.SPLIT_PATH = os.path.join(root, 'split')
        self.index_list = os.path.join(root, "index_list/mini_imagenet")
        self.TEXT_PATH = os.path.join(root, 'class_text.json')

        self.tfm = tfm
        self.tfm_test = tfm_test
        self.class_per_task = class_per_task
        self.b = b
        self.mode = mode
        self.task_id = task_id
        self.g_dist_file_old = g_dist_file_old

        task_split = {0: 60, 1: 65, 2: 70, 3: 75, 4: 80, 5: 85, 6: 90, 7: 95, 8: 100}

        self.end_class_id = task_split[task_id] - 1

        csv_path = osp.join(self.SPLIT_PATH, mode + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        self.data = []
        self.targets = []
        self.text = {}
        self.data2label = {}
        lb = -1

        self.wnids = []
        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(self.IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            self.data.append(path)
            self.targets.append(lb)
            self.data2label[path] = lb
            pass

        self.text = self.read_all_current_txt(CLASSNAME_miniImageNet[:task_split[task_id]])

        if mode == 'train':
            self.class_to_label = {class_name: i for i, class_name in
                                   enumerate(CLASSNAME_miniImageNet[:task_split[task_id]])}

            if self.task_id > 0:
                index_path = os.path.join(self.index_list, 'session_{}.txt'.format(task_id + 1))
                self.data, self.targets = self.select_from_txt(self.data2label, index_path)  # 所有的训练数据
                assert os.path.exists(self.g_dist_file_old)
                with open(self.g_dist_file_old, 'rb') as f:
                    self.g_dist = pickle.load(f)

            else:
                self.data, self.targets = self.select_from_classes(self.data, self.targets, np.arange(task_split[0]))
                pass
        else:
            self.data, self.targets = self.select_from_classes(self.data, self.targets, np.arange(task_split[task_id]))
        pass

    def select_from_txt(self, data2label, index_path):
        index = []
        lines = [x.strip() for x in open(index_path, 'r').readlines()]
        for line in lines:
            index.append(line.split('/')[3])
        data_tmp = []
        targets_tmp = []
        for i in index:
            img_path = os.path.join(self.IMAGE_PATH, i)
            data_tmp.append(img_path)
            targets_tmp.append(data2label[img_path])
            pass
        return data_tmp, targets_tmp

    def select_from_classes(self, data, targets, index):
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])
            pass
        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, targets = self.data[i], self.targets[i]
        image = self.tfm(Image.open(path).convert('RGB'))

        class_name = CLASSNAME_miniImageNet[targets]
        text = self.text
        txt = random.choice(text[class_name])
        txt_target = targets

        if self.mode == 'train':

            if self.task_id > 0:
                pseudo_label = []
                pseudo_sim_list = []
                have_select = 0
                while True:
                    select_class = random.randint(0, self.end_class_id - self.class_per_task)
                    g_dist_all = self.g_dist[select_class]
                    pseudo_sim = []
                    for dim in range(len(g_dist_all)):
                        dim_param = g_dist_all[dim]
                        mean = dim_param['mean']
                        std = dim_param['std']
                        pseudo_value = np.random.normal(mean, std, 1)[0]
                        pseudo_sim.append(pseudo_value)
                    pseudo_sim = np.array(pseudo_sim)
                    pseudo_sim_list.append(pseudo_sim)
                    pseudo_label.append(select_class)
                    have_select += 1
                    if have_select == self.b:
                        break
                pseudo_label = np.array(pseudo_label)
                pseudo_sim_list = np.array(pseudo_sim_list)
                return image, targets, pseudo_sim_list, pseudo_label, txt, txt_target
            return image, targets, txt, txt_target
        return image, targets

    def read_all_current_txt(self, class_name):
        select_class = {}
        with open(self.TEXT_PATH, 'r', encoding='utf-8') as file:
            data = json.load(file)

        for name in class_name:
            if name in data:
                select_class[name] = data[name]
        return select_class

    pass


class MiniImageNetForGDIST(TorchDataset):

    def __init__(self, dataset):
        self.dataset = dataset
        pass

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        path, targets = self.dataset.data[i], self.dataset.targets[i]
        image = self.dataset.tfm_test(Image.open(path).convert('RGB'))
        return image, targets

    pass


class CIFAR100(TorchDataset):
    def __init__(self, tfm, tfm_test, task_id, mode, shot=5, class_per_task=5, b=2, g_dist_file_old=''):
        root = os.path.join("/root/autodl-tmp/data", 'cifar100')
        self.TEXT_PATH = os.path.join(root, 'class_text.json')
        self.novel_task_len = int(40 / class_per_task)
        self.task_len = self.novel_task_len + 1

        self.tfm = tfm
        self.tfm_test = tfm_test
        self.class_per_task = class_per_task
        self.b = b
        self.mode = mode
        self.task_id = task_id
        self.g_dist_file_old = g_dist_file_old

        task_split = [[] for x in range(9)]

        for i in range(60):
            task_split[0].append(i)

        for i in range(1, 9):
            for j in range(5):
                task_split[i].append(60 + (i - 1) * 5 + j)
        if task_id == 0:

            current_class_indices = task_split[0]
        else:
            current_class_indices = list(range(60 + (task_id - 1) * 5)) + task_split[task_id]

        current_class_names = [CLASSNAME_CFIAR100[idx] for idx in current_class_indices]
        self.text = self.read_all_current_txt(current_class_names)

        if mode == 'train':
            if self.task_id > 0:
                assert os.path.exists(self.g_dist_file_old)
                with open(self.g_dist_file_old, 'rb') as f:
                    self.g_dist = pickle.load(f)
                pass

            self.class_to_label = {class_name: i for i, class_name in
                                   enumerate(CLASSNAME_CFIAR100[:len(task_split[task_id])])}
            select_class_id = task_split[task_id]
            self.cifar100 = torchvision.datasets.CIFAR100(root=os.path.expanduser("~/.cache"),
                                                          download=True, train=True, transform=tfm)
            self.class_idx_dict = {x: [] for x in select_class_id}
            self.end_class_id = select_class_id[-1]

            for i, label in enumerate(self.cifar100.targets):
                if label in self.class_idx_dict:
                    self.class_idx_dict[label].append(i)

            self.data_id = []
            for c in select_class_id:
                idx_list = self.class_idx_dict[c]
                if c >= 60:
                    idx_list = random.sample(idx_list, shot)
                self.data_id.extend(idx_list)
                pass
            self.shot = shot
        else:
            self.data_id = []
            task_to_id_end = {0: 60}
            start = 65
            for i in range(1, 9):
                task_to_id_end[i] = start
                start += 5

            select_class_id = [x for x in range(task_to_id_end[task_id])]
            self.end_class_id = select_class_id[-1]
            self.cifar100 = torchvision.datasets.CIFAR100(root=os.path.expanduser("~/.cache"),
                                                          download=True, train=False, transform=tfm_test)

            self.class_idx_dict = {x: [] for x in select_class_id}
            for i, label in enumerate(self.cifar100.targets):
                if label in select_class_id:
                    self.data_id.append(i)
            pass
        pass

    def __len__(self):
        return len(self.data_id)

    def __getitem__(self, index):
        now_data = self.cifar100[self.data_id[index]]
        class_name = CLASSNAME_CFIAR100[now_data[1]]
        text = self.text
        # 获取该类对应的所有文本描述
        txt = random.choice(text[class_name])
        txt_target = now_data[1]

        if self.mode == 'train':

            if self.task_id > 0:
                pseudo_label = []
                pseudo_sim_list = []
                have_select = 0
                while True:
                    select_class = random.randint(0, self.end_class_id - self.class_per_task)
                    g_dist_all = self.g_dist[select_class]
                    pseudo_sim = []
                    for dim in range(len(g_dist_all)):
                        dim_param = g_dist_all[dim]
                        mean = dim_param['mean']
                        std = dim_param['std']
                        pseudo_value = np.random.normal(mean, std, 1)[0]
                        pseudo_sim.append(pseudo_value)
                    pseudo_sim = np.array(pseudo_sim)
                    pseudo_sim_list.append(pseudo_sim)
                    pseudo_label.append(select_class)
                    have_select += 1
                    if have_select == self.b:
                        break
                pseudo_label = np.array(pseudo_label)
                pseudo_sim_list = np.array(pseudo_sim_list)
                return now_data[0], now_data[1], pseudo_sim_list, pseudo_label, txt, txt_target
            return now_data[0], now_data[1], txt, txt_target
        return now_data

    pass

    def read_all_current_txt(self, class_name):
        select_class = {}
        with open(self.TEXT_PATH, 'r', encoding='utf-8') as file:
            data = json.load(file)

        for name in class_name:
            if name in data:
                select_class[name] = data[name]
        return select_class


class CIFAR100ForGDIST(TorchDataset):

    def __init__(self, dataset):
        self.dataset = dataset
        self.dataset.cifar100.transform = self.dataset.tfm_test
        pass

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        now_data = self.dataset.cifar100[self.dataset.data_id[i]]
        return now_data

    pass


class CUB200(TorchDataset):
    # 原先b=7
    def __init__(self, data_root, tfm, tfm_test, task_id, mode, shot=5, class_per_task=10, b=7, g_dist_file_old=''):
        cub_root = os.path.join(data_root, 'CUB_200_2011')
        image_data_txt = os.path.join(cub_root, 'images.txt')
        image_root = os.path.join(cub_root, 'images')
        label_txt = os.path.join(cub_root, 'image_class_labels.txt')
        train_test_split = os.path.join(cub_root, 'train_test_split.txt')
        self.TEXT_PATH = os.path.join(cub_root, 'class_text.json')

        self.tfm = tfm
        self.tfm_test = tfm_test
        self.class_per_task = class_per_task
        self.b = b
        self.mode = mode
        self.task_id = task_id
        self.g_dist_file_old = g_dist_file_old

        self.novel_task_len = int(100 / class_per_task)
        self.task_len = self.novel_task_len + 1

        if self.mode == 'train' and self.task_id > 0:
            assert os.path.exists(self.g_dist_file_old)
            with open(self.g_dist_file_old, 'rb') as f:
                self.g_dist = pickle.load(f)

        image_id_split = {}
        with open(train_test_split, 'r') as f:
            image_split = f.readlines()
            for i in range(len(image_split)):
                image_split[i] = image_split[i].replace('\n', '')
                image_id, is_train = image_split[i].split(" ")
                image_id_split[image_id] = eval(is_train)

        image_id_path_dict = {}
        with open(image_data_txt, 'r') as f:
            image_id_list = f.readlines()
            for i in range(len(image_id_list)):
                image_id_list[i] = image_id_list[i].replace('\n', '')
                image_id, path = image_id_list[i].split(" ")
                image_id_path_dict[image_id] = os.path.join(image_root, path)

        image_id_label_dict = {}
        with open(label_txt, 'r') as f:
            image_label_list = f.readlines()
            for i in range(len(image_label_list)):
                image_label_list[i] = image_label_list[i].replace('\n', '')
                image_id, label = image_label_list[i].split(" ")
                image_id_label_dict[image_id] = eval(label) - 1

        self.images_list = []
        self.labeled_list = []
        task_split = [[] for x in range(self.task_len)]
        for i in range(100):
            task_split[0].append(i)
        for i in range(1, self.task_len):
            for j in range(self.class_per_task):
                task_split[i].append(100 + (i - 1) * self.class_per_task + j)
        select_class_id = task_split[task_id]

        self.class_to_label = {class_name: i for i, class_name in
                               enumerate(CLASSNAME_CUB200[:len(task_split[task_id])])}

        current_and_previous_indices = []
        for t in range(task_id + 1):
            current_and_previous_indices.extend(task_split[t])

        current_class_names = [CLASSNAME_CUB200[idx] for idx in current_and_previous_indices]

        self.text = self.read_all_current_txt(current_class_names)

        if mode == 'train':
            self.end_class_id = select_class_id[-1]

            self.class_idx_dict = {x: [] for x in select_class_id}
            for key in image_id_path_dict:
                if image_id_split[key] == 1:
                    label = image_id_label_dict[key]
                    if label in self.class_idx_dict:
                        self.class_idx_dict[label].append(key)

            for c in select_class_id:
                idx_list = self.class_idx_dict[c]
                if c >= 100:
                    idx_list = random.sample(idx_list, shot)
                for id in idx_list:
                    self.images_list.append(image_id_path_dict[id])
                    self.labeled_list.append(image_id_label_dict[id])

            self.shot = shot
        else:
            self.data = []
            task_to_id_end = {0: 100}
            start = 100 + self.class_per_task
            for i in range(1, self.task_len):
                task_to_id_end[i] = start
                start += self.class_per_task

            select_class_id = [x for x in range(task_to_id_end[task_id])]
            for key in image_id_path_dict:
                if image_id_split[key] == 0:
                    label = image_id_label_dict[key]
                    if label in select_class_id:
                        self.images_list.append(image_id_path_dict[key])
                        self.labeled_list.append(label)
            print('test set number', len(self.images_list))
            pass
        pass

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        img_name, label = self.images_list[idx], self.labeled_list[idx]
        image = self.tfm(Image.open(img_name).convert('RGB'))

        class_name = CLASSNAME_CUB200[label]
        text = self.text

        txt = random.choice(text[class_name])
        txt_target = label

        if self.mode == 'train':
            if self.task_id > 0:
                pseudo_label = []
                pseudo_sim_list = []
                have_select = 0
                while True:
                    select_class = random.randint(0, self.end_class_id - self.class_per_task)
                    g_dist_all = self.g_dist[select_class]
                    pseudo_sim = []
                    for dim in range(len(g_dist_all)):
                        dim_param = g_dist_all[dim]
                        mean = dim_param['mean']
                        std = dim_param['std']
                        pseudo_value = np.random.normal(mean, std, 1)[0]
                        pseudo_sim.append(pseudo_value)
                    pseudo_sim = np.array(pseudo_sim)
                    pseudo_sim_list.append(pseudo_sim)
                    pseudo_label.append(select_class)
                    have_select += 1
                    if have_select == self.b:
                        break
                pseudo_label = np.array(pseudo_label)
                pseudo_sim_list = np.array(pseudo_sim_list)

                return image, label, pseudo_sim_list, pseudo_label, txt, txt_target
            return image, label, txt, txt_target
        return image, label

    def read_all_current_txt(self, class_name):
        select_class = {}
        with open(self.TEXT_PATH, 'r', encoding='utf-8') as file:
            data = json.load(file)

        for name in class_name:
            if name in data:
                select_class[name] = data[name]
        return select_class

    pass

    def label_to_class(self):
        pass


class CUB200ForGDIST(TorchDataset):

    def __init__(self, dataset):
        self.dataset = dataset
        pass

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        img_name, label = self.dataset.images_list[i], self.dataset.labeled_list[i]
        image = self.dataset.tfm_test(Image.open(img_name).convert('RGB'))
        return image, label

    pass


class CUB200FromText(TorchDataset):

    def __init__(self, data_root, tfm, tfm_test, task_id, mode, shot=5, class_per_task=10, b=7, g_dist_file_old=''):
        cub_root = os.path.join(data_root, 'CUB_200_2011')
        image_data_txt = os.path.join(cub_root, 'images.txt')
        image_root = os.path.join(cub_root, 'images')
        label_txt = os.path.join(cub_root, 'image_class_labels.txt')
        train_test_split = os.path.join(cub_root, 'train_test_split.txt')
        session_b_path = os.path.join(cub_root, 'index_list', "cub200")

        self.tfm = tfm
        self.tfm_test = tfm_test
        self.class_per_task = class_per_task
        self.b = b
        self.mode = mode
        self.task_id = task_id
        self.g_dist_file_old = g_dist_file_old

        self.novel_task_len = int(100 / class_per_task)
        self.task_len = self.novel_task_len + 1

        if self.mode == 'train' and self.task_id > 0:
            assert os.path.exists(self.g_dist_file_old)
            with open(self.g_dist_file_old, 'rb') as f:
                self.g_dist = pickle.load(f)

        image_id_split = {}
        with open(train_test_split, 'r') as f:
            image_split = f.readlines()
            for i in range(len(image_split)):
                image_split[i] = image_split[i].replace('\n', '')
                image_id, is_train = image_split[i].split(" ")
                image_id_split[image_id] = eval(is_train)

        image_id_path_dict = {}
        with open(image_data_txt, 'r') as f:
            image_id_list = f.readlines()
            for i in range(len(image_id_list)):
                image_id_list[i] = image_id_list[i].replace('\n', '')
                image_id, path = image_id_list[i].split(" ")
                image_id_path_dict[image_id] = os.path.join(image_root, path)

        image_id_label_dict = {}
        with open(label_txt, 'r') as f:
            image_label_list = f.readlines()
            for i in range(len(image_label_list)):
                image_label_list[i] = image_label_list[i].replace('\n', '')
                image_id, label = image_label_list[i].split(" ")
                image_id_label_dict[image_id] = eval(label) - 1

        self.images_list = []
        self.labeled_list = []

        if mode == 'train':
            task_split = [[] for x in range(self.task_len)]
            for i in range(100):
                task_split[0].append(i)
            for i in range(1, self.task_len):
                for j in range(self.class_per_task):
                    task_split[i].append(100 + (i - 1) * self.class_per_task + j)

            select_class_id = task_split[task_id]
            self.end_class_id = select_class_id[-1]

            self.class_idx_dict = {x: [] for x in select_class_id}
            for key in image_id_path_dict:
                if image_id_split[key] == 1:
                    label = image_id_label_dict[key]
                    if label in self.class_idx_dict:
                        self.class_idx_dict[label].append(key)

            for c in select_class_id:
                if c < 100 or self.task_id <= 0:
                    for id in self.class_idx_dict[c]:
                        self.images_list.append(image_id_path_dict[id])
                        self.labeled_list.append(image_id_label_dict[id])
                    pass
                else:
                    idx_list = self.class_idx_dict[c]
                    is_random = False
                    if is_random:
                        for id in random.sample(idx_list, shot):
                            self.images_list.append(image_id_path_dict[id])
                            self.labeled_list.append(image_id_label_dict[id])
                        pass
                    else:
                        now_image_paths = self.read_session_b(session_b_txt=os.path.join(
                            session_b_path, "session_{}.txt".format(self.task_id + 1)), c=c + 1)
                        for now_image_path in now_image_paths:
                            self.images_list.append(os.path.join(data_root, now_image_path))
                            self.labeled_list.append(c)
                        pass
                    pass

            self.shot = shot
        else:
            self.data = []
            task_to_id_end = {0: 100}
            start = 100 + self.class_per_task
            for i in range(1, self.task_len):
                task_to_id_end[i] = start
                start += self.class_per_task

            select_class_id = [x for x in range(task_to_id_end[task_id])]
            for key in image_id_path_dict:
                if image_id_split[key] == 0:
                    label = image_id_label_dict[key]
                    if label in select_class_id:
                        self.images_list.append(image_id_path_dict[key])
                        self.labeled_list.append(label)
            print('test set number', len(self.images_list))
            pass
        pass

    def read_session_b(self, session_b_txt, c):
        with open(session_b_txt, 'r') as f:
            image_path_list = f.readlines()
            now_paths = [image_path.strip() for image_path in image_path_list if f"{c}." in image_path]
        return now_paths

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        img_name, label = self.images_list[idx], self.labeled_list[idx]
        image = self.tfm(Image.open(img_name).convert('RGB'))

        if self.mode == 'train' and self.task_id > 0:
            pseudo_label = []
            pseudo_sim_list = []
            have_select = 0
            while True:
                select_class = random.randint(0, self.end_class_id - self.class_per_task)
                g_dist_all = self.g_dist[select_class]
                pseudo_sim = []
                for dim in range(len(g_dist_all)):
                    dim_param = g_dist_all[dim]
                    mean = dim_param['mean']
                    std = dim_param['std']
                    pseudo_value = np.random.normal(mean, std, 1)[0]
                    pseudo_sim.append(pseudo_value)
                pseudo_sim = np.array(pseudo_sim)
                pseudo_sim_list.append(pseudo_sim)
                pseudo_label.append(select_class)
                have_select += 1
                if have_select == self.b:
                    break
            pseudo_label = np.array(pseudo_label)
            pseudo_sim_list = np.array(pseudo_sim_list)

            return image, label, pseudo_sim_list, pseudo_label

        return image, label

    pass
