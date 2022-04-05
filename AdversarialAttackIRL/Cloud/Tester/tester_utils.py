import os

imagenet_tag_map = {
    'baseball': ['棒球', '棒球手', '棒球帽'],
    'computer keyboard': ['电脑键盘', '键盘'],
    'container': ['集装箱', '集装箱船', '运货车厢'],
    'engine': ['引擎', '发动机', '汽车', '蒸汽机车', '跑车', '赛车'],
    'helicopter': ['直升飞机', '直升机', '飞机'],
    'microphone': ['麦克风', '话筒'],
    'musical instrument': ['钢琴', '立式钢琴', '三角钢琴', '管风琴', '手风琴', '琵琶', '吉他', '鼓', '电吉他', '短号', '长号', '口琴', '西洋琴', '小提琴',
                           '班卓琴', '竖琴', '笛子', '排箫'],
    'remote control': ['遥控器'],
    'toilet': ['厕所', '马桶', '便器', '便池', '便坑', '马桶座'],
    'weapon': ['枪', '炮', '舰', '武器', '步枪', '手枪', '机枪', '冲锋枪', '坦克', '驱逐舰', '战斗机', '装甲车', '刀', '剑', '大炮', '狙击步枪', '导弹',
               '子弹', '火箭筒', '自动步枪']
}

image_idx2class = {
    0: 'baseball',
    1: 'computer_keyboard',
    2: 'container',
    3: 'engine',
    4: 'helicopter',
    5: 'microphone',
    6: 'musical_instrument',
    7: 'remote_control',
    8: 'toilet',
    9: 'weapon'
}


def parse_ae_name(name):
    # Parse photo file name
    photo_name_split = name.split('_')
    true_label_prompt_pos = photo_name_split.index('TrueLabel')
    adv_label_prompt_pos = photo_name_split.index('LocalAdvLabel')
    id_pos = photo_name_split.index('id')
    true_label_list = photo_name_split[true_label_prompt_pos + 1:adv_label_prompt_pos]
    adv_label_list = photo_name_split[adv_label_prompt_pos + 1:id_pos]
    true_label = ' '.join(true_label_list)
    adv_label = ' '.join(adv_label_list)
    id = int(photo_name_split[id_pos + 1:][0].split('.')[0])
    return {
        'true_label': true_label,
        'adv_label': adv_label,
        'id': id
    }


def write_log_file(log_file, string):
    if os.path.exists(log_file):
        with open(log_file, 'a') as f:
            f.write(string + '\n')
    else:
        with open(log_file, 'w') as f:
            f.write(string + '\n')
    print(string)


def coarse_grained_match(tag, candidates):
    for sub_tag in candidates:
        if tag in sub_tag or sub_tag in tag:
            return True
    return False


def fine_grained_match(tag, candidates):
    return tag in candidates


def listdir_without_hidden(path):
    orig = os.listdir(path)
    return [x for x in orig if x[0] != '.']
