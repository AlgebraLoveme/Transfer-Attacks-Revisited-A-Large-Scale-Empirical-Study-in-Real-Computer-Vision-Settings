from UM_utils import imagenet_tag_map

def fine_grained_matching(value, candidate_list):
    return value in candidate_list

def coarse_grained_matching(value, candidate_list):
    for item in candidate_list:
        if value in item: return True
    return False
