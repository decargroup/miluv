import pandas as pd
import yaml
from os.path import join
from typing import Any

def get_experiment_info(path):
    exp_name = path.split('/')[-1]
    df = pd.read_csv(join("config", "experiments.csv"))
    row: pd.DataFrame = df[df["experiment"] == exp_name]
    return row.to_dict(orient="records")[0]

def get_anchors(anchor_constellation):
    with open('config/uwb/anchors.yaml', 'r') as file:
        return yaml.safe_load(file)[anchor_constellation]

def get_tags(flatten=False):
    with open('config/uwb/tags.yaml', 'r') as file:
        moment_arms = yaml.safe_load(file)

    if flatten:
        arms = {}
        for robot, arm in moment_arms.items():
            arms.update(arm)
        return arms
    else:
        return moment_arms

def tags_to_df(anchors: Any = None, 
               moment_arms: Any = None, 
               april_tags: Any = None) -> pd.DataFrame:

    uwb_tags = None
    april_tags = None

    if anchors is not None:
        tags = []
        for key, value in anchors.items():
            # parse value
            value = value.strip('[]').split(',')
            tags.append({
                'tag_id': int(key),
                'parent_id': 'world',
                'position.x': float(value[0]),
                'position.y': float(value[1]),
                'position.z': float(value[2]),
            })
        # Convert the data dictionary to a DataFrame
        uwb_tags = pd.DataFrame(tags)
    
    if moment_arms is not None:
        tags = []
        for robot, arms in moment_arms.items():
            for tag, value in arms.items():
                value = value.strip('[]').split(',')
                tags.append({
                    'tag_id': int(tag),
                    'parent_id': robot,
                    'position.x': float(value[0]),
                    'position.y': float(value[1]),
                    'position.z': float(value[2]),
                })
        tags = pd.DataFrame(tags)
    if uwb_tags is not None:
        uwb_tags = pd.concat([uwb_tags, tags], 
                                    ignore_index=True)
    else:
        uwb_tags = tags
    
    if april_tags is not None:
        tags = []
        for key, value in april_tags.items():
            # parse value
            value = value.strip('[]').split(',')
            tags.append({
                'tag_id': int(key),
                'parent_id': 'world',
                'position.x': float(value[0]),
                'position.y': float(value[1]),
                'position.z': float(value[2]),
            })
        # Convert the data dictionary to a DataFrame
        april_tags = pd.DataFrame(tags)
    return uwb_tags, april_tags