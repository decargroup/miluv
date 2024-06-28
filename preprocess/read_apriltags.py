import xml.etree.ElementTree as ET
import pandas as pd
import yaml

def read_vsk(
    fname, 
    marker_prefix="apriltags_", 
    parameter_prefix="apriltags_apriltags"
):
    out_data = []
    tree = ET.parse(fname)
    root = tree.getroot()
    
    # Get order of marker IDs
    marker_ids = []
    for parent in root.findall("MarkerSet"):  
        for child in parent.findall("Markers"):
            for grandchild in child.findall("Marker"):
                name = grandchild.attrib["NAME"]
                try:
                    id = int(name.split(marker_prefix)[1])
                except Exception as e:
                    print(f"Error parsing name {name} using marker_prefix {marker_prefix}")
                    print(e)
                    continue
                marker_ids.append(id)
    
    # Get marker positions
    i = 0
    for child in root.findall("Parameters"):  # Parameters,
        for grandchild in child.findall("Parameter"):  # Parameters,
            name = grandchild.attrib["NAME"]
            try:
                family, rest = name.split(parameter_prefix)
            except Exception as e:
                print(f"Error parsing name {name} using parameter_prefix {parameter_prefix}")
                print(e)
                continue
            id_, coord = rest.split("_")
            id_ = str(marker_ids[i])
            value = 1e-3 * float(grandchild.attrib["VALUE"])
            out_data.append(dict(family=family, id=id_, coord=coord, value=value))
            if coord == "z":
                i += 1
    df = pd.DataFrame(out_data)
    pt = pd.pivot_table(data=df, index="id", values="value", columns="coord")
    pt["position"] = pt.apply(lambda row: str([row["x"], row["y"], row["z"]]), axis=1)
    pt.drop(columns=["x", "y", "z"], inplace=True)
    return pt.sort_index()

# Configuration 0
data_0_part1 = read_vsk(
    "data/setup/apriltags_0_part1_v1.vsk",
    parameter_prefix="apriltags_00_apriltags_00"
)
data_0_part2 = read_vsk(
    "data/setup/apriltags_0_part2_v1.vsk", 
    parameter_prefix="apriltags_00_1_apriltags_00_1"
)
data_0 = pd.concat([data_0_part1, data_0_part2]).sort_index()

# Configuration 0b
data_0b = read_vsk(
    "data/setup/apriltags_0_v2.vsk",
    marker_prefix="april_tags_",
    parameter_prefix="april_tags_0_v2_april_tags_0_v2"
)

# Configuration 1
data_1_part1 = read_vsk(
    "data/setup/apriltags_1_part1.vsk",
    marker_prefix="april_tags_",
    parameter_prefix="april_tags_0_v2_april_tags_0_v2"
)
data_1_part2 = read_vsk(
    "data/setup/apriltags_1_part2.vsk",
    parameter_prefix="apriltags_1_apriltags_1"
)
data_1 = pd.concat([data_1_part1, data_1_part2]).sort_index()

# Configuration 2
data_2_part1 = read_vsk(
    "data/setup/apriltags_2_part1.vsk",
    marker_prefix="april_tags_",
    parameter_prefix="april_tags_0_v2_april_tags_0_v2"
)
data_2_part2 = read_vsk(
    "data/setup/apriltags_2_part2.vsk",
    parameter_prefix="apriltags_3_apriltags_3"
)
data_2 = pd.concat([data_2_part1, data_2_part2]).sort_index()

# Save to yaml file
data = {
    "0": data_0.to_dict()["position"],
    "0b": data_0b.to_dict()["position"],
    "1": data_1.to_dict()["position"],
    "2": data_2.to_dict()["position"]
}
with open('config/apriltags/apriltags.yaml', 'w') as file:
    yaml.dump(data, file)
