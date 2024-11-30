import numpy as np
import xml.etree.ElementTree as ET


def get_image_points(path):
    tree = ET.parse(path)
    root = tree.getroot()
    markers_section = root.find("./chunk/frames/frame/markers")
    markers = markers_section.findall('marker')
    marker_ids = sorted([int(marker.get('marker_id')) for marker in markers])

    all_camera_ids = set()
    for marker in markers:
        for location in marker.findall('location'):
            all_camera_ids.add(int(location.get('camera_id')))
    camera_ids = sorted(all_camera_ids)

    num_images = len(camera_ids)
    num_points = len(marker_ids)

    table = np.empty((num_images, num_points, 2), dtype=object)
    table.fill(None)

    camera_id_to_index = {cam_id: i for i, cam_id in enumerate(camera_ids)}
    marker_id_to_index = {marker_id: j for j, marker_id in enumerate(marker_ids)}

    for marker in markers:
        marker_id = int(marker.get('marker_id'))
        marker_index = marker_id_to_index[marker_id]

        for location in marker.findall('location'):
            camera_id = int(location.get('camera_id'))
            camera_index = camera_id_to_index[camera_id]
            x = float(location.get('x'))
            y = float(location.get('y'))

            table[camera_index, marker_index, :] = [x, y]

    return table


def get_object_points(path):

    tree = ET.parse(path)
    root = tree.getroot()
    markers_section = root.find("./chunk/markers")
    markers = markers_section.findall('marker')
    marker_positions = []

    for marker in markers:
        reference = marker.find("reference")
        if reference is not None:
            x = float(reference.get("x"))
            y = float(reference.get("y"))
            z = float(reference.get("z"))
            marker_positions.append([x, y, z])

    return np.array(marker_positions)