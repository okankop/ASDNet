import os


def generate_clip_meta(entity_meta_data, midone, half_clip_size):
    max_span_left = _get_clip_max_span(entity_meta_data, midone, -1,
                                       half_clip_size+1)
    max_span_right = _get_clip_max_span(entity_meta_data, midone, 1,
                                        half_clip_size+1)

    clip_data = entity_meta_data[midone-max_span_left:midone+max_span_right+1]
    clip_data = _extend_clip_data(clip_data, max_span_left, max_span_right,
                                  half_clip_size)
    return clip_data


def _get_clip_max_span(csv_data, midone, direction, max):
    idx = 0
    for idx in range(0, max):
        if midone+(idx*direction) < 0:
            return idx-1
        if midone+(idx*direction) >= len(csv_data):
            return idx-1

    return idx


def _extend_clip_data(clip_data, max_span_left, max_span_right, half_clip_size):
    if max_span_left < half_clip_size:
        for i in range(half_clip_size-max_span_left):
            clip_data.insert(0, clip_data[0])

    if max_span_right < half_clip_size:
        for i in range(half_clip_size-max_span_right):
            clip_data.insert(-1, clip_data[-1])

    return clip_data
