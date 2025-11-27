import os
import numpy as np
from typing import List
from pydub import AudioSegment

def load_raw_audios(paths: List[str], max_len:int=None, min_len:int=None) -> List[AudioSegment]:
    audios = []
    for path in paths:
        for file_name in os.listdir(path):
            if file_name.endswith("wav"):
                background = AudioSegment.from_file(
                    os.path.join(path, file_name),
                    format="wav")
                if max_len is not None and len(background) > max_len:
                    background = background[:max_len]
                if min_len is not None and len(background) < min_len:
                    padding = AudioSegment.silent(
                        duration=min_len-len(background),
                        frame_rate=background.frame_rate)
                    background = background.append(padding, crossfade=0)
                audios.append(background)
    return audios

def _match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

def _get_random_time_segment(segment_len, sample_len):
    """
    Gets a random time segment of duration segment_ms in audio clip.
    :param segment_len: the duration of the segment in ms
    :param sample_len: the length of the audio clip in ms
    :return: a tuple of (segment_start, segment_end) in ms
    """
    if sample_len <= segment_len:
        return 0, sample_len
    segment_start = np.random.randint(low=0, high=sample_len-segment_len)
    segment_end = segment_start + segment_len - 1
    return segment_start, segment_end


def _is_overlapping(segment_time, previous_segments):
    """
    Checks if the time of a segment overlaps with the times of existing segments.
    :param segment_time: a tuple of (segment_start, segment_end) for the new segment
    :param previous_segments: a list of tuples of (segment_start, segment_end) for the existing segments
    :return: True if the time segment overlaps with any of the existing segments, False otherwise
    """
    segment_start, segment_end = segment_time
    overlap = False
    for previous_start, previous_end in previous_segments:
        if segment_start <= previous_end and segment_end >= previous_start:
            overlap = True
            break
    return overlap


def _insert_audio_clip(background, audio_clip, previous_segments, sample_len, attempts=5):
    """
    Insert a new audio segment over the background at a random time step, ensuring that the
    audio segment does not overlap with existing segments.
    :param background: the background audio recording.
    :param audio_clip: the audio clip to be inserted/overlaid.
    :param previous_segments: times when audio segments have already been placed
    :param sample_len: the length of the audio clip in ms
    :param attempts: the number of attempts to find where to insert audio clip
    :return: the updated background audio
    """
    segment_len = len(audio_clip)
    segment_time = _get_random_time_segment(segment_len, sample_len)
    retry_cnt = attempts

    while _is_overlapping(segment_time, previous_segments) and retry_cnt >= 0:
        segment_time = _get_random_time_segment(segment_len, sample_len)
        retry_cnt -= 1

    if not _is_overlapping(segment_time, previous_segments):
        previous_segments.append(segment_time)
        new_background = background.overlay(audio_clip, position=segment_time[0])
    else:
        new_background = background
        segment_time = (sample_len, sample_len)

    return new_background, segment_time


def generate_audio(output_path, backgrounds, positives, negatives, sample_len, max_positive=1, max_negatives=3):
    """
    Creates a training example with a given background, positives, and negatives.
    :param output_path: the path to the file where to save a result
    :param backgrounds: a background audio recording
    :param positives: a list of positive audio segments
    :param negatives: a list of negative audio segments
    :param sample_len: the length of the audio clip in ms
    :param max_positive: the maximum number of positive segments to overlay
    :param max_negatives: the maximum number of negative segments to overlay
    :return: a tuple `(positive_num, negative_num)` with amount of positive and negative segments
    """

    background = backgrounds[np.random.randint(len(backgrounds))]

    # Make background quieter
    background = background - 20

    previous_segments = []

    positive_num = np.random.randint(0, max_positive + 1)
    positive_indices = np.random.randint(len(positives), size=positive_num)
    positive_samples = [positives[i] for i in positive_indices]
    for positive in positive_samples:
        background, segment_time = _insert_audio_clip(
            background,
            positive,
            previous_segments,
            sample_len)

    negative_num = np.random.randint(0, max_negatives + 1)
    negative_indices = np.random.randint(len(negatives), size=negative_num)
    negative_samples = [negatives[i] for i in negative_indices]
    for negative in negative_samples:
        background, _ = _insert_audio_clip(
            background,
            negative,
            previous_segments,
            sample_len)

    background = _match_target_amplitude(background, -20.0)
    _ = background.export(output_path, format="wav")
    return positive_num, negative_num
