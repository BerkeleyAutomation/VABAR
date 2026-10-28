"""A red direction marker identifies a syringe, not a preferred fragment."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize('whole_score', [.75, .04])
def test_whole_syringe_beats_redder_fragment_and_unmarked_spare(whole_score):
    path = Path(__file__).parents[1] / 'graphs/gap_perception/scripts/perceive_target.py'
    spec = importlib.util.spec_from_file_location('perceive_target', path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    rgb = np.full((20, 40, 3), 180, dtype=np.uint8)
    rgb[2:8, 2:4] = [220, 20, 20]
    whole = np.zeros(rgb.shape[:2], dtype=np.uint8)
    whole[2:8, 2:20] = 1
    fragment = np.zeros_like(whole)
    fragment[2:8, 2:4] = 1
    spare = np.zeros_like(whole)
    spare[12:18, 2:20] = 1

    class Selected(Exception):
        pass

    class Context:
        def tool(self, name, **kwargs):
            if name == 'sam3.segment_text':
                if kwargs['query'] != 'syringe':
                    return {'masks': [], 'scores': []}
                return {'masks': [spare, fragment, whole],
                        'scores': [.95, .03, whole_score]}
            assert name == 'geometry.mask_to_world_points'
            np.testing.assert_array_equal(kwargs['mask'], whole)
            raise Selected

    camera = {'name': 'overhead', 'rgb': rgb, 'depth': None, 'intrinsics': None, 'pose': None}
    with pytest.raises(Selected):
        script.run(Context(), {'cameras': [camera]})
