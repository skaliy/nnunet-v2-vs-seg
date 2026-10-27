import csv
import io
import json
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from pathlib import Path

import numpy as np
from pydicom import dcmread

from nnunet_inference.redcap_output import write_redcap_output, model_repeat_instance
from nnunet_inference import redcap_output
from nnunet_inference.tests.synthetic import write_image as _write_test_image

DICTIONARY_HEADER = 'Variable / Field Name,Form Name,Field Type,Field Label\n'
DICTIONARY_ROWS = ('mask_base64_gzip_compressed,pr2mask,text,mask_base64_gzip_compressed\n'
                   'number_of_pixel,pr2mask,text,Number of pixel\nphysical_size,pr2mask,text,Physical size\n')
PR2MASK_COLUMNS = ['mask_base64_gzip_compressed', 'number_of_pixel', 'physical_size']
ROUTING = ('record_id', 'redcap_event_name', 'redcap_repeat_instrument', 'redcap_repeat_instance')
DEPLOYMENT = {'model_type': 'nnunet_large', 'bundle_sha256': 'e91f17a3af1621a773a562ba12e426404322119234b91195fc8ef9382b0e79cc', 'members': [{'member_id': f'fold_{n}'} for n in range(5)]}


def fixture(root, empty=False):
    source, work = root/'source', root/'work'
    source.mkdir()
    (work/'labels'/'1.2.8').mkdir(parents=True)
    mask = np.zeros((3, 2, 3), dtype=np.uint16)
    if not empty:
        mask[0, 0, 2] = mask[2, 1, 0] = 1
    # Oblique to the usual axial grid, anisotropic and irregular slice positions.
    for i, offset in enumerate([0., 2., 5.]):
        kwargs = dict(orientation=(0, 1, 0, 0, 0, 1), pixel_spacing=(.7, .9), position=(offset, 12, 31))
        p = _write_test_image(source/f'{2-i}', i, **kwargs)
        d = dcmread(p); d.PatientID = 'SYNTHETIC'; d.ReferringPhysicianName = 'EventName:test_arm_1'; d.save_as(p)
        p = _write_test_image(work/'labels'/'1.2.8'/f'{2-i}', i, series_uid='1.2.8', sop_uid=f'1.2.8.{i+1}', **kwargs)
        d = dcmread(p); d.PixelData = mask[i].astype('<u2').tobytes(); d.save_as(p)
    report = work/'redcap'/'1.2.9'
    report.mkdir(parents=True)
    rows = [dict(record_id='SYNTHETIC', redcap_event_name='test_arm_1', redcap_repeat_instrument='pr2mask',
                 redcap_repeat_instance=str(i), field_name=name, value=value)
            for i, size in [(1, '20'), (2, '50'), (3, '50')]
            for name, value in [('mask_base64_gzip_compressed', f'm{i}'), ('number_of_pixel', size), ('physical_size', f'{size}.{i}')]]
    (report/'output.json').write_text(json.dumps([] if empty else rows))
    with zipfile.ZipFile(report/'output_data_dictionary.zip', 'w') as z:
        z.writestr('OriginID.txt', 'PR2MASK')
        # Like pr2mask, list the columns only when there is a region.
        z.writestr('instrument.csv', DICTIONARY_HEADER + ('' if empty else DICTIONARY_ROWS))
    return source, work, mask, report, ([] if empty else rows)


class Pr2maskDictionaryTests(unittest.TestCase):
    def test_shipped_dictionary_lists_pr2mask_columns(self):
        names = [row['Variable / Field Name'] for row in redcap_output.pr2mask_dictionary_rows()]
        self.assertIn('number_of_pixel', names)
        self.assertIn('physical_size', names)
        self.assertEqual(len(names), len(set(names)))
        self.assertFalse([name for name in names if name.startswith('vs_')])


class RedcapOutputTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        reference = Path(tmp.name)/'pr2mask_dictionary.csv'
        reference.write_text(DICTIONARY_HEADER + DICTIONARY_ROWS)
        patcher = patch.object(redcap_output, 'PR2MASK_DICTIONARY_PATH', reference)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_largest_region_fills_pr2mask_columns_in_model_instance(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, work, _, report, _ = fixture(Path(tmp))
            original = (report/'output_data_dictionary.zip').read_bytes()
            write_redcap_output(work, source, DEPLOYMENT)
            rows = json.loads((report/'output.json').read_text())
            self.assertEqual([r['field_name'] for r in rows], PR2MASK_COLUMNS)
            self.assertEqual({tuple(r[k] for k in ROUTING) for r in rows},
                             {('SYNTHETIC', 'test_arm_1', 'pr2mask', '10')})
            values = {r['field_name']: r['value'] for r in rows}
            # Regions 2 and 3 tie on size: the lower region number fills the columns.
            self.assertEqual([values[name] for name in PR2MASK_COLUMNS], ['m2', '50', '50.2'])
            # pr2mask's dictionary already lists its columns and is left as written.
            self.assertEqual((report/'output_data_dictionary.zip').read_bytes(), original)

    def test_empty_mask_sends_blank_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, work, _, report, _ = fixture(Path(tmp), empty=True)
            write_redcap_output(work, source, DEPLOYMENT)
            rows = json.loads((report/'output.json').read_text())
            self.assertEqual([r['field_name'] for r in rows], PR2MASK_COLUMNS)
            self.assertEqual({tuple(r[k] for k in ROUTING) for r in rows},
                             {('SYNTHETIC', 'test_arm_1', 'pr2mask', '10')})
            self.assertEqual({r['value'] for r in rows}, {''})
            with zipfile.ZipFile(report/'output_data_dictionary.zip') as z:
                self.assertEqual(z.read('OriginID.txt'), b'PR2MASK')
                fields = [r['Variable / Field Name'] for r in csv.DictReader(io.StringIO(z.read('instrument.csv').decode()))]
            self.assertEqual(fields, PR2MASK_COLUMNS)

    def test_mixed_destinations_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, work, _, report, rows = fixture(Path(tmp))
            rows[-1]['record_id'] = 'OTHER'
            (report/'output.json').write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, 'Mixed REDCap destinations'):
                write_redcap_output(work, source, DEPLOYMENT)

    def test_models_have_distinct_destinations_and_reruns_replace_complete_result(self):
        stored = {}
        dynunet = dict(DEPLOYMENT, model_type='dynunet', bundle_sha256='c5642793b0f57b49ea8f04a08d2b8c9deff1523813d87d89eec204f85846d6ec')
        for deployment, empty in [(DEPLOYMENT, False), (dynunet, False), (DEPLOYMENT, True)]:
            with tempfile.TemporaryDirectory() as tmp:
                source, work, _, report, _ = fixture(Path(tmp), empty=empty)
                write_redcap_output(work, source, deployment)
                for row in json.loads((report/'output.json').read_text()):
                    stored[tuple(row[k] for k in ROUTING) + (row['field_name'],)] = row['value']
        by_instance = {}
        for key, value in stored.items():
            by_instance.setdefault(key[3], {})[key[4]] = value
        self.assertEqual(set(by_instance), {'10', '11'})
        self.assertEqual(by_instance['10'], dict.fromkeys(PR2MASK_COLUMNS, ''))
        self.assertEqual(by_instance['11']['physical_size'], '50.2')

    def test_unknown_and_colliding_registry_entries_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unregistered'):
            model_repeat_instance('f'*64)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'registry.json'
            for registry in ({'a'*64: 10, 'b'*64: 10}, {'a'*64: 9}):
                with self.subTest(registry=registry):
                    path.write_text(json.dumps(registry))
                    with patch.object(redcap_output, 'MODEL_INSTANCES_PATH', path):
                        with self.assertRaisesRegex(ValueError, 'Invalid'):
                            model_repeat_instance('a'*64)


if __name__ == '__main__': unittest.main()
