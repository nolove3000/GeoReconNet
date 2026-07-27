import unittest
import numpy as np
from formal_experiments.data import FORMAL_SENSOR_INDICES, build_selected_features, select_raw_inputs


class FormalExperimentDataTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(7); self.modal=rng.normal(size=(4,6,21)).astype(np.float32); self.frequency=rng.uniform(1,10,size=(4,6)).astype(np.float32)
    def test_dimensions(self):
        expected={(1,7):22,(2,7):44,(3,7):66,(4,7):88,(5,7):110,(6,7):132,
                  (6,6):114,(6,5):96,(6,4):78,(6,3):60,(6,2):42,(6,1):24,(6,0):6}
        for config,dimension in expected.items():
            modal,frequency=select_raw_inputs(self.modal,self.frequency,*config)
            self.assertEqual(build_selected_features(modal,frequency).shape,(4,dimension))
    def test_sensor_subset_is_fixed_and_tri_axial(self):
        for count in range(1,7):
            modal,_=select_raw_inputs(self.modal,self.frequency,6,count)
            components=[7*a+s for a in range(3) for s in FORMAL_SENSOR_INDICES[count]]
            np.testing.assert_array_equal(modal,self.modal[:,:,components])

    def test_reduced_sensor_subsets_are_nested_free_edge_tails(self):
        for count in range(1,7):
            self.assertEqual(FORMAL_SENSOR_INDICES[count],tuple(range(7-count,7)))
    def test_l2_sign_invariance_after_selection(self):
        modal,frequency=select_raw_inputs(self.modal,self.frequency,4,7); first=build_selected_features(modal,frequency)
        changed=modal.copy(); changed[:,0]*=-3; second=build_selected_features(changed,frequency)
        np.testing.assert_allclose(first,second,rtol=2e-6,atol=2e-6)

    def test_frequency_only_features(self):
        modal,frequency=select_raw_inputs(self.modal,self.frequency,6,0)
        self.assertEqual(modal.shape,(4,6,0))
        np.testing.assert_array_equal(build_selected_features(modal,frequency),self.frequency)

if __name__=='__main__': unittest.main()
