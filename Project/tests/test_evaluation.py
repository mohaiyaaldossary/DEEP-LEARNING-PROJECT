import unittest
import numpy as np
import cv2
from evaluate_video import iou,match_matrix
from zone_state import validate_polygon

class EvaluationTest(unittest.TestCase):
    def test_one_to_one_maximum_cardinality(self):
        pairs=match_matrix(np.array([[.9,.6],[.6,.1]]),.5)
        self.assertEqual(set(pairs),{(0,1),(1,0)})
    def test_empty_and_disjoint(self):
        self.assertEqual(match_matrix(np.zeros((0,2)),.5),[])
        self.assertEqual(iou([0,0,1,1],[2,2,3,3]),0)
        self.assertEqual(iou([0,0,2,2],[0,0,2,2]),1)
    def test_boundary_is_inside(self):
        p=np.array([[0,0],[10,0],[10,10],[0,10]],np.int32)
        self.assertGreaterEqual(cv2.pointPolygonTest(p,(0.,5.),False),0)
        self.assertLess(cv2.pointPolygonTest(p,(-1.,5.),False),0)
    def test_nan_rejected(self):
        self.assertIsNotNone(validate_polygon([(0,0),(float('nan'),0),(2,2)]))
if __name__=='__main__':unittest.main()
