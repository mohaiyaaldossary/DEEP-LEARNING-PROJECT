import unittest
from zone_state import ZoneEvents, validate_polygon

class EventsTest(unittest.TestCase):
    def test_entry_stay_exit_reentry(self):
        m=ZoneEvents(exit_confirm=2)
        self.assertEqual(m.update(0,{1:False})[0],[])
        self.assertEqual(m.update(1,{1:True})[0],[(1,'entry')])
        self.assertEqual(m.update(2,{1:True})[0],[])
        m.update(3,{1:False});m.update(4,{1:False})
        self.assertEqual(m.update(5,{1:True})[0],[(1,'entry')])
    def test_missing_is_not_exit_or_permanent_warning(self):
        m=ZoneEvents(exit_confirm=2)
        m.update(0,{1:True});m.update(1,{1:False})
        self.assertFalse(m.update(2,{})[1])
        m.update(3,{1:False})
        self.assertEqual(m.update(4,{1:True})[0],[])
    def test_initial_multiple_and_expiry(self):
        m=ZoneEvents(ttl=2)
        self.assertEqual(len(m.update(0,{1:True,2:True})[0]),2)
        m.update(3,{})
        self.assertEqual(m.states,{})
        self.assertEqual(m.update(4,{1:True})[0],[(1,'initial_presence')])
    def test_boundary_flicker(self):
        m=ZoneEvents(exit_confirm=2);m.update(0,{1:True})
        m.update(1,{1:False})
        self.assertEqual(m.update(2,{1:True})[0],[])
    def test_polygons(self):
        self.assertIsNone(validate_polygon([(0,0),(10,0),(10,10),(0,10)]))
        for p in [[(0,0),(1,1)],[(0,0),(1,1),(2,2)],[(0,0),(10,10),(0,10),(10,0)],[(0,0),(10,0),(10,10),(0,0)],[(0,0),(10,0),(10,10),(5,0),(0,10)]]:
            self.assertIsNotNone(validate_polygon(p))

if __name__=='__main__':unittest.main()
