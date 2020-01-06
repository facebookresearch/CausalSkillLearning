#!/usr/bin/env python
import unittest

def func(x):
	return x+1

class MyTest(unittest.TestCase):
	def test_1(self):
		self.assertTrue(func(4)==5)

	def test_2(self):
		self.assertTrue(func(4)==6)

if __name__ == '__main__':
    unittest.main()