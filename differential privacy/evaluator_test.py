import unittest, evaluator, os, DPapyori, apyori

class Evaluator_Tester(unittest.TestCase):

    # Confirm that the evaluator loads transactions the same way for both the original library
    # and our DP implementation
    def test_load_transactions(self):
        input_file = 'csv_files/CleanDataSetA.csv'
        expected_output = ['Lassi', 'Coffee Powder', 'Butter', 'Yougurt', 'Ghee', 'Cheese']
        self.assertEqual(evaluator.load(os.path.abspath(input_file), DPapyori)[0], expected_output)
        self.assertEqual(evaluator.load(os.path.abspath(input_file), apyori)[0], expected_output)

    # Confirm truncation leaves about 85 % of transactions intact
    def test_truncate_transactions(self):
        input_file = 'csv_files/CleanDataSetA.csv'
        example_epsilon = 0.5
        t = evaluator.load(os.path.abspath(input_file), DPapyori)
        t_prime = evaluator.truncate_transactions(t, example_epsilon)
        self.assertTrue((len(t_prime)/len(t) >= 0.85))

    def test_recall_precision_f_score(self):
        correct = {"food1, food2" : 0.15, "food1, food3" : 0.24}
        private = {"food1, food2" : 0.17, "food3" : 0.17}
        test = evaluator.recall(private, correct)
        test2 = evaluator.precision(private, correct)
        test3 = evaluator.f_score(private, correct)
        self.assertTrue(test == 0.5)      
        self.assertTrue(test2 == 0.5)
        self.assertTrue(test3 == 0.5)  

class DP_Tester(unittest.TestCase):

    # Tester to confirm newly added functions in the apyori fork work as expected
    def test_geo_mech(self):
        example_epsilon = 0.5
        single = DPapyori.G(example_epsilon)
        self.assertTrue(type(single) == int)
        multiple = DPapyori.G(example_epsilon, 10)
        self.assertTrue(len(multiple) == 10)


def run_all_the_tests():
    classes = [Evaluator_Tester, DP_Tester]
    loader = unittest.TestLoader()

    suites_list = []
    for test_class in classes:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)
        
    big_suite = unittest.TestSuite(suites_list)

    runner = unittest.TextTestRunner()
    results = runner.run(big_suite)
    print(results)


if __name__ == '__main__':
    run_all_the_tests()#unittest.main()
