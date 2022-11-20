from qs-1 import train_test_split_test

def test_random_split_same():
    random_state_1 = 40
    random_state_2 = 40
    test_size = 0.2
    x_train1, x_test1, y_train1, y_test1 = train_test_split_test(test_size,random_state_1)
    x_train2, x_test2, y_train2, y_test2 = train_test_split_test(test_size,random_state_2)
    assert  x_train1 == x_train2
    assert  y_train1 == y_train2
    assert  x_test1 == x_test2
    assert  y_test1 == y_test2
    
    
    
def test_random_split_same():
    random_state_1 = 40
    random_state_2 = 30
    test_size = 0.2
    x_train1, x_test1, y_train1, y_test1 = train_test_split_test(test_size,random_state_1)
    x_train2, x_test2, y_train2, y_test2 = train_test_split_test(test_size,random_state_2)
    assert  x_train1 != x_train2
    assert  y_train1 != y_train2
    assert  x_test1 != x_test2
    assert  y_test1 != y_test2
    
