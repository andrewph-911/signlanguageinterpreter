from typing import List


def check_duplicate_list(mylist):
    if len(mylist) != len(set(mylist)):
        return
    else:
        return


l0 = [0, 1, 2]
print(check_duplicate_list(l0))

l1: list[str] = ['a', 'b', 'c', 'a']
print(check_duplicate_list(l1))
