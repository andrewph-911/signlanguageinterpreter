
def check_palindrome(my_str):
    if len(my_str) < 1:
        return True
    else:
        if my_str[0] == my_str[-1]:
            return check_palindrome(my_str[1:-1])
        else:
            return False


if check_palindrome(labels[index]):
    print(labels[index])
else:
    print('false')