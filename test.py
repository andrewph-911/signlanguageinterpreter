def check_palindrome(my_str):
    if len(my_str) < 1:
        return True
    else:
        if my_str[0] != my_str[-1]:
            return check_palindrome(my_str[1:-1])
        else:
            return False

my_string = ['A','A', 'A', 'N', 'G']
if check_palindrome(my_string[1]) != True:
    print(my_string)
else:
    print(0)