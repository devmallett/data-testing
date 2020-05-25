# Finding k cloesest elements to a given value
# python k-mid.py


k = 4
X = 35

arr = [ 12, 16, 22, 30, 35, 39, 42, 45, 48, 50, 53, 55, 56 ]
low = arr[0]
def low_checker(k, X, arr, low):
    if k <= arr[low]:
        print("low")
    else:
        print("null")

low_checker(k, X, arr, low)